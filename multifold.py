from torch.func import vmap
import gc
import time
import shutil
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn.functional import sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits
from tensordict import TensorDict

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.utils.nn.archs import MLP, Conv2dNN

# torchstrap dropped the skorch driver: EarlyStopping is now a plain callable used
# inside omnitrain.fit_ensemble, not passed to a removed `.fit(callbacks=...)`.
# from torchstrap.callbacks import EarlyStopping
from torchstrap.utils.data import TensorBatchSampler, random_split

from omnitrain import (
    fit_ensemble,
    # predict_ensemble,
    grad_and_loss,
    eval_loss,
    predict_proba,
    warmup_compiled,
    load_compile_cache,
    save_compile_cache,
    _input_to_device,
)

from dataset import (
    TensorDictDataset,
    get_stacked_batch_loader,
    classwise_undersample_and_split,
    build_input_transform,
)
from model_io import save_model_weights, save_ensemble_state, load_ensemble_state
from config import load_config, Config
from systematics import SysVar

# Chunk the vmap over replicas during inference (predict) to bound peak GPU
# memory for the CNN bin_counts route; None means no chunking (MLP routes).
# Set in __main__ from feature_mode.
_PREDICT_REPLICA_CHUNK = None


_LOG_MEM_LAST_T: float | None = None


def _log_mem(tag: str):
    """Print host anon-RSS (the OOM driver) + CUDA alloc + wall-clock delta at a
    phase boundary.

    The OmniFold OOM is host RAM (anon-rss ~40 GB on a 46 GB box), NOT the GPU or
    the memmap page cache (file-rss was 64 MB at kill). The static weight/sampler
    budget only accounts for ~18 GB, so this traces where the rest goes and whether
    it GROWS across iterations (a leak) or is a fixed per-phase peak. RssAnon comes
    straight from /proc/self/status (Linux); CUDA numbers are torch's allocator.

    `dt` is seconds since the previous marker; on CUDA we synchronize first so async
    GPU work is attributed to the phase that launched it (otherwise a "pause"
    between markers is just an unflushed kernel queue). Use it to find which silent
    inter-stage phase (reweight inference / CPU class-balancing / paging) is slow.
    """
    global _LOG_MEM_LAST_T
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    dt = 0.0 if _LOG_MEM_LAST_T is None else now - _LOG_MEM_LAST_T
    _LOG_MEM_LAST_T = now

    anon = rss = -1.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) / 1e6  # kB -> GB
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1e6
    except OSError:
        pass
    cu = ""
    if torch.cuda.is_available():
        cu = (
            f" | cuda alloc={torch.cuda.memory_allocated() / 1e9:.2f}"
            f" reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )
    print(
        f"[mem] {tag:28s} dt={dt:7.1f}s anon={anon:6.2f} rss={rss:6.2f} GB{cu}",
        flush=True,
    )


def _release_heap():
    """Return freed-but-retained heap pages to the OS at iteration end.

    The OmniFold loop cats/clones large `(R, n)` f32 weight tensors every
    iteration; even after Python frees them, glibc's arena holds the medium
    (<32 MB) blocks instead of returning them, so anon-RSS ratchets up ~1 GB/iter
    (a fragmentation artifact, not a true leak — the multi-GB tensors are mmap'd
    and already returned on free). `gc.collect()` drops any cycle-held refs, then
    `malloc_trim(0)` hands the arena's free top back to the kernel. No-op on
    non-glibc libc. Cheap (~ms) and runs once per iteration.
    """
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError, AttributeError:
        pass


def reweight_inference_loaders(
    data: TensorDictDataset,
    batch_size: int = 32,
    num_replicas: int = 2,
    drop_last: bool = False,
    load_only_first=None,
    has_unmatched: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    mp_method: str = "thread",
):
    all_indices = torch.arange(len(data))
    neg_mask = data.target < 0.5
    neg_indices = all_indices[neg_mask]

    if not has_unmatched:
        sampler = TensorBatchSampler(
            # Inference uses identical indices for every replica (just memory
            # chunking), so broadcast a view instead of cloning N copies — a
            # (N, n_jets) int64 stack is ~3.7 GB for the 11.7M missed gen jets.
            neg_indices.unsqueeze(0).expand(num_replicas, -1),
            # Inference is just memory chunking; never drop indices. Clamp the
            # batch to the class size so a small class (e.g. B-side fakes in an
            # AB-split closure) still yields one partial batch.
            batch_size=min(batch_size, neg_indices.shape[0]),
            batch_dim=1,
            drop_last=drop_last,
        )
        return get_stacked_batch_loader(
            data,
            sampler,
            load_only_first=load_only_first,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mp_method=mp_method,
        )

    is_matched = data.td["is_matched"][data.indices][neg_mask]
    matched_indices = neg_indices[is_matched == 1]
    unmatched_indices = neg_indices[is_matched == 0]

    matched_sampler = TensorBatchSampler(
        matched_indices.unsqueeze(0).expand(num_replicas, -1),
        batch_size=min(batch_size, matched_indices.shape[0]),
        batch_dim=1,
        drop_last=drop_last,
    )
    matched_loader = get_stacked_batch_loader(
        data,
        matched_sampler,
        load_only_first=load_only_first,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mp_method=mp_method,
    )

    unmatched_sampler = TensorBatchSampler(
        unmatched_indices.unsqueeze(0).expand(num_replicas, -1),
        # B-side fakes can be < batch_size in AB-split closure; clamp so all
        # fakes are still covered in a single batch rather than erroring.
        batch_size=min(batch_size, unmatched_indices.shape[0]),
        batch_dim=1,
        drop_last=drop_last,
    )
    unmatched_loader = get_stacked_batch_loader(
        data,
        unmatched_sampler,
        load_only_first=load_only_first,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mp_method=mp_method,
    )

    return (matched_loader, unmatched_loader)


def train_test_multi_loaders(
    data: TensorDictDataset,
    *,
    num_replicas: int,
    batch_size: int,
    train_size: float = 0.5,
    undersample_size: float | int = 1.0,
    stratifys: tuple[bool, bool] = (False, False),
    vmap_randomness: str = "different",
    vmap_in_dims: int | tuple = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    load_only_first: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    mp_method: str = "thread",
):
    pos_mask = pos_mask or data.target > 0.5
    neg_mask = neg_mask or data.target < 0.5

    all_indices = torch.arange(len(data))
    pos_indices = all_indices[pos_mask]
    neg_indices = all_indices[neg_mask]

    print(
        f"--- Undersampling and Train-Test splitting {pos_indices.shape[0]} positive samples and "
        f"{neg_indices.shape[0]} negative samples, with {num_replicas} replicas."
    )

    pos_stratify = data.td["pth_bin"][data.indices][pos_mask] if stratifys[0] else None
    neg_stratify = data.td["pth_bin"][data.indices][neg_mask] if stratifys[1] else None

    stacked_train_indices, stacked_valid_indices = classwise_undersample_and_split(
        pos_indices,
        neg_indices,
        undersample_size=undersample_size,
        split_sizes=(train_size, 1.0 - train_size),
        pos_stratify=pos_stratify,
        neg_stratify=neg_stratify,
        num_replicas=num_replicas,
        vmap_randomness=vmap_randomness,
        vmap_in_dims=vmap_in_dims,
        generator=generator,
    )

    print(
        "------ Subsampled "
        f"{str(undersample_size * 100) + '%' if isinstance(undersample_size, float) else 2 * undersample_size}, "
        f"with Train size: {stacked_train_indices.shape} and Valid size: {stacked_valid_indices.shape}"
    )

    train_sampler = TensorBatchSampler(
        stacked_train_indices,
        batch_size=batch_size,
        batch_dim=1,
        drop_last=drop_last,
    )
    valid_sampler = TensorBatchSampler(
        stacked_valid_indices,
        batch_size=batch_size,
        batch_dim=1,
        drop_last=drop_last,
    )

    train_loader = get_stacked_batch_loader(
        data,
        train_sampler,
        load_only_first=load_only_first,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mp_method=mp_method,
    )
    valid_loader = get_stacked_batch_loader(
        data,
        valid_sampler,
        load_only_first=load_only_first,
        num_workers=num_workers,
        pin_memory=pin_memory,
        mp_method=mp_method,
    )

    return train_loader, valid_loader


def read_datasets(
    detlvl_src: Path,
    partlvl_src: Path,
    *,
    num_replicas: int = 1,
    a_size=0.5,
    mode: Literal["normal", "ab_closure"] = "normal",
    generator: torch.Generator | None = None,
    out_path: Path | None = None,
):
    detlvl_td = TensorDict.load_memmap(detlvl_src)
    print("Loaded (data, reco) TensorDict from:", detlvl_src)

    partlvl_td = TensorDict.load_memmap(partlvl_src)
    print("Loaded particle level TensorDict from:", partlvl_src)

    if mode == "ab_closure":
        _detlvl_all = torch.arange(len(detlvl_td), dtype=torch.long)
        _detlvl_matched_indices = _detlvl_all[detlvl_td["is_matched"] == 1]
        _detlvl_fake_indices = _detlvl_all[detlvl_td["is_matched"] == 0]

        _partlvl_all = torch.arange(len(partlvl_td))
        _partlvl_matched_indices = _partlvl_all[partlvl_td["is_matched"] == 1]

        _partlvl_missed_indices = _partlvl_all[partlvl_td["is_matched"] == 0]

        (
            _detlvl_matched_indices_a,
            _partlvl_matched_indices_a,
            _detlvl_matched_indices_b,
            _partlvl_matched_indices_b,
        ) = random_split(
            _detlvl_matched_indices,
            _partlvl_matched_indices,
            sizes=(a_size, 1.0 - a_size),
            stratify=detlvl_td["pth_bin"][_detlvl_matched_indices],
        )

        _detlvl_fake_indices_a, _detlvl_fake_indices_b = random_split(
            _detlvl_fake_indices,
            sizes=(a_size, 1.0 - a_size),
            stratify=detlvl_td["pth_bin"][_detlvl_fake_indices],
        )

        _partlvl_missed_indices_a, _partlvl_missed_indices_b = random_split(
            _partlvl_missed_indices,
            sizes=(a_size, 1.0 - a_size),
            stratify=partlvl_td["pth_bin"][_partlvl_missed_indices],
        )

        _detlvl_a_indices = torch.cat(
            [_detlvl_matched_indices_a, _detlvl_fake_indices_a], dim=-1
        )

        _detlvl_b_indices = torch.cat(
            [_detlvl_matched_indices_b, _detlvl_fake_indices_b], dim=-1
        )

        _partlvl_a_indices = torch.cat(
            [_partlvl_matched_indices_a, _partlvl_missed_indices_a], dim=-1
        )

        _partlvl_b_indices = torch.cat(
            [_partlvl_matched_indices_b, _partlvl_missed_indices_b], dim=-1
        )

        detlvl_ds = TensorDictDataset(
            detlvl_td,
            is_categorical=True,
            indices=torch.cat([_detlvl_a_indices, _detlvl_b_indices], dim=-1),
            target=torch.cat(
                [
                    torch.ones_like(_detlvl_a_indices),
                    torch.zeros_like(_detlvl_b_indices),
                ],
                dim=-1,
            ),
            num_replicas=num_replicas,
        )

        partlvl_ds = TensorDictDataset(
            partlvl_td,
            is_categorical=True,
            indices=torch.cat([_partlvl_b_indices, _partlvl_b_indices], dim=-1),
            target=torch.cat(
                [
                    torch.ones_like(_partlvl_b_indices),
                    torch.zeros_like(_partlvl_b_indices),
                ],
                dim=-1,
            ),
            num_replicas=num_replicas,
        )

        detlvl_matched_indices = _detlvl_matched_indices_b
        detlvl_fake_indices = _detlvl_fake_indices_b

        partlvl_matched_indices = _partlvl_matched_indices_b
        partlvl_missed_indices = _partlvl_missed_indices_b

        if out_path is not None:
            with (out_path / "index_split.npz").open("wb") as outf:
                np.savez(
                    outf,
                    detlvl_matched_indices=detlvl_matched_indices.detach().numpy(),
                    partlvl_matched_indices=partlvl_matched_indices.detach().numpy(),
                    detlvl_fake_indices=detlvl_fake_indices.detach().numpy(),
                    partlvl_missed_indices=partlvl_missed_indices.detach().numpy(),
                )

    else:
        detlvl_ds = TensorDictDataset(
            detlvl_td, is_categorical=True, num_replicas=num_replicas
        )
        partlvl_ds = TensorDictDataset(
            partlvl_td, is_categorical=True, num_replicas=num_replicas
        )

        detlvl_matched_indices = detlvl_ds.indices[detlvl_td["is_matched"] == 1]
        detlvl_fake_indices = detlvl_ds.indices[detlvl_td["is_matched"] == 0]

        partlvl_matched_indices = partlvl_ds.indices[partlvl_td["is_matched"] == 1]
        partlvl_missed_indices = partlvl_ds.indices[partlvl_td["is_matched"] == 0]

    fake_scaling_indices = torch.cat([detlvl_matched_indices] * 2)
    fake_scaling_targets = torch.cat(
        [
            torch.ones_like(detlvl_matched_indices, dtype=torch.float32),
            torch.zeros_like(detlvl_matched_indices, dtype=torch.float32),
        ]
    )

    miss_scaling_indices = torch.cat([partlvl_matched_indices] * 2)
    miss_scaling_targets = torch.cat(
        [
            torch.ones_like(partlvl_matched_indices, dtype=torch.float32),
            torch.zeros_like(partlvl_matched_indices, dtype=torch.float32),
        ]
    )

    fake_scaling_ds = TensorDictDataset(
        detlvl_td,
        indices=fake_scaling_indices,
        target=fake_scaling_targets,
        sample_weight=torch.ones_like(fake_scaling_indices, dtype=torch.float32),
        is_categorical=True,
        num_replicas=num_replicas,
    )

    miss_scaling_ds = TensorDictDataset(
        partlvl_td,
        indices=miss_scaling_indices,
        target=miss_scaling_targets,
        sample_weight=torch.ones_like(miss_scaling_indices, dtype=torch.float32),
        is_categorical=True,
        num_replicas=num_replicas,
    )

    return (
        detlvl_ds,
        partlvl_ds,
        fake_scaling_ds,
        miss_scaling_ds,
        detlvl_matched_indices,
        partlvl_matched_indices,
        detlvl_fake_indices,
        partlvl_missed_indices,
    )


def _eps(x: torch.Tensor):
    return x[x > 0.0].min() * torch.finfo(x.dtype).eps


def _weight_stats(w: torch.Tensor, *, clamp_min: float, clamp_max: float):
    """Per-replica summary statistics for a (num_replicas, N) weight tensor.

    Processed one replica row at a time so transient memory stays bounded
    even when N ~ 1.5e7 — a single (R, N) float64 promotion of the
    renormalised gen weights would alone exceed ~2 GiB.
    """
    w = w.detach()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    R = w.shape[0]
    keys = (
        "mean",
        "std",
        "min",
        "max",
        "median",
        "sum",
        "ess",
        "frac_at_clamp_min",
        "frac_at_clamp_max",
    )
    out = {k: np.empty(R, dtype=np.float64) for k in keys}
    for r in range(R):
        wr = w[r].to(torch.float64)
        s = wr.sum()
        s2 = (wr * wr).sum()
        out["mean"][r] = float(wr.mean())
        out["std"][r] = float(wr.std())
        out["min"][r] = float(wr.min())
        out["max"][r] = float(wr.max())
        out["median"][r] = float(wr.median())
        out["sum"][r] = float(s)
        out["ess"][r] = float((s * s) / s2.clamp_min(1e-30))
        out["frac_at_clamp_min"][r] = float((wr <= clamp_min * 1.001).double().mean())
        out["frac_at_clamp_max"][r] = float((wr >= clamp_max * 0.999).double().mean())
        del wr
    return out


def _save_unfolding_weights(path, gen_weights, reco_weights):
    """Write one OmniFold iteration's reweights to a standalone ``.npz``
    (``arr_0`` = gen, ``arr_1`` = reco).

    Streaming one file per iteration instead of accumulating the whole history
    keeps host RAM flat: at R=40 each (R, n_jets) array is ~2.3 GB and the old
    ``w_unfolding`` list grew to ~18 GB by the last iteration — a primary driver
    of the host-RAM OOM. ``np.savez`` writes each array straight to the file
    (``.numpy()`` is a zero-copy view of the CPU tensor; ``tofile`` streams it,
    no large transient copy). Concatenating the per-iteration files in order
    (``niter0`` = priors, then ``niter1..N``) reproduces the old combined layout
    ``[part_prior, det_prior, (gen, reco) * n_iter]``.
    """
    np.savez(
        path,
        gen_weights.detach().cpu().numpy(),
        reco_weights.detach().cpu().numpy(),
    )


def _reassemble_unfolding_weights(out_dir: Path, num_iterations: int) -> Path | None:
    """Combine the streamed ``w_unfolding_niter{i}.npz`` files (each ``arr_0`` = gen,
    ``arr_1`` = reco) into the single legacy ``w_unfolding.npz`` the downstream scripts
    (``histograms.py``, ``plot_closure.py``, ``plot_ab_closure.py``) expect, whose
    layout is ``arr_{2i}`` = gen / ``arr_{2i+1}`` = reco for iteration ``i`` (i = 0 is
    the prior, then ``1..num_iterations``).

    Done as a final pass so the run itself never holds the full history. To stay
    RAM-flat here too, write the combined ``.npz`` (a ZIP of ``arr_<k>.npy`` members)
    member-by-member with ``zipfile`` + ``numpy.lib.format.write_array``, loading only
    ONE per-iteration file (gen+reco, ~2.8 GB at R=40) at a time — never the ~18 GB
    that the old in-RAM ``w_unfolding`` list (and a plain ``np.savez(*all)``) would.
    Returns the combined path, or ``None`` if no per-iteration files were found.
    """
    import zipfile
    import numpy.lib.format as npformat

    files = [
        out_dir / f"w_unfolding_niter{i}.npz" for i in range(num_iterations + 1)
    ]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"[reassemble] no w_unfolding_niter*.npz under {out_dir}; skipping.")
        return None

    # Prune stale higher-index per-iteration files left by a prior, longer run. The
    # caller passes the count actually completed (which, on an early stop, is < the
    # config `num_iterations`), so any `niter{i>num_iterations}.npz` on disk is from a
    # previous run. Without this, an early-stopped run silently splices those stale
    # gen/reco blocks onto its own iters 0..num_iterations in w_unfolding.npz.
    for stale in out_dir.glob("w_unfolding_niter*.npz"):
        try:
            idx = int(stale.stem.removeprefix("w_unfolding_niter"))
        except ValueError:
            continue
        if idx > num_iterations:
            print(f"[reassemble] removing stale {stale.name} (idx {idx} > {num_iterations}).")
            stale.unlink()

    combined = out_dir / "w_unfolding.npz"
    slot = 0
    # ZIP_STORED (no compression) matches np.savez; allowZip64 + per-member
    # force_zip64 since each (40, ~14.5M) f32 gen array is ~2.3 GB (> the 2 GB
    # non-zip64 member limit).
    with zipfile.ZipFile(combined, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in files:
            with np.load(f) as z:
                for key in ("arr_0", "arr_1"):  # gen, then reco
                    with zf.open(f"arr_{slot}.npy", "w", force_zip64=True) as dest:
                        npformat.write_array(dest, np.ascontiguousarray(z[key]))
                    slot += 1
    print(
        f"[reassemble] wrote {combined} ({slot} arrays from {len(files)} iteration files)."
    )
    return combined


# Names of the four classifier ensembles, used for the per-iteration resume
# checkpoint files (one .pt per ensemble) under embedding/<sysvar>/checkpoints/.
_CKPT_ENSEMBLE_NAMES = ("detlvl", "miss_scaling", "partlvl", "fake_scaling")


def _find_latest_complete_checkpoint(ckpt_dir: Path) -> int | None:
    """Highest N for which ``ckpt_dir/iter{N}/complete.marker`` exists, else None.

    The marker is written last by ``_save_resume_checkpoint`` so a half-written
    (crashed mid-write) checkpoint dir is ignored here.
    """
    if not ckpt_dir.is_dir():
        return None
    best: int | None = None
    for d in ckpt_dir.glob("iter*"):
        if not (d / "complete.marker").exists():
            continue
        try:
            n = int(d.name[len("iter") :])
        except ValueError:
            continue
        if best is None or n > best:
            best = n
    return best


def _save_resume_checkpoint(
    ckpt_dir: Path,
    iteration: int,
    *,
    w_matched,
    w_miss,
    w_fake,
    prev_gen,
    prev_reco,
    stacked_stats,
    states,
    keep_last_only: bool,
) -> None:
    """Persist everything needed to warm-restart at the iteration AFTER ``iteration``.

    Writes to ``ckpt_dir/iter{iteration+1}/``: the raw cumulative component weights
    + rolling priors (``weights.npz``), the stacked weight-stat history
    (``stats.npz``), and the FULL state of each ensemble (``<name>.pt`` via
    ``save_ensemble_state`` — params + buffers + Adam moments + active_mask). The
    ``complete.marker`` is written LAST so an interrupted write is not trusted.
    ``states`` is a ``{name: State}`` dict; all four are on CPU and post-``rearm``
    at the iteration boundary, exactly the snapshot a continuous run would carry in.
    """
    n = iteration + 1
    d = ckpt_dir / f"iter{n}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / "weights.npz",
        w_matched=w_matched.detach().cpu().numpy(),
        w_miss=w_miss.detach().cpu().numpy(),
        w_fake=w_fake.detach().cpu().numpy(),
        prev_gen=prev_gen.detach().cpu().numpy(),
        prev_reco=prev_reco.detach().cpu().numpy(),
    )
    np.savez(d / "stats.npz", **stacked_stats)
    for name, state in states.items():
        save_ensemble_state(state, d / f"{name}.pt")
    (d / "complete.marker").write_text(f"iteration {n} complete\n")

    if keep_last_only:
        # Prune older checkpoints only AFTER the new one is fully committed, so a
        # crash never leaves zero usable checkpoints. Bounds storage to one
        # iteration's worth (full state x4 ensembles x num_replicas).
        for other in ckpt_dir.glob("iter*"):
            if other.is_dir() and other.name != f"iter{n}":
                shutil.rmtree(other, ignore_errors=True)


@torch.no_grad()
def _load_resume_checkpoint(ckpt_dir: Path, n: int, *, states):
    """Restore the iteration-boundary state written by ``_save_resume_checkpoint``.

    Returns ``(w_matched, w_miss, w_fake, prev_gen, prev_reco,
    weight_stats_history)`` (tensors on CPU — the loop pages them as usual) and
    loads each ensemble in ``states`` in place via ``load_ensemble_state``.
    """
    d = ckpt_dir / f"iter{n}"
    wz = np.load(d / "weights.npz")
    w_matched = torch.as_tensor(wz["w_matched"])
    w_miss = torch.as_tensor(wz["w_miss"])
    w_fake = torch.as_tensor(wz["w_fake"])
    prev_gen = torch.as_tensor(wz["prev_gen"])
    prev_reco = torch.as_tensor(wz["prev_reco"])

    sz = np.load(d / "stats.npz")
    n_done = next(iter(sz.values())).shape[0] if sz.files else 0
    weight_stats_history = [{k: sz[k][i] for k in sz.files} for i in range(n_done)]

    for name, state in states.items():
        load_ensemble_state(state, d / f"{name}.pt")

    return w_matched, w_miss, w_fake, prev_gen, prev_reco, weight_stats_history


def rearm_replicas(state):
    # Re-activate every replica that EarlyStopping froze (set active_mask=False)
    # in the previous fit, so the next OmniFold iteration trains the full
    # ensemble again. The frozen `State` forbids *rebinding* active_mask and no
    # longer exposes `num_replicas`; fill the device-resident (N,) mask in place.
    # state.active_mask = torch.ones(
    #     state.num_replicas, dtype=torch.bool, device=state.device
    # )
    state.active_mask.fill_(True)


def build_classifier(input_for_transform, layer_sizes, cfg, device, optimizer_kwargs):
    """Init a per-step ensemble: a 2D-CNN for bin_counts, else the MLP.

    `input_for_transform` is the (subset of) the dataset input used to fit the
    z-norm / per-channel transform. For bin_counts the input is a (C, H, W) image
    so `in_channels` comes from `shape[-3]`; layer_sizes is ignored there.
    Returns the `(module, state)` 2-tuple from `StatelessModule.init` (the
    refactor dropped the separate optimizer object; the optimizer state lives in
    `state.optimizer_state`).
    """
    if not isinstance(cfg, Config):
        cfg = Config(cfg)
    transform = build_input_transform(
        cfg["input_transform"], input_for_transform, device=device
    )
    common = dict(
        dropout_prob=cfg.dropout_prob,
        input_transform=transform,
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )
    if cfg.get("feature_mode") == "bin_counts":
        # collapse_spatial: convs reduce the (H, W) grid to 1×1 by the last layer
        # (valid convs, kernels derived from input_size) instead of preserving it +
        # global avg-pool. cnn_channels must be deep enough to reach 1×1.
        module, state = StatelessModule.init(
            Conv2dNN,
            Adam,
            in_channels=input_for_transform.shape[-3],
            conv_channels=cfg.cnn_channels,
            head_sizes=(1,),
            input_size=tuple(input_for_transform.shape[-2:]),
            collapse_spatial=cfg.cnn_collapse,
            **common,
        )
        for k, v in state.params_dict.items():
            if v.ndim == 5:
                state.params_dict[k] = v.to(memory_format=torch.channels_last_3d)
        return module, state
    return StatelessModule.init(MLP, Adam, layer_sizes=layer_sizes, **common)


@torch.inference_mode()
def reweigh_samples(
    model,
    state,
    loader,
    out,
    *,
    init_offset=0,
    forward=None,
    eps=None,
    clamp_min=1e-3,
    clamp_max=1e3,
    chunk_size=None,
):
    """Fold the per-jet OmniFold reweight into ``out`` IN PLACE, streamed over ``loader``.

    No gather / ``torch.cat`` / return: each batch's reweight multiplies the matching
    ``out[..., offset:offset+b]`` slice and is then discarded, so peak memory is one
    batch of probabilities rather than the full ``(R, n)`` reweight tensor. ``out`` is
    the cumulative weight buffer to update in place — ``w_matched`` / ``w_miss`` /
    ``w_fake``, or a dataset's contiguous positive-class weight block. ``init_offset``
    is where this loader's first jet lands in ``out`` (0 when ``out`` is exactly this
    loader's block). The loader must visit jets in the same column order as ``out``
    (the ordering the old ``w_* *= rewt`` already relied on).

    ``forward`` is the prebuilt-once reweighting forward (predict_proba: sigmoid +
    vmap + replica chunk, optionally bf16-autocast/compiled for the CNN route),
    constructed once per ensemble and reused across every OmniFold iteration so a
    torch.compile cache is warmed a single time. ``rewt`` lands on ``state.device``
    (GPU); it is moved to ``out.device`` (the weights live on CPU) before the multiply.
    """
    model.eval()
    if forward is None:
        forward = predict_proba(model, chunk_size=chunk_size)

    offset = init_offset
    for batch in loader:
        x = _input_to_device(batch[0], state)
        pdata = forward(state.params_dict, state.buffers_dict, x)
        preco = 1.0 - pdata
        if eps is None:
            eps = _eps(preco)  # computed once (first batch), reused after
        rewt = (pdata / preco.add_(eps)).squeeze_(-1)
        if clamp_min is not None or clamp_max is not None:
            rewt.clamp_(min=clamp_min, max=clamp_max)  # in place; do NOT return here

        size = rewt.shape[-1]
        out[..., offset : offset + size].mul_(rewt.to(out.device))
        offset = offset + size


def _reset_optimizer_lr(*states, base_lr):
    """Tier 0.1: reset each ensemble's per-replica Adam LR to the config base. The
    ``LRScheduler`` lazily captures its ``base_lrs_`` from ``state.optimizer_state["lr"]``,
    so without this the annealed LR carries across warm-started OmniFold iterations and
    ratchets down run-wide to ``min_lr`` (dead by ~iter4). Cheap; works on CPU or GPU."""
    for s in states:
        s.optimizer_state["lr"].fill_(float(base_lr))


def _cold_restart_state(state, build_args, *, base_lr):
    """Tier 0.2 (opt-in): re-initialize an ensemble in place — fresh params, zeroed Adam
    moments, base LR — keeping the module/state object identity so the prebuilt
    grad/forward closures (and any compile cache) stay valid. ``build_args`` is the
    ``(input, layer_sizes, cfg, device, optimizer_kwargs)`` tuple originally passed to
    ``build_classifier``; the fresh model is built on the same device and discarded after
    its tensors are copied in. NOTE: ``build_classifier`` also re-fits the input transform
    (deterministic, but not free) — acceptable for an opt-in path."""
    _, fresh = build_classifier(*build_args)
    for k, v in fresh.params_dict.items():
        state.params_dict[k].copy_(v.to(state.params_dict[k].device))
    for k in ("exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs", "state_steps"):
        t = state.optimizer_state.get(k)
        if torch.is_tensor(t):
            t.zero_()
    state.optimizer_state["lr"].fill_(float(base_lr))
    rearm_replicas(state)
    return state


def _best_valid_median(history):
    """Median-over-replicas of the per-replica best (min-over-epochs) valid loss, for the
    Tier 1 convergence stop. ``history["valid_loss"]`` is a per-epoch list of (N,) host
    tensors."""
    vl = history.get("valid_loss", [])
    if not vl:
        return float("nan")
    stk = torch.stack([torch.as_tensor(v).reshape(-1).float() for v in vl])  # (E, N)
    return float(stk.min(dim=0).values.median())


def run(cfg: Config) -> None:
    # Entry point for the OmniFold unfolding run. Body is the former
    # `if __name__ == "__main__"` block, factored into a callable so it can be
    # driven from a marimo notebook (molab cloud GPU) as well as the CLI script
    # path below. `_PREDICT_REPLICA_CHUNK` is a module global read by
    # `get_sample_reweights`; declare it global so the per-run assignment in the
    # CNN branch rebinds the module-level name instead of creating a local.
    global _PREDICT_REPLICA_CHUNK

    # TF32 matmul: notable speedup on Ampere+ for the vmapped fp32 classifiers, at
    # a ~1e-3 relative shift in the unfolded weights (accepted for this ensemble).
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    generator = torch.Generator().manual_seed(cfg["dataseed"])

    sys_var = cfg.sys_var
    feature_mode = cfg.get("feature_mode", "angularities")

    # Optional torch.compile of the train/valid vmapped forward(+grad). When on,
    # keep the per-batch shape static so torch.compile doesn't recompile on the
    # smaller trailing batch — drop the partial train/valid batch. The reweighting
    # predict() path is not compiled (it uses its own vmap), so its loaders keep
    # full coverage and are left untouched.
    compile_forward = cfg.compile_forward
    train_drop_last = compile_forward

    # The bin_counts route is a vmapped 2D-CNN; chunk predict() over replicas to
    # bound peak GPU memory during inference reweighting.
    is_cnn = feature_mode == "bin_counts"
    if is_cnn:
        _PREDICT_REPLICA_CHUNK = cfg.predict_replica_chunk

    # Inference (reweighting) batch. The MLP route is cheap, so keep big batches
    # (batch_size*5). For the CNN route, match the *training* batch so unchunked
    # forward-only inference is a strict subset of the training memory profile (same
    # replicas + batch, but inference_mode frees activations the backward would
    # retain) — it therefore fits wherever training fit, letting predict_replica_chunk
    # be null (no chunking) for a much faster single parallel 40-replica vmap instead
    # of 10 serial chunks. Set predict_replica_chunk back to a small int to re-bound
    # peak memory on a smaller GPU (the 4070), in which case bump this multiplier.
    reweight_batch = cfg["batch_size"] * (1 if is_cnn else 5)
    # --- old: 2x for CNN, relying on chunk=4 to bound memory ---
    # reweight_batch = cfg["batch_size"] * (2 if is_cnn else 5)

    # Threaded prefetch for the memmap gather — OPT-IN, default OFF, and currently
    # NOT recommended: the torchdata Loader leaks worker threads across per-epoch
    # iterators, so anon RAM grows unboundedly over a route's epochs (~37 GB seen),
    # evicts the page cache, and thrashes disk (miss_scaling degraded 9 → 146 s/epoch).
    # Even leak-free it only helps a genuinely disk-bound route; the right fix for
    # the I/O-bound 18.8 GB part-level input is to shrink it (uint8/fp16) so it stays
    # cached. Leave at 0 unless the leak is fixed AND a run is truly disk-bound.
    loader_workers = cfg.get("num_workers", 0)

    # UNFOLDING_PRIOR_SAME does the AB-split closure in-memory on the
    # nominal tensordicts; no separate on-disk artefact is materialised
    # for it, so redirect the source path back to nominal.
    root_dir = cfg.features_root
    src = (
        root_dir
        / "tensordicts"
        / str(SysVar.NONE if sys_var == SysVar.UNFOLDING_PRIOR_SAME else sys_var)
    )
    print("Reading files from:", src)

    # out_dir = Path("./outputs") / f"unfolding_{sys_var_dir}" / cfg["feature_mode"]
    out_dir = root_dir / "embedding" / str(sys_var)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        detlvl_ds,
        partlvl_ds,
        fake_scaling_ds,
        miss_scaling_ds,
        detlvl_matched_indices,
        partlvl_matched_indices,
        detlvl_fake_indices,
        partlvl_missed_indices,
    ) = read_datasets(
        src / "det_lvl",
        src / "part_lvl",
        num_replicas=cfg["num_replicas"],
        mode=("ab_closure" if sys_var == SysVar.UNFOLDING_PRIOR_SAME else "normal"),
        out_path=(out_dir if sys_var == SysVar.UNFOLDING_PRIOR_SAME else None),
    )

    detlvl_train_loader, detlvl_valid_loader = train_test_multi_loaders(
        detlvl_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        num_workers=loader_workers,
        drop_last=train_drop_last,
        stratifys=(False, True),
    )
    fake_scaling_train_loader, fake_scaling_valid_loader = train_test_multi_loaders(
        fake_scaling_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        num_workers=loader_workers,
        drop_last=train_drop_last,
        stratifys=(True, True),
    )
    reco_match_loader, reco_fake_loader = reweight_inference_loaders(
        detlvl_ds,
        batch_size=reweight_batch,
        num_replicas=cfg["num_replicas"],
        has_unmatched=True,
        # Synchronous (num_workers=0). The threaded Loader leaks worker threads
        # per iterator — anon ballooned to ~37 GB over a route's epochs, evicting
        # the page cache and thrashing disk (~146 s/epoch). 0 spawns no threads,
        # keeps the reweight output order trivially correct, and buffers nothing.
        num_workers=0,
    )
    print(
        "Initialized loaders to get"
        " f : p(reco) --> p(data) and"
        " g : p(reco fake) --> p(reco match)"
    )

    partlvl_train_loader, partlvl_valid_loader = train_test_multi_loaders(
        partlvl_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        num_workers=loader_workers,
        drop_last=train_drop_last,
        stratifys=(True, True),
    )
    miss_scaling_train_loader, miss_scaling_valid_loader = train_test_multi_loaders(
        miss_scaling_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        num_workers=loader_workers,
        drop_last=train_drop_last,
        stratifys=(True, True),
    )
    gen_match_loader, gen_miss_loader = reweight_inference_loaders(
        partlvl_ds,
        batch_size=reweight_batch,
        num_replicas=cfg["num_replicas"],
        has_unmatched=True,
        # Synchronous (num_workers=0). The threaded Loader leaks worker threads
        # per iterator — anon ballooned to ~37 GB over a route's epochs, evicting
        # the page cache and thrashing disk (~146 s/epoch). 0 spawns no threads,
        # keeps the reweight output order trivially correct, and buffers nothing.
        num_workers=0,
    )
    print(
        "Initialized loaders to get"
        " f : p(gen) --> p(unfolded) and"
        " g : p(gen miss) --> p(gen match)"
    )

    n_data = detlvl_ds.target.gt(0.5).sum()
    n_reco = detlvl_ds.target.lt(0.5).sum()
    n_gen = partlvl_ds.target.lt(0.5).sum()

    print("Number of data jets:", n_data)
    print("Number of gen jets, reco jets:", n_gen, n_reco)

    num_features = detlvl_ds.td["input"].shape[-1]
    layer_sizes = cfg.layer_sizes(num_features)
    device = cfg.device

    optimizer_kwargs = cfg.optimizer_kwargs

    torch.manual_seed(cfg["modelseed"])

    # StatelessModule.init now returns a 2-tuple (module, state) (was 3-tuple
    # with a separate optimizer object).
    detlvl_ensemble, detlvl_state = build_classifier(
        detlvl_ds.input, layer_sizes, cfg, device, optimizer_kwargs
    )

    miss_scaling_ensemble, miss_scaling_state = build_classifier(
        partlvl_ds.td["input"][partlvl_matched_indices],
        layer_sizes,
        cfg,
        device,
        optimizer_kwargs,
    )

    partlvl_ensemble, partlvl_state = build_classifier(
        partlvl_ds.input, layer_sizes, cfg, device, optimizer_kwargs
    )

    fake_scaling_ensemble, fake_scaling_state = build_classifier(
        detlvl_ds.td["input"][detlvl_matched_indices],
        layer_sizes,
        cfg,
        device,
        optimizer_kwargs,
    )

    # Tier 0.2 (opt-in): capture the per-ensemble build args so each iteration can cold-
    # restart its classifier (fresh params + zeroed Adam). Only materialize the matched-
    # index input slices when the flag is on (indexing allocates).
    _cold_start = bool(cfg.get("cold_start_per_iter", False))
    _base_lr = float(cfg.optimizer_kwargs["lr"])
    if _cold_start:
        detlvl_build = (detlvl_ds.input, layer_sizes, cfg, device, optimizer_kwargs)
        miss_build = (
            partlvl_ds.td["input"][partlvl_matched_indices],
            layer_sizes, cfg, device, optimizer_kwargs,
        )
        partlvl_build = (partlvl_ds.input, layer_sizes, cfg, device, optimizer_kwargs)
        fake_build = (
            detlvl_ds.td["input"][detlvl_matched_indices],
            layer_sizes, cfg, device, optimizer_kwargs,
        )

    # Build the manual-loop drivers ONCE per ensemble and reuse them across every
    # OmniFold iteration (model + transform are fixed; only weights/data-weights
    # change), so a `torch.compile` cache is warmed a single time. Replaces the
    # removed StatelessModule.compile / .fit. The predict path is intentionally
    # NOT compiled (its inference loaders keep full coverage -> variable trailing
    # batch -> would recompile); it runs the chunked vmap under inference_mode.
    # if compile_forward:
    #     print("Compiling classifier forward passes (torch.compile)...")
    #     for _ensemble in (detlvl_ensemble, ...):
    #         _ensemble.compile(**cfg.compile_kwargs)
    if compile_forward:
        print("Compiling classifier train/valid forward passes (torch.compile)...")
    _ensembles = (
        detlvl_ensemble,
        miss_scaling_ensemble,
        partlvl_ensemble,
        fake_scaling_ensemble,
    )
    # grad_and_loss bakes in the sample-weighted BCE (omnitrain.evaluate); no criterion
    # arg. eval_loss keeps the criterion slot for signature symmetry but also uses
    # evaluate, so pass the same BCE for documentation.
    # bf16 autocast on the CNN train/valid fwd+bwd: halves the activation memory the
    # backward retains (unblocks the large batch on the 8 GB GPU) and ~2x the launch-
    # bound grouped conv. MLP route stays fp32. Mirrors predict_fwd's is_cnn gating.
    train_autocast = torch.bfloat16 if is_cnn else None
    grad_loss = {
        e: grad_and_loss(
            e,
            compile=compile_forward,
            compile_kwargs=cfg.compile_kwargs,
            autocast_dtype=train_autocast,
        )
        for e in _ensembles
    }
    valid_loss = {
        e: eval_loss(
            e,
            binary_cross_entropy_with_logits,
            compile=compile_forward,
            compile_kwargs=cfg.compile_kwargs,
            autocast_dtype=train_autocast,
        )
        for e in _ensembles
    }

    # Reweighting (inference) forward, built ONCE per ensemble and reused across all
    # OmniFold iterations (model + transform are fixed). For the CNN route this is the
    # dominant, previously-unoptimized cost: bench_reweight.py shows bf16 autocast +
    # torch.compile(dynamic) + replica chunking is ~3x faster than the eager fp32 path
    # (compile is the real lever; eager bf16 alone gains nothing). The batch dim is
    # dynamic (the inference loaders keep full coverage -> partial trailing batch);
    # inference has no dropout, so the dynamic-batch compile is safe here (unlike the
    # training forward, which is why the train/valid loaders drop_last). The MLP route
    # keeps the eager fp32 forward (cheap; just reused now, not rebuilt per call).
    predict_fwd = {
        e: predict_proba(
            e,
            chunk_size=_PREDICT_REPLICA_CHUNK,
            autocast_dtype=torch.bfloat16 if is_cnn else None,
            compile=is_cnn and compile_forward,
            compile_kwargs=cfg.compile_kwargs if (is_cnn and compile_forward) else None,
            # Static-shape compile via padding: the inference loaders keep full
            # coverage, so each has a different trailing-batch size; padding every
            # batch up to `reweight_batch` keeps the compiled graph at one shape
            # (dynamic=True still recompiled per size and blew the recompile_limit).
            pad_to=reweight_batch if is_cnn else None,
        )
        for e in _ensembles
    }

    # Pre-compile the train/valid forwards on dummy batches so the first real batch
    # pays no compile cost, and persist the artifacts (Mega-Cache) keyed by
    # architecture/shape so re-runs skip recompilation. All four ensembles share the
    # arch+shape, so the first warmup compiles and the rest are Inductor cache hits.
    if compile_forward:
        # Collapse changes the conv arch (kernels/depth), so key the cache on it too —
        # otherwise the avg-pool bundle would be (harmlessly but wastefully) preloaded.
        arch_tag = "_collapse" if (is_cnn and cfg.cnn_collapse) else ""
        compile_cache = (
            root_dir
            / "compile_cache"
            / f"{feature_mode}{arch_tag}_R{cfg['num_replicas']}_B{cfg['batch_size']}.bin"
        )
        if load_compile_cache(compile_cache):
            print(f"Loaded torch.compile cache from {compile_cache}")
        feature_shape = tuple(detlvl_ds.input.shape[1:])
        print("Warming up compiled train/valid forwards on dummy batches...")
        # Page one model onto the GPU at a time (warm, then park back on CPU) so the
        # warmup peak is one model + one dummy batch, not all four models resident.
        _warmed = {}
        for e, s in (
            (detlvl_ensemble, detlvl_state),
            (miss_scaling_ensemble, miss_scaling_state),
            (partlvl_ensemble, partlvl_state),
            (fake_scaling_ensemble, fake_scaling_state),
        ):
            s = s.to(device)
            warmup_compiled(
                e,
                s,
                batch_size=cfg["batch_size"],
                feature_shape=feature_shape,
                grad_loss=grad_loss[e],
                valid_loss_fn=valid_loss[e],
            )
            # CNN route: also warm the compiled reweighting forward (dynamic batch) on
            # a dummy inference batch so the first real reweight pays no compile stall.
            if is_cnn:
                warmup_compiled(
                    e,
                    s,
                    batch_size=reweight_batch,
                    feature_shape=feature_shape,
                    predict_forward=predict_fwd[e],
                )
            _warmed[e] = s.to("cpu")
        detlvl_state = _warmed[detlvl_ensemble]
        miss_scaling_state = _warmed[miss_scaling_ensemble]
        partlvl_state = _warmed[partlvl_ensemble]
        fake_scaling_state = _warmed[fake_scaling_ensemble]
        if save_compile_cache(compile_cache):
            print(f"Saved torch.compile cache to {compile_cache}")

    # Keep only the currently-training ensemble on the GPU. The four models train
    # strictly sequentially each OmniFold iteration, so the other three are parked
    # on the CPU between their phases and paged back in just-in-time (see the per-
    # phase `.to(device)` / `.to("cpu")` swaps in the loop below). Peak GPU memory
    # then holds one model's params + Adam moments, not all four. Park every model
    # here so the loop starts with one-at-a-time even when compile (and its warmup
    # bring-up above) is off.
    detlvl_state = detlvl_state.to("cpu")
    miss_scaling_state = miss_scaling_state.to("cpu")
    partlvl_state = partlvl_state.to("cpu")
    fake_scaling_state = fake_scaling_state.to("cpu")
    _log_mem("setup done (pre-loop)")

    # --- Resume detection -------------------------------------------------------
    # A run can resume at an arbitrary iteration (e.g. after a molab 12 h timeout).
    # Each completed iteration writes a full checkpoint under
    # embedding/<sysvar>/checkpoints/iter{N}/ (cumulative component weights + all
    # four ensemble states incl. Adam moments). `resume` auto-detects the latest
    # complete checkpoint; `resume_from_iter` overrides it explicitly.
    ckpt_dir = out_dir / "checkpoints"
    _resume_from = cfg.get("resume_from_iter", None)
    if _resume_from is not None:
        start_iter = int(_resume_from)
    elif cfg.get("resume", False):
        start_iter = _find_latest_complete_checkpoint(ckpt_dir) or 0
    else:
        start_iter = 0
    _keep_last_ckpt = cfg.get("keep_last_checkpoint_only", True)

    # w_data is the (fixed) data-jet weight vector — deterministic from the dataset,
    # so it is always (re)derived, never checkpointed.
    w_data = (
        (
            detlvl_ds.td["weight"][detlvl_ds.indices][detlvl_ds.target > 0.5]
            .expand(cfg["num_replicas"], -1)
            .clone()
        )
        if cfg["num_replicas"] > 1
        else detlvl_ds.td["weight"][detlvl_ds.indices][detlvl_ds.target > 0.5].clone()
    )

    if start_iter == 0:
        # OmniFold reweights are streamed to disk one file per iteration (see
        # _save_unfolding_weights) instead of accumulating the full (R, n_jets)
        # history in RAM. The loop only ever reads back the *previous* gen reweights
        # (the old w_unfolding[-3]) and the *previous* reco reweights
        # (w_unfolding[-1]), so we keep just those two rolling tensors. The priors
        # are iteration 0.
        # prev_gen is an independent, persistent buffer: the loop copy_'s the new
        # cur_gen into it each iteration (see block B), so it must NOT alias
        # partlvl_ds.weight. prev_reco's buffer is (re)established by the pre-loop
        # block below (clone), so its init can stay a view (only used for the niter0
        # save on the next line). On resume both come back as independent tensors.
        prev_gen = partlvl_ds.sample_weight[..., partlvl_ds.target < 0.5].clone()
        prev_reco = detlvl_ds.sample_weight[..., detlvl_ds.target < 0.5]
        _save_unfolding_weights(out_dir / "w_unfolding_niter0.npz", prev_gen, prev_reco)

        w_matched = (
            (
                partlvl_ds.td["weight"][partlvl_matched_indices]
                .expand(cfg["num_replicas"], -1)
                .clone()
            )
            if cfg["num_replicas"] > 1
            else partlvl_ds.td["weight"][partlvl_matched_indices].clone()
        )
        w_miss = (
            (
                partlvl_ds.td["weight"][partlvl_missed_indices]
                .expand(cfg["num_replicas"], -1)
                .clone()
            )
            if cfg["num_replicas"] > 1
            else partlvl_ds.td["weight"][partlvl_missed_indices].clone()
        )
        w_fake = (
            (
                detlvl_ds.td["weight"][detlvl_fake_indices]
                .expand(cfg["num_replicas"], -1)
                .clone()
            )
            if cfg["num_replicas"] > 1
            else detlvl_ds.td["weight"][detlvl_fake_indices].clone()
        )
        weight_stats_history: list[dict[str, np.ndarray]] = []
    else:
        # Resume: restore the cumulative component weights, rolling priors, stat
        # history, and the four ensemble states (in place) from checkpoint iter{N}.
        print(
            f"[resume] continuing at iteration {start_iter + 1}/{cfg['num_iterations']} "
            f"from {ckpt_dir / f'iter{start_iter}'}"
        )
        (
            w_matched,
            w_miss,
            w_fake,
            prev_gen,
            prev_reco,
            weight_stats_history,
        ) = _load_resume_checkpoint(
            ckpt_dir,
            start_iter,
            states={
                "detlvl": detlvl_state,
                "miss_scaling": miss_scaling_state,
                "partlvl": partlvl_state,
                "fake_scaling": fake_scaling_state,
            },
        )

    print("Saving config file to folder", out_dir, "...")
    shutil.copy("runtime-files/config.json", out_dir / "config.json")

    early_stopping_kwargs = dict(patience=cfg.early_stopping_patience)

    # Optional per-replica LR scheduler (torchstrap-native, freeze-aware), built once
    # here and passed to every fit_ensemble. `policy` selects the LRScheduler policy;
    # the rest are its kwargs. Absent → constant LR.
    _lr_sched = cfg.lr_schedule
    _lr_policy = _lr_sched.pop("policy", None)
    _lr_kwargs = _lr_sched  # remaining keys are the policy's kwargs
    if _lr_policy:
        print(
            f"LR scheduler: {_lr_policy}({', '.join(f'{k}={v}' for k, v in _lr_kwargs.items())})"
        )

    # Tier 2.7: per-step odds clamp for reweigh_samples. The default [1e-3, 1e3] is a 10^6
    # range (non-binding) — tighten so each OmniFold step is a small, trustworthy nudge.
    _step_clip = cfg.get("step_odds_clip", {"clamp_min": 0.25, "clamp_max": 4.0})
    _step_lo, _step_hi = float(_step_clip["clamp_min"]), float(_step_clip["clamp_max"])
    print(f"Per-step odds clamp: [{_step_lo}, {_step_hi}]")

    # Phase-1 diagnostics: per-iteration weight stats and training history.
    # See plan: /home/tanmaypani/.claude/plans/the-prior-reweighing-given-piped-fiddle.md
    reweight_clamp = cfg.reweight_clamp
    # weight_stats_history is initialised above (empty for a fresh run, restored
    # from checkpoint on resume) — do not reset it here.
    # weight_stats_history: list[dict[str, np.ndarray]] = []
    fit_history_dir = out_dir / "fit_history"
    fit_history_dir.mkdir(parents=True, exist_ok=True)

    # Explainability: optionally persist the converged det-/part-level models per
    # iteration so feature_importance.py can attribute the reweighting post-hoc.
    # Off by default to keep standard runs lean. Only params+buffers are saved
    # (see model_io.save_model_weights); the z-norm transform rides along in the
    # buffers. See /home/tanmaypani/.claude/plans/analyze-this-project-the-expressive-firefly.md
    save_model_states = cfg.get("save_model_states", False)
    model_states_dir = out_dir / "model_states"
    if save_model_states:
        model_states_dir.mkdir(parents=True, exist_ok=True)

    if start_iter >= cfg["num_iterations"]:
        print(
            f"[resume] start_iter={start_iter} >= num_iterations="
            f"{cfg['num_iterations']}; run already complete, nothing to do."
        )
        # Still (re)build the combined w_unfolding.npz from the streamed per-iteration
        # files so downstream readers have it even on a no-op resume.
        _reassemble_unfolding_weights(out_dir, cfg["num_iterations"])
        return

    # The carried-forward / saved reco reweight (`prev_reco`) is the RAW cumulative
    # reco block cat([w_matched, w_fake]) normalized to sum n_data — NO class
    # balancing. The detlvl *training* weight is the class-balanced version produced
    # by set_sample_weight (is_categorical=True). These must stay distinct: doing the
    # normalization on the post-balancing buffer double-scales by n_reco/reco_sum and
    # unbalances the detlvl classes. So normalize the raw reco slice in place, clone
    # it out as prev_reco BEFORE set_sample_weight mutates the buffer, then donate the
    # buffer for training (the per-replica reco scale washes out in class balancing).
    # No cat: overwrite the dataset's own persistent (R, N) weight buffer in place (it
    # already holds last iteration's donated weight vector and is in the same
    # [w_data | w_matched | w_fake] column layout), instead of allocating a fresh
    # _full = torch.cat(...) each iteration — that (R, N) alloc/free is the host-RAM
    # fragmentation source. w_data is constant but the in-place class balancing mutates
    # it, so re-copy all three slices every time.
    _n_data = w_data.shape[-1]
    _nm = w_matched.shape[-1]
    W = detlvl_ds.weight
    assert W.shape[-1] == _n_data + _nm + w_fake.shape[-1]
    W[..., :_n_data].copy_(w_data)
    W[..., _n_data : _n_data + _nm].copy_(w_matched)
    W[..., _n_data + _nm :].copy_(w_fake)
    w_reco_sum = w_matched.sum(1, keepdim=True) + w_fake.sum(1, keepdim=True)
    W[..., _n_data:].mul_(n_data).div_(w_reco_sum)
    prev_reco = W[..., _n_data:].clone()  # allocate the persistent snapshot buffer
    detlvl_ds.set_sample_weight(W, copy=False)  # W is self.weight: no-op install + rebalance
    # --- old (cat-allocating; pre-copy_ refactor) -----------------------------------
    # _full = torch.cat([w_data, w_matched, w_fake], dim=1)
    # _full[..., _n_data:].mul_(n_data).div_(w_reco_sum)
    # prev_reco = _full[..., _n_data:].clone()
    # detlvl_ds.set_sample_weight(_full, copy=False)
    # --- old (buggy) ----------------------------------------------------------------
    # Integer index `[..., w_data.shape[-1]]` (missing `:`) selects one column and the
    # in-place div_ raises a broadcast error; and it normalizes the class-balanced
    # block by the raw sum (double-scale). Replaced by the block above.
    # detlvl_ds.set_sample_weight(
    #     torch.cat([w_data, w_matched, w_fake], dim=1), copy=False
    # )
    # w_reco_sum = w_matched.sum(1, keepdim=True) + w_fake.sum(1, keepdim=True)
    # detlvl_ds.weight[..., w_data.shape[-1]].div_(w_reco_sum).mul_(n_data)
    # prev_reco = detlvl_ds.weight[..., w_data.shape[-1]]

    for iteration in range(start_iter, cfg["num_iterations"]):
        print(
            "###############################################################################################"
        )
        print(f"Iteration: {iteration + 1}/{cfg['num_iterations']}")
        print(
            "###############################################################################################"
        )

        _log_mem(f"iter {iteration + 1} start")

        # Tier 0.1: re-arm the LR each iteration (states + their annealed LR warm-start;
        # the scheduler sniffs its base from state.optimizer_state["lr"], so without this
        # the LR ratchets down run-wide and hits min_lr by ~iter4 -> dead later iterations).
        _reset_optimizer_lr(
            detlvl_state, miss_scaling_state, partlvl_state, fake_scaling_state,
            base_lr=_base_lr,
        )

        detlvl_state = detlvl_state.to(device)  # page this model onto the GPU
        if _cold_start and iteration > start_iter:
            detlvl_state = _cold_restart_state(detlvl_state, detlvl_build, base_lr=_base_lr)
        # detlvl_history = detlvl_ensemble.fit(
        #     Adam, binary_cross_entropy_with_logits, detlvl_state,
        #     detlvl_train_loader, valid_iterator=detlvl_valid_loader,
        #     num_epochs=cfg["num_epochs"],
        #     callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
        #     randomness="different",
        # )
        detlvl_history = fit_ensemble(
            detlvl_ensemble,
            detlvl_state,
            detlvl_train_loader,
            detlvl_valid_loader,
            num_epochs=cfg["num_epochs"],
            grad_loss=grad_loss[detlvl_ensemble],
            valid_loss_fn=valid_loss[detlvl_ensemble],
            patience=cfg.early_stopping_patience,
            valid_every=cfg.get("valid_every", 1),
            lr_policy=_lr_policy,
            lr_kwargs=_lr_kwargs,
            verbose=True,
            name="detlvl",
        )
        _log_mem(f"iter {iteration + 1} detlvl trained")

        # reco-match reweight: fold IN PLACE into the contiguous positive-class block
        # of miss_scaling_ds (reset to 1 first so the fold acts as a fresh SET, == the
        # old `miss_scaling_ds.sample_weight[..., pos] = reco_match_reweights`), then
        # carry that block into the cumulative w_matched. One forward, no gathered
        # (R, n) reweight tensor and no torch.cat. Positives are the prefix block: the
        # ds is built with targets = [ones(n_match), zeros(n_match)].
        n_miss_pos = int(miss_scaling_ds.target.gt(0.5).sum())
        miss_scaling_ds.weight[..., :n_miss_pos].fill_(1.0)
        reweigh_samples(
            detlvl_ensemble,
            detlvl_state,
            reco_match_loader,
            miss_scaling_ds.weight,
            forward=predict_fwd[detlvl_ensemble],
            clamp_min=_step_lo,
            clamp_max=_step_hi,
        )
        w_matched.mul_(miss_scaling_ds.weight[..., :n_miss_pos])

        # fake reweight: single consumer — fold straight into the cumulative w_fake.
        reweigh_samples(
            detlvl_ensemble,
            detlvl_state,
            reco_fake_loader,
            w_fake,
            forward=predict_fwd[detlvl_ensemble],
            clamp_min=_step_lo,
            clamp_max=_step_hi,
        )

        if save_model_states:
            save_model_weights(
                detlvl_state, model_states_dir / f"iter{iteration + 1}_detlvl.pt"
            )

        rearm_replicas(detlvl_state)
        detlvl_state = detlvl_state.to("cpu")  # park; done with detlvl this iteration
        _log_mem(f"iter {iteration + 1} detlvl done")

        # --- folded in place above (no gathered reweight tensor / torch.cat) ---
        # w_matched *= reco_match_reweights
        # w_fake *= fake_reweights
        # miss_scaling_ds.sample_weight[..., miss_scaling_ds.target > 0.5] = (
        #     reco_match_reweights
        # )
        miss_scaling_state = miss_scaling_state.to(device)  # page onto the GPU
        if _cold_start and iteration > start_iter:
            miss_scaling_state = _cold_restart_state(
                miss_scaling_state, miss_build, base_lr=_base_lr
            )
        # miss_scaling_history = miss_scaling_ensemble.fit(
        #     Adam, binary_cross_entropy_with_logits, miss_scaling_state,
        #     miss_scaling_train_loader, valid_iterator=miss_scaling_valid_loader,
        #     num_epochs=cfg["num_epochs"],
        #     callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
        #     randomness="different",
        # )
        miss_scaling_history = fit_ensemble(
            miss_scaling_ensemble,
            miss_scaling_state,
            miss_scaling_train_loader,
            miss_scaling_valid_loader,
            num_epochs=cfg["num_epochs"],
            grad_loss=grad_loss[miss_scaling_ensemble],
            valid_loss_fn=valid_loss[miss_scaling_ensemble],
            patience=cfg.early_stopping_patience,
            valid_every=cfg.get("valid_every", 1),
            lr_policy=_lr_policy,
            lr_kwargs=_lr_kwargs,
            verbose=True,
            name="miss_scaling",
        )
        _log_mem(f"iter {iteration + 1} miss_scaling trained")

        # miss reweight: single consumer — fold straight into the cumulative w_miss.
        reweigh_samples(
            miss_scaling_ensemble,
            miss_scaling_state,
            gen_miss_loader,
            w_miss,
            forward=predict_fwd[miss_scaling_ensemble],
            clamp_min=_step_lo,
            clamp_max=_step_hi,
        )

        rearm_replicas(miss_scaling_state)
        miss_scaling_state = miss_scaling_state.to("cpu")  # park; done this iteration
        _log_mem(f"iter {iteration + 1} miss_scaling done")

        # --- folded in place above ---
        # w_miss *= miss_reweights

        # Normalize in place (cur_gen * n_data / cur_gen.sum) — avoids a second
        # (R, n_gen) ~2.3 GB allocation from the out-of-place .mul.
        # cur_gen = torch.cat([w_matched, w_miss], dim=1)
        # cur_gen.div_(cur_gen.sum(1, keepdim=True)).mul_(n_data)

        # Same pattern as the reco side: the new `prev_gen` is the RAW cur_gen block
        # cat([w_matched, w_miss]) normalized to sum n_data (no class balancing),
        # cloned out BEFORE set_sample_weight class-balances the buffer for training.
        # No cat: overwrite partlvl_ds's own (R, 2*n_gen) weight buffer in place
        # ([w_matched | w_miss | prev_gen] layout). Copy the OLD prev_gen into the
        # prior block BEFORE updating prev_gen from the normalized cur_gen block.
        _n_gen = prev_gen.shape[-1]  # == w_matched.shape[-1] + w_miss.shape[-1]
        _nm = w_matched.shape[-1]
        W = partlvl_ds.weight
        assert W.shape[-1] == _n_gen + prev_gen.shape[-1]
        W[..., :_nm].copy_(w_matched)
        W[..., _nm:_n_gen].copy_(w_miss)
        W[..., _n_gen:].copy_(prev_gen)  # old prev_gen -> prior block (read before update)
        w_gen_sum = w_matched.sum(1, keepdim=True) + w_miss.sum(1, keepdim=True)
        W[..., :_n_gen].mul_(n_data).div_(w_gen_sum)
        prev_gen.copy_(W[..., :_n_gen])
        partlvl_ds.set_sample_weight(W, copy=False)  # W is self.weight: rebalance in place
        # --- old (cat-allocating; pre-copy_ refactor) ---
        # _full = torch.cat([w_matched, w_miss, prev_gen], dim=1)
        # _full[..., :_n_gen].mul_(n_data).div_(w_gen_sum)
        # prev_gen.copy_(_full[..., :_n_gen])
        # partlvl_ds.set_sample_weight(_full, copy=False)
        # --- old (buggy: normalizes the class-balanced cur_gen block by the raw sum;
        #     prev_gen aliased a buffer that class balancing had already rescaled) ---
        # partlvl_ds.set_sample_weight(
        #     torch.cat([w_matched, w_miss, prev_gen], dim=1), copy=False
        # )
        # w_gen_sum = w_matched.sum(1, keepdim=True) + w_miss.sum(1, keepdims=True)
        # partlvl_ds.weight[..., : prev_gen.shape[-1]].div_(w_gen_sum).mul_(n_data)
        # prev_gen = partlvl_ds.weight[..., : prev_gen.shape[-1]]
        partlvl_state = partlvl_state.to(device)  # page onto the GPU
        if _cold_start and iteration > start_iter:
            partlvl_state = _cold_restart_state(
                partlvl_state, partlvl_build, base_lr=_base_lr
            )
        # partlvl_history = partlvl_ensemble.fit(
        #     Adam, binary_cross_entropy_with_logits, partlvl_state,
        #     partlvl_train_loader, valid_iterator=partlvl_valid_loader,
        #     num_epochs=cfg["num_epochs"],
        #     callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
        #     randomness="different",
        # )
        partlvl_history = fit_ensemble(
            partlvl_ensemble,
            partlvl_state,
            partlvl_train_loader,
            partlvl_valid_loader,
            num_epochs=cfg["num_epochs"],
            grad_loss=grad_loss[partlvl_ensemble],
            valid_loss_fn=valid_loss[partlvl_ensemble],
            patience=cfg.early_stopping_patience,
            valid_every=cfg.get("valid_every", 1),
            lr_policy=_lr_policy,
            lr_kwargs=_lr_kwargs,
            verbose=True,
            name="partlvl",
        )
        _log_mem(f"iter {iteration + 1} partlvl trained")
        # gen-match reweight (reused): fold IN PLACE into the contiguous positive-class
        # block of fake_scaling_ds (reset to 1 -> fold == fresh SET, == the old
        # `fake_scaling_ds.sample_weight[..., pos] = gen_match_reweights`), then carry
        # that block into the cumulative w_matched. One forward, no gather / torch.cat.
        n_fake_pos = int(fake_scaling_ds.target.gt(0.5).sum())
        fake_scaling_ds.weight[..., :n_fake_pos].fill_(1.0)
        reweigh_samples(
            partlvl_ensemble,
            partlvl_state,
            gen_match_loader,
            fake_scaling_ds.weight,
            forward=predict_fwd[partlvl_ensemble],
            clamp_min=_step_lo,
            clamp_max=_step_hi,
        )
        w_matched.mul_(fake_scaling_ds.weight[..., :n_fake_pos])

        # miss reweight: single consumer — fold straight into the cumulative w_miss.
        reweigh_samples(
            partlvl_ensemble,
            partlvl_state,
            gen_miss_loader,
            w_miss,
            forward=predict_fwd[partlvl_ensemble],
            clamp_min=_step_lo,
            clamp_max=_step_hi,
        )

        if save_model_states:
            save_model_weights(
                partlvl_state, model_states_dir / f"iter{iteration + 1}_partlvl.pt"
            )

        rearm_replicas(partlvl_state)
        partlvl_state = partlvl_state.to("cpu")  # park; done this iteration
        _log_mem(f"iter {iteration + 1} partlvl done")

        # --- folded in place above (no gathered reweight tensor / torch.cat) ---
        # w_matched *= gen_match_reweights
        # w_miss *= miss_reweights
        # fake_scaling_ds.sample_weight[..., fake_scaling_ds.target > 0.5] = (
        #     gen_match_reweights
        # )
        fake_scaling_state = fake_scaling_state.to(device)  # page onto the GPU
        if _cold_start and iteration > start_iter:
            fake_scaling_state = _cold_restart_state(
                fake_scaling_state, fake_build, base_lr=_base_lr
            )
        # fake_scaling_history = fake_scaling_ensemble.fit(
        #     Adam, binary_cross_entropy_with_logits, fake_scaling_state,
        #     fake_scaling_train_loader, valid_iterator=fake_scaling_valid_loader,
        #     num_epochs=cfg["num_epochs"],
        #     callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
        #     randomness="different",
        # )
        fake_scaling_history = fit_ensemble(
            fake_scaling_ensemble,
            fake_scaling_state,
            fake_scaling_train_loader,
            fake_scaling_valid_loader,
            num_epochs=cfg["num_epochs"],
            grad_loss=grad_loss[fake_scaling_ensemble],
            valid_loss_fn=valid_loss[fake_scaling_ensemble],
            patience=cfg.early_stopping_patience,
            valid_every=cfg.get("valid_every", 1),
            lr_policy=_lr_policy,
            lr_kwargs=_lr_kwargs,
            verbose=True,
            name="fake_scaling",
        )
        _log_mem(f"iter {iteration + 1} fake_scaling trained")

        # fake reweight: single consumer — fold straight into the cumulative w_fake.
        reweigh_samples(
            fake_scaling_ensemble,
            fake_scaling_state,
            reco_fake_loader,
            w_fake,
            forward=predict_fwd[fake_scaling_ensemble],
            clamp_min=_step_lo,
            clamp_max=_step_hi,
        )
        # --- folded in place above ---
        # w_fake *= fake_reweights
        # See the pre-loop block: no cat — overwrite detlvl_ds's own (R, N) weight
        # buffer in place ([w_data | w_matched | w_fake]) and snapshot the raw-
        # normalized reco slice into the persistent prev_reco before rebalancing.
        _n_data = w_data.shape[-1]
        _nm = w_matched.shape[-1]
        W = detlvl_ds.weight
        assert W.shape[-1] == _n_data + _nm + w_fake.shape[-1]
        W[..., :_n_data].copy_(w_data)
        W[..., _n_data : _n_data + _nm].copy_(w_matched)
        W[..., _n_data + _nm :].copy_(w_fake)
        w_reco_sum = w_matched.sum(1, keepdim=True) + w_fake.sum(1, keepdim=True)
        W[..., _n_data:].mul_(n_data).div_(w_reco_sum)
        prev_reco.copy_(W[..., _n_data:])
        detlvl_ds.set_sample_weight(W, copy=False)  # W is self.weight: rebalance in place
        # --- old (cat-allocating; pre-copy_ refactor) ---
        # _full = torch.cat([w_data, w_matched, w_fake], dim=1)
        # _full[..., _n_data:].mul_(n_data).div_(w_reco_sum)
        # prev_reco.copy_(_full[..., _n_data:])
        # detlvl_ds.set_sample_weight(_full, copy=False)
        # --- old (buggy: integer index crash + normalizes the class-balanced block) -
        # detlvl_ds.set_sample_weight(
        #     torch.cat([w_data, w_matched, w_fake], dim=1), copy=False
        # )
        # w_reco_sum = w_matched.sum(1, keepdim=True) + w_fake.sum(1, keepdim=True)
        # detlvl_ds.weight[..., w_data.shape[-1]].div_(w_reco_sum).mul_(n_data)
        # prev_reco = detlvl_ds.weight[..., w_data.shape[-1]]

        rearm_replicas(fake_scaling_state)
        fake_scaling_state = fake_scaling_state.to("cpu")  # park; done this iteration
        _log_mem(f"iter {iteration + 1} fake_scaling done")

        _save_unfolding_weights(
            out_dir / f"w_unfolding_niter{iteration + 1}.npz", prev_gen, prev_reco
        )

        # --- Phase-1 diagnostics: weight stats + training history --------
        iter_stats: dict[str, np.ndarray] = {}
        for _name, _w in (
            ("w_data", w_data),
            ("w_matched", w_matched),
            ("w_miss", w_miss),
            ("w_fake", w_fake),
            ("gen", prev_gen),
            ("reco", prev_reco),
        ):
            for _k, _v in _weight_stats(_w, **reweight_clamp).items():
                iter_stats[f"{_name}/{_k}"] = _v
        weight_stats_history.append(iter_stats)

        stacked = {
            k: np.stack([h[k] for h in weight_stats_history])
            for k in weight_stats_history[0]
        }
        np.savez(out_dir / f"weight_stats_niter{cfg['num_iterations']}.npz", **stacked)

        # --- Tier 1: principled stopping (convergence on ln2 + ESS floor) -------------
        _ln2 = float(np.log(2.0))
        _detlvl_bv = _best_valid_median(detlvl_history)
        _eps = float(cfg.get("convergence_eps", 0.005))
        _converged = _detlvl_bv >= (_ln2 - _eps)
        _ess_floor = float(cfg.get("ess_floor_frac", 0.0))
        _gen_ess_frac = float(np.median(iter_stats["gen/ess"]) / prev_gen.shape[-1])
        _reco_ess_frac = float(np.median(iter_stats["reco/ess"]) / prev_reco.shape[-1])
        print(
            f"[stop-check] iter {iteration + 1}: detlvl best-valid(med)={_detlvl_bv:.4f} "
            f"(ln2-eps={_ln2 - _eps:.4f})  gen_ESS={_gen_ess_frac:.3%}  "
            f"reco_ESS={_reco_ess_frac:.3%}"
        )
        _ess_breach = _ess_floor > 0 and min(_gen_ess_frac, _reco_ess_frac) < _ess_floor
        _stop_after_this_iter = cfg.get("early_stop_on_convergence", True) and (
            _converged or _ess_breach
        )
        if _stop_after_this_iter:
            _why = (
                "detlvl valid >= ln2-eps (no separable detector-level signal)"
                if _converged
                else f"ESS < {_ess_floor:.0%} (weights collapsed)"
            )
            print(
                f"[stop-check] STOPPING after iter {iteration + 1}: {_why}. "
                f"Trust weights up to w_unfolding_niter{iteration + 1}.npz."
            )

        for _name, _hist in (
            ("detlvl", detlvl_history),
            ("miss_scaling", miss_scaling_history),
            ("partlvl", partlvl_history),
            ("fake_scaling", fake_scaling_history),
        ):
            _hist.to_file(fit_history_dir / f"iter{iteration + 1}_{_name}.json")

        # --- Resume checkpoint: full iteration-boundary snapshot ----------
        # All four states are on CPU and post-`rearm_replicas` here, exactly what a
        # continuous run carries into the next iteration. Writing it last (after the
        # diagnostics/history) means a complete checkpoint implies all per-iteration
        # outputs are on disk too.
        _save_resume_checkpoint(
            ckpt_dir,
            iteration,
            w_matched=w_matched,
            w_miss=w_miss,
            w_fake=w_fake,
            prev_gen=prev_gen,
            prev_reco=prev_reco,
            stacked_stats=stacked,
            states={
                "detlvl": detlvl_state,
                "miss_scaling": miss_scaling_state,
                "partlvl": partlvl_state,
                "fake_scaling": fake_scaling_state,
            },
            keep_last_only=_keep_last_ckpt,
        )

        _release_heap()
        _log_mem(f"iter {iteration + 1} end (post-trim)")
        _last_completed_iter = iteration  # niter0..niter{iteration+1} are on disk

        if _stop_after_this_iter:
            break

    # Combine the streamed per-iteration files into the legacy single w_unfolding.npz
    # (arr_{2i}=gen / arr_{2i+1}=reco) that histograms.py / plot_closure.py expect.
    # RAM-flat: one iteration file loaded at a time (see _reassemble_unfolding_weights).
    # Cap at the iteration actually completed (an early stop ends below
    # cfg["num_iterations"]); the reassembler prunes any stale higher-index files.
    _reassemble_unfolding_weights(out_dir, _last_completed_iter + 1)

    print("Done !")


if __name__ == "__main__":
    run(load_config())
