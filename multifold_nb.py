import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    # Thin driver for running multifold.run(cfg) on marimo molab (cloud Blackwell
    # GPU) or locally. The heavy lifting stays in multifold.py / omnitrain.py; this
    # notebook only handles cloud setup (clone, deps, data download), gates the run
    # behind a Helion/Blackwell smoke test, launches the (long, blocking) unfolding
    # in a single cell, then plots the per-iteration weight diagnostics.
    #
    # Local vs molab is auto-detected: when run from a checkout the clone/pip/download
    # cells no-op (the repo, torchstrap, and the tensordicts are already present), so
    # the same notebook is debuggable on the 4070 and deployable on molab unchanged.
    import os
    import sys
    import json
    import shutil
    import tarfile
    import subprocess
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    import marimo as mo

    # --- cloud setup constants (only used on a fresh molab session) ---------------
    # This study repo (HTTPS — molab has no SSH key). Brings all sibling modules
    # (omnitrain, dataset, config, model_io, systematics) + runtime-files/config.json.
    REPO_URL = "https://github.com/TanmayPani/jet-angularity-study.git"
    REPO_NAME = "jet-angularity-study"
    # torchstrap is not on PyPI; install from git over HTTPS. thoda is NOT needed —
    # the training import graph never touches it.
    TORCHSTRAP_SPEC = "git+https://github.com/TanmayPani/torchstrap.git"

    # Google Drive transfer of the ~5.8 GB uint8 tensordict memmaps, tarred into one
    # archive containing `nominal/det_lvl/` + `nominal/part_lvl/`. Set EITHER a direct
    # file id (preferred — one API call) OR a folder name to glob within.
    DATA_TARBALL_NAME = "tensordicts_nominal_bin_counts.tar"
    GDRIVE_FILE_ID = ""  # e.g. "1AbC...". If empty, search GDRIVE_FOLDER_NAME below.
    GDRIVE_FOLDER_NAME = "jet-angularity-study-data"

    # In-memory overrides applied to cfg before the run (does NOT touch the committed
    # config.json). Use a reduced smoke run first to validate end-to-end on Blackwell
    # within the 12 h cap, then clear these for the full run.
    SMOKE_RUN = True
    SMOKE_OVERRIDES = {"num_iterations": 1, "num_data_subsample": 50000}

    # Resume a timed-out run: each completed iteration writes a full checkpoint
    # under embedding/<sysvar>/checkpoints/. Set True and just re-run the run cell
    # after a molab 12 h timeout — it auto-detects the latest complete checkpoint
    # and continues. (No effect on a fresh output dir with no checkpoints.)
    RESUME_RUN = False


@app.cell
def _():
    mo.md(
        r"""
# MultiFold unfolding on molab

Run order (top → bottom):

1. **Deps / clone** — clone this repo + `pip install` torchstrap (no-op locally).
2. **Config** — load `runtime-files/config.json`, apply smoke overrides.
3. **Data** — pull the tensordict tarball from Google Drive and extract (no-op if present).
4. **Helion smoke test** — JIT the fused Adam kernel for the GPU in isolation. *Must pass.*
5. **Run** — press the button to launch `multifold.run(cfg)` (long, blocking).
6. **Plots** — per-iteration weight stats.

> Before a fresh molab session, upload `token.json` and the OAuth client-secret
> JSON via the sidebar (gitignored), and set `GDRIVE_FILE_ID` in the setup cell.
"""
    )
    return


@app.cell
def _():
    # --- Step 1: deps / clone (auto-detects local checkout vs fresh molab) --------
    _here = Path.cwd()
    _in_repo = (_here / "multifold.py").exists() and (
        _here / "runtime-files" / "config.json"
    ).exists()

    if _in_repo:
        repo_root = _here
    else:
        if not (_here / REPO_NAME).exists():
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
        repo_root = _here / REPO_NAME
        os.chdir(repo_root)

    # torchstrap: editable install locally; git install on molab if missing.
    try:
        import torchstrap  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", TORCHSTRAP_SPEC], check=True
        )

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    deps_ready = True
    print(f"[deps] repo_root={repo_root}  in_repo={_in_repo}")
    return (deps_ready,)


@app.cell
def _(deps_ready):
    # --- Step 2: config ----------------------------------------------------------
    assert deps_ready
    import torch
    from config import load_config
    from multifold import run as run_unfolding
    import gdrive_helper

    cfg = load_config()
    if SMOKE_RUN:
        cfg.update(SMOKE_OVERRIDES)
    if RESUME_RUN:
        cfg["resume"] = True

    print(
        f"[cfg] feature_mode={cfg.get('feature_mode')} "
        f"sys_var={cfg.sys_var} device={cfg.device} "
        f"num_replicas={cfg['num_replicas']} num_iterations={cfg['num_iterations']} "
        f"num_data_subsample={cfg['num_data_subsample']} "
        f"compile_forward={cfg.compile_forward}"
    )
    return cfg, gdrive_helper, run_unfolding, torch


@app.cell
def _(cfg, gdrive_helper):
    # --- Step 3: data download + extract (no-op if already present) ---------------
    data_root = cfg.features_root / "tensordicts" / str(cfg.sys_var)
    det_meta = data_root / "det_lvl" / "meta.json"
    part_meta = data_root / "part_lvl" / "meta.json"

    if det_meta.exists() and part_meta.exists():
        print(f"[data] already present at {data_root} — skipping download.")
    else:
        # Extract into the tensordicts/ dir; the tarball holds `nominal/det_lvl` etc.
        extract_dir = cfg.features_root / "tensordicts"
        extract_dir.mkdir(parents=True, exist_ok=True)
        tarball = extract_dir / DATA_TARBALL_NAME

        service = gdrive_helper.get_service()
        file_id = GDRIVE_FILE_ID
        if not file_id:
            folders = gdrive_helper.find_folder_id(GDRIVE_FOLDER_NAME)
            if not folders:
                raise FileNotFoundError(
                    f"No Drive folder '{GDRIVE_FOLDER_NAME}' and no GDRIVE_FILE_ID set."
                )
            matches = gdrive_helper.GDrivePath(folders[0]["id"]).glob(DATA_TARBALL_NAME)
            if not matches:
                raise FileNotFoundError(
                    f"'{DATA_TARBALL_NAME}' not found in folder '{GDRIVE_FOLDER_NAME}'."
                )
            file_id = matches[0]["id"]

        if not tarball.exists():
            print(f"[data] downloading {DATA_TARBALL_NAME} (~5.8 GB) ...")
            gdrive_helper._download_file(service, file_id, tarball)
        print(f"[data] extracting {tarball} -> {extract_dir} ...")
        with tarfile.open(tarball) as _tf:
            _tf.extractall(extract_dir)

        if not (det_meta.exists() and part_meta.exists()):
            raise RuntimeError(
                f"Extraction did not yield det_lvl/part_lvl meta.json under {data_root}."
            )
        print(f"[data] ready at {data_root}")

    data_ready = True
    return (data_ready,)


@app.cell
def _(deps_ready, torch):
    # --- Step 4: Helion / Blackwell smoke test (the one genuine risk) -------------
    # The fused Adam (torchstrap) JIT-compiles a Triton/Helion kernel with an
    # Ada-pinned config. On Blackwell (sm_120) it SHOULD compile + run; this fires
    # that JIT in isolation on a tiny (R, T) state BEFORE the full run, so a sm_120
    # failure surfaces here instead of 30 min into training. There is NO CPU
    # auto-fallback when CUDA is requested — a failure here means stop, not degrade.
    assert deps_ready
    helion_ok = False

    if not torch.cuda.is_available():
        print("[helion] CUDA not available — training would fall to the CPU path "
              "(hopeless at 29M jets). Check the molab GPU toggle.")
    else:
        cap = torch.cuda.get_device_capability()
        print(f"[helion] device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")

        from torchstrap.optimizer import adam as _adam_mod
        if not getattr(_adam_mod, "_HAS_HELION", False):
            print("[helion] WARNING: helion did not import — no CUDA Adam kernel "
                  "registered; the op would run the slow vectorized path on CUDA.")

        R, T = 4, 1024
        dev = "cuda"
        _p = torch.randn(R, T, device=dev)
        _g = torch.randn(R, T, device=dev)
        _m = torch.zeros(R, T, device=dev)
        _v = torch.zeros(R, T, device=dev)
        _steps = torch.zeros(R, device=dev)
        _full = lambda x: torch.full((R,), x, device=dev)  # noqa: E731
        try:
            _adam_mod.adam_step_(
                _p, _g, _m, _v, None, _steps,
                _full(1e-3), _full(0.9), _full(0.999), _full(1e-8), _full(0.01),
                torch.ones(R, device=dev),  # active_mask
                False, False, True,         # amsgrad, maximize, decoupled_wd
            )
            torch.cuda.synchronize()
            helion_ok = bool(torch.isfinite(_p).all())
            print(f"[helion] fused Adam step OK (finite={helion_ok}).")
        except Exception as _e:  # surface, do not swallow
            print(f"[helion] FAILED to JIT/run the fused Adam kernel: {_e!r}")
            raise

    return (helion_ok,)


@app.cell
def _():
    # --- Step 5a: launch gate ----------------------------------------------------
    run_button = mo.ui.run_button(label="🚀 Launch unfolding run")
    run_button
    return (run_button,)


@app.cell
def _(cfg, data_ready, helion_ok, run_button, run_unfolding):
    # --- Step 5b: the (long, blocking) run ---------------------------------------
    mo.stop(not run_button.value, mo.md("*Press the button above to launch the run.*"))
    mo.stop(not data_ready, mo.md("**Data not ready** — run Step 3 first."))
    mo.stop(not helion_ok, mo.md("**Helion smoke test did not pass** — run Step 4 first."))

    # num_workers stays 0: the torchdata threaded loader leaks worker threads.
    cfg["num_workers"] = cfg.get("num_workers", 0)
    run_unfolding(cfg)
    run_done = True
    return (run_done,)


@app.cell
def _(cfg, run_done):
    # --- Step 6: per-iteration weight diagnostics --------------------------------
    assert run_done
    out_dir = cfg.features_root / "embedding" / str(cfg.sys_var)

    _niter_files = sorted(out_dir.glob("w_unfolding_niter*.npz"))
    _iters, _gen_means, _reco_means = [], [], []
    for _f in _niter_files:
        _n = int(_f.stem.split("niter")[-1])
        _z = np.load(_f)
        # arr_0 = gen weights, arr_1 = reco weights (per multifold._save_unfolding_weights)
        _iters.append(_n)
        _gen_means.append(float(np.asarray(_z["arr_0"]).mean()))
        _reco_means.append(float(np.asarray(_z["arr_1"]).mean()))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(_iters, _gen_means, "o-", label="gen weight mean")
    ax.plot(_iters, _reco_means, "s-", label="reco weight mean")
    ax.set_xlabel("OmniFold iteration")
    ax.set_ylabel("mean per-jet weight")
    ax.set_title(f"{cfg.get('feature_mode')} / {cfg.sys_var}")
    ax.legend()
    fig
    return


if __name__ == "__main__":
    app.run()
