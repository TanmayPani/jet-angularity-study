"""Unified classifier likelihood-ratio reweighter.

This is the production replacement for the sequential-GP *marginal* reweighters
(`omnisequential.py` / `reverse_omnisequential.py`). Those matched **1-D marginals**
one observable at a time and were structurally blind to differences living in the
*joint* (correlation) structure of the phase space. This module instead trains a binary
classifier `f(x)` over the **full joint feature vector** and uses the per-jet odds
`r(x) = f / (1 - f)` as the reweight, which captures all correlations in a single ND
function. It reuses the exact model-construction dispatch the unfolding uses
(`multifold.build_classifier`: an MLP for the scalar/angularities feature vector, a
`Conv2dNN` for the `bin_counts` (2,9,9) image) and the same training driver
(`omnitrain.fit_ensemble`), so the reweighting and the unfolding share machinery.

Two modes:

  --mode gen_prior   Reweight Pythia6 *gen* toward an alternate generator
                     (Herwig7 / Pythia8). pos = alt-gen, neg = p6 gen. The gen odds
                     `r` is applied to gen-matches / misses, the SAME `r` is reused on
                     reco-matches (a matched pair shares its gen jet), fakes pass
                     through unchanged.  -> embedding/unf_prior_<gen>/

  --mode data_reco   Reweight Pythia6 *reco* toward real data (the LIKE_DATA closure
                     prior). A 1-iteration OmniFold built from two classifier steps:
                       1. reco step:  pos = real data, neg = p6 reco (reco-matches ⊕
                          fakes)  ->  r_reco on reco-matches + fakes.
                       2. gen pull-back (OmniFold step 2): matched gen jets inherit
                          their reco partner's r_reco; a SECOND classifier
                          (pos = p6 gen weighted by the pulled-back matched ratio,
                          neg = nominal p6 gen) gives a smooth gen density ratio r_gen
                          applied to gen-matches AND misses (the misses, ~80% of gen
                          jets, have no reco partner so they only get the learned
                          pull-back).  -> embedding/unf_prior_like_data/

The per-jet odds are clamped to [ODDS_CLIP_LO, ODDS_CLIP_HI] = [0.1, 10] and collapsed
across the replica ensemble by geometric mean (the natural average for a log-normal
density ratio); a post-hoc calibration renormalizes the neg-weighted mean ratio to 1 so
total cross-section is preserved. All paths are cfg-driven
(`features/<feature_mode>/embedding/<sysvar>/`); the four output arrows
{gen-matches, reco-matches, misses, fakes}.arrow are written so
`preprocessing.make_datasets_for_unfolding` and `multifold` pick them up unchanged.

CLI:
    python reweight_embedding.py --mode data_reco
    python reweight_embedding.py --mode gen_prior --generator herwig7
"""

import argparse
import os
import shutil

import numpy as np
import pyarrow as pa
import torch

from torch.nn.functional import binary_cross_entropy_with_logits

from config import Config, load_config
from dataset import TensorDictDataset
from preprocessing import N_BINS, jet_columns, replace_table_column, to_tensordict
from multifold import (
    build_classifier,
    reweigh_samples,
    reweight_inference_loaders,
    train_test_multi_loaders,
)
from omnitrain import eval_loss, fit_ensemble, grad_and_loss, predict_proba
from systematics import SysVar

# Reuse the path-independent, already-battle-tested helpers from the (now legacy-CLI)
# make_alt_embedding module: the odds clamp window, arrow IO, per-pth-bin weight
# flattening, the KS feature-alignment audit, and the alt-generator table builders.
from make_alt_embedding import (
    ODDS_CLIP_HI,
    ODDS_CLIP_LO,
    _audit_feature_alignment,
    _flatten_weights_per_bin,
    _read_arrow,
    _write_arrow,
    prepare_herwig7,
    prepare_pythia8,
)


# Scalar feature modes assemble the `input` tensor from named columns; bin_counts is the
# (2, 9, 9) charged+neutral image fed to the CNN. Mirrors preprocessing.FEATURE_MODES.
_SCALAR_MODES = ("angularities", "kinematics", "combined")
# Default input transform per feature mode. The scalar angularity columns carry -1
# "softdrop failed" sentinels, so log1p would NaN — use plain z_norm. The bin_counts
# image is non-negative counts and uses the per-channel log1p path (the unfolding's).
_DEFAULT_TRANSFORM = {
    "angularities": "z_norm",
    "kinematics": "z_norm",
    "combined": "z_norm",
    "bin_counts": "log1p_per_channel_z_norm",
}


def _embedding_dir(cfg, feature_mode, sysvar):
    """`<dataset_root>/features/<feature_mode>/embedding/<sysvar>` (current layout)."""
    return cfg.dataset_root / "features" / feature_mode / "embedding" / str(sysvar)


def _columns_for(feature_mode):
    """Feature columns passed to `to_tensordict`. None for the bin_counts image route
    (it densifies the bin_* list columns instead of selecting scalars)."""
    return None if feature_mode == "bin_counts" else list(jet_columns)


def _ess_over_n(w):
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return float((s * s) / (w * w).sum()) if s > 0 else float("nan")


def estimate_density_ratio(
    pos_table,
    neg_table,
    *,
    feature_mode,
    cfg,
    num_replicas,
    num_epochs,
    device,
    cache_dir,
    label="",
    predict_table=None,
    weight_flatten=True,
    audit=False,
):
    """Train a binary classifier separating `pos_table` (target 1) from `neg_table`
    (target 0) over the joint feature vector and return the per-row density-ratio
    `r = p_pos(x) / p_neg(x)` evaluated on `predict_table` (defaults to `neg_table`),
    so that `neg` reweighted by `r` matches `pos`.

    Returns `(r_mean: np.ndarray (len(predict_table),) float32, history)`.

    Encapsulates: cross-section equalization -> optional KS audit -> per-pth-bin weight
    flattening (training only) -> to_tensordict -> classwise undersample/split loaders
    -> build_classifier (MLP or Conv2dNN) -> fit_ensemble (EarlyStopping, valid-monitored
    LR schedule) -> neg-only inference -> clamped odds -> geometric-mean replica collapse
    -> post-hoc calibration -> ESS / ln2 diagnostics.
    """
    if predict_table is None:
        predict_table = neg_table
    columns = _columns_for(feature_mode)
    tag = f"[reweight:{label}]" if label else "[reweight]"

    # --- 1. Cross-section equalize sum(w_pos) -> sum(w_neg) so the classifier learns the
    # density ratio rather than a mixture-prior-corrupted variant. ---
    w_pos = pos_table["weight"].to_numpy().astype(np.float32)
    w_neg = neg_table["weight"].to_numpy().astype(np.float32)
    s_pos, s_neg = float(w_pos.sum()), float(w_neg.sum())
    if s_pos <= 0 or s_neg <= 0:
        raise RuntimeError(f"{tag} sum(w_pos)={s_pos}, sum(w_neg)={s_neg}; cannot equalize")
    pos_eq = replace_table_column(pos_table, "weight", w_pos * (s_neg / s_pos))
    print(f"{tag} equalized: sum(w_pos)={s_pos:.4g} -> {s_neg:.4g} (x{s_neg / s_pos:.4g})")

    if audit and feature_mode in _SCALAR_MODES:
        _audit_feature_alignment(pos_eq, neg_table, list(jet_columns))

    # --- 2. Training tables: flatten the per-jet MC weight per pth_bin so the multi-decade
    # pT-hat cross-section spread doesn't dominate the BCE loss. Inference / calibration
    # keep the ORIGINAL weights (predict_table below). ---
    pos_train, neg_train = pos_eq, neg_table
    if weight_flatten:
        pos_train = replace_table_column(pos_eq, "weight", _flatten_weights_per_bin(pos_eq))
        neg_train = replace_table_column(neg_table, "weight", _flatten_weights_per_bin(neg_table))
        print(f"{tag} training weights flattened per pth_bin (mean=1 within each bin)")

    # --- 3. Build the (pos, neg) training TensorDict. data_like=pos (target 1). ---
    train_td_dir = os.path.join(cache_dir, f"{label or 'ratio'}_train_td")
    if os.path.exists(train_td_dir):
        shutil.rmtree(train_td_dir)
    train_td = to_tensordict(
        pos_train,
        neg_train,
        columns=columns,
        prefix=train_td_dir,
        max_chunksize=100000,
        feature_mode=feature_mode,
    )
    ds = TensorDictDataset(train_td, is_categorical=True, num_replicas=num_replicas)
    print(f"{tag} classifier TD rows: {len(ds)} (feature_mode={feature_mode})")

    # --- 4. cfg view that drives build_classifier's MLP/CNN dispatch + transform. ---
    transform = cfg.get("input_transform", _DEFAULT_TRANSFORM.get(feature_mode, "z_norm"))
    if feature_mode in _SCALAR_MODES and "log1p" in transform:
        # log1p(-1) = -inf on the softdrop sentinels -> NaN loss. Force z_norm.
        print(f"{tag} overriding input_transform {transform!r} -> 'z_norm' (scalar sentinels)")
        transform = "z_norm"
    cfg_local = Config({**cfg, "feature_mode": feature_mode,
                        "num_replicas": num_replicas, "input_transform": transform})

    rng = torch.Generator().manual_seed(int(cfg["dataseed"]))
    torch.manual_seed(int(cfg["modelseed"]))

    is_cnn = feature_mode == "bin_counts"
    train_loader, valid_loader = train_test_multi_loaders(
        ds,
        train_size=float(cfg["train_size"]),
        undersample_size=cfg.get("num_data_subsample", 1.0),
        batch_size=int(cfg["batch_size"]),
        num_replicas=num_replicas,
        generator=rng,
        num_workers=0,
        stratifys=(False, True),
    )

    num_features = ds.input.shape[-1]
    ensemble, state = build_classifier(
        ds.input, cfg_local.layer_sizes(num_features), cfg_local, device, cfg_local.optimizer_kwargs
    )

    # --- 5. Train. Mirror multifold's loss/lr wiring (valid-monitored ReduceLROnPlateau). ---
    autocast = torch.bfloat16 if is_cnn else None
    grad_loss = grad_and_loss(ensemble, autocast_dtype=autocast)
    valid_loss = eval_loss(ensemble, binary_cross_entropy_with_logits, autocast_dtype=autocast)
    lr_sched = cfg_local.lr_schedule
    lr_policy = lr_sched.pop("policy", None)

    state = state.to(device)
    history = fit_ensemble(
        ensemble,
        state,
        train_loader,
        valid_loader,
        num_epochs=num_epochs,
        grad_loss=grad_loss,
        valid_loss_fn=valid_loss,
        patience=cfg_local.early_stopping_patience,
        valid_every=cfg.get("valid_every", 1),
        lr_policy=lr_policy,
        lr_kwargs=lr_sched,
        verbose=True,
        name=label or "ratio",
    )

    # Headline closure metric: a final valid_loss at/above ln(2)=0.6931 means the two
    # samples are inseparable (no joint signal the marginal GP would have missed); below
    # ln(2) quantifies real joint structure the classifier is now correcting.
    vl = list(history.get("valid_loss", []))
    if vl:
        vmean = [float(np.asarray(v).mean()) for v in vl]
        print(f"{tag} valid_loss: first={vmean[0]:.4f} min={min(vmean):.4f} "
              f"last={vmean[-1]:.4f}  (ln2 floor = 0.6931)")
        if min(vmean) >= 0.69:
            print(f"{tag} NOTE: valid_loss never beat ln(2) — little/no separable joint "
                  f"signal; the reweighting will be near-unity.")

    # --- 6. Inference on predict_table (ORIGINAL weights) via the neg-only loader. ---
    infer_td_dir = os.path.join(cache_dir, f"{label or 'ratio'}_infer_td")
    if os.path.exists(infer_td_dir):
        shutil.rmtree(infer_td_dir)
    infer_td = to_tensordict(
        predict_table.slice(0, 1),  # 1-row dummy pos; required by the data_like/sim_like API
        predict_table,
        columns=columns,
        prefix=infer_td_dir,
        max_chunksize=100000,
        feature_mode=feature_mode,
    )
    infer_ds = TensorDictDataset(infer_td, is_categorical=True, num_replicas=num_replicas)
    infer_loader = reweight_inference_loaders(
        infer_ds,
        batch_size=int(cfg["batch_size"]) * (1 if is_cnn else 5),
        num_replicas=num_replicas,
        has_unmatched=False,
        num_workers=0,
    )
    n_pred = len(predict_table)
    forward = predict_proba(
        ensemble,
        chunk_size=cfg_local.predict_replica_chunk if is_cnn else None,
        autocast_dtype=autocast,
    )
    # reweigh_samples folds the clamped per-replica odds INTO `r` in place (loader order
    # == predict_table row order), giving r[(num_replicas, n_pred)].
    r = torch.ones((num_replicas, n_pred), dtype=torch.float32)
    state = state.to(device)
    reweigh_samples(
        ensemble, state, infer_loader, r,
        forward=forward, clamp_min=ODDS_CLIP_LO, clamp_max=ODDS_CLIP_HI,
    )

    # Geometric mean across replicas (log-normal density ratio).
    r_mean = torch.exp(torch.log(r.clamp_min(1e-12)).mean(dim=0)).numpy().astype(np.float32)

    # --- 7. Post-hoc calibration: renormalize so the predict-weighted mean(r) = 1 (preserve
    # total cross-section; the per-jet shape carries the physics). ---
    w_pred = predict_table["weight"].to_numpy().astype(np.float64)
    calib = float((r_mean.astype(np.float64) * w_pred).sum() / w_pred.sum())
    if np.isfinite(calib) and calib > 0:
        if calib < 0.5 or calib > 2.0:
            print(f"{tag} WARNING: calibration factor {calib:.4g} far from 1 — classifier "
                  f"odds are scale-biased (near-disjoint feature support); shape preserved.")
        r_mean = (r_mean.astype(np.float64) / calib).astype(np.float32)
        print(f"{tag} post-hoc calibration: w-weighted mean(r) {calib:.4g} -> 1")

    # --- 8. Diagnostics: clamp fractions, quantiles, ESS. ---
    r_np = r.numpy().ravel()
    n_lo = int((r_np <= ODDS_CLIP_LO * 1.001).sum())
    n_hi = int((r_np >= ODDS_CLIP_HI * 0.999).sum())
    qs = np.quantile(r_mean, [0.01, 0.1, 0.5, 0.9, 0.99])
    ess_pre = _ess_over_n(w_pred) / n_pred
    ess_post = _ess_over_n(w_pred * r_mean.astype(np.float64)) / n_pred
    print(f"{tag} r_mean quantiles [1,10,50,90,99]% = {[f'{q:.4g}' for q in qs]}\n"
          f"{tag} clamped lo={n_lo / r_np.size:.2%} hi={n_hi / r_np.size:.2%}  "
          f"ESS/n: {ess_pre:.3f} -> {ess_post:.3f} (target >= 0.50)")

    shutil.rmtree(train_td_dir, ignore_errors=True)
    shutil.rmtree(infer_td_dir, ignore_errors=True)
    return r_mean, history


# ----------------------------------------------------------------------------------------
# Mode drivers
# ----------------------------------------------------------------------------------------
_GEN_PRIOR = {
    "herwig7": (SysVar.UNFOLDING_PRIOR_HERWIG7, "outputs/alt_embedding/herwig7", prepare_herwig7),
    "pythia8": (SysVar.UNFOLDING_PRIOR_PYTHIA8, "outputs/alt_embedding/pythia8", prepare_pythia8),
}


def _load_nominal_arrows(cfg, feature_mode):
    """Read the four nominal Pythia6 embedding arrows for the given feature_mode."""
    emb = _embedding_dir(cfg, feature_mode, SysVar.NONE)
    return {
        name: _read_arrow(str(emb / f"{name}.arrow"))
        for name in ("gen-matches", "reco-matches", "misses", "fakes")
    }


def _write_four(out_dir, gm, rm, mi, fk):
    os.makedirs(out_dir, exist_ok=True)
    for name, t in (("gen-matches", gm), ("reco-matches", rm), ("misses", mi), ("fakes", fk)):
        path = os.path.join(out_dir, f"{name}.arrow")
        _write_arrow(t, path)
        print(f"  wrote {path}  (n={len(t)}, sum_w={t['weight'].to_numpy().sum():.4g})")


def run_gen_prior(cfg, *, generator, feature_mode, num_replicas, num_epochs, device):
    """Reweight Pythia6 gen toward an alternate generator. r on gen-matches/misses, the
    same r reused on reco-matches, fakes unchanged."""
    out_sysvar, cache_dir, prepare = _GEN_PRIOR[generator]
    os.makedirs(cache_dir, exist_ok=True)
    arrows = _load_nominal_arrows(cfg, feature_mode)
    gm, rm, mi, fk = (arrows["gen-matches"], arrows["reco-matches"], arrows["misses"], arrows["fakes"])
    n_gm = len(gm)
    assert len(rm) == n_gm, f"reco-matches ({len(rm)}) != gen-matches ({n_gm})"

    alt_table = prepare(cache_dir)
    p6_gen = pa.concat_tables((gm, mi))  # [gen-matches ; misses] order
    print(f"[gen_prior:{generator}] alt={len(alt_table)} p6_gen={len(p6_gen)} "
          f"(gen-matches={n_gm}, misses={len(mi)})")

    r_gen, _ = estimate_density_ratio(
        alt_table, p6_gen,
        feature_mode=feature_mode, cfg=cfg, num_replicas=num_replicas,
        num_epochs=num_epochs, device=device, cache_dir=cache_dir,
        label=f"gen_{generator}", audit=True,
    )
    r_gm, r_mi = r_gen[:n_gm], r_gen[n_gm:]

    gm_out = replace_table_column(gm, "weight", gm["weight"].to_numpy().astype(np.float32) * r_gm)
    rm_out = replace_table_column(rm, "weight", rm["weight"].to_numpy().astype(np.float32) * r_gm)
    mi_out = replace_table_column(mi, "weight", mi["weight"].to_numpy().astype(np.float32) * r_mi)
    _write_four(str(_embedding_dir(cfg, feature_mode, out_sysvar)), gm_out, rm_out, mi_out, fk)
    print(f"[gen_prior:{generator}] done -> {_embedding_dir(cfg, feature_mode, out_sysvar)}")


def run_data_reco(cfg, *, feature_mode, num_replicas, num_epochs, device):
    """Reweight Pythia6 reco toward real data (the LIKE_DATA closure prior): a
    1-iteration OmniFold (reco density-ratio step + gen pull-back step)."""
    out_sysvar = SysVar.UNFOLDING_PRIOR_LIKE_DATA
    cache_dir = os.path.join("outputs", "reweight_embedding", "data_reco", feature_mode)
    os.makedirs(cache_dir, exist_ok=True)

    arrows = _load_nominal_arrows(cfg, feature_mode)
    gm, rm, mi, fk = (arrows["gen-matches"], arrows["reco-matches"], arrows["misses"], arrows["fakes"])
    n_gm, n_rm = len(gm), len(rm)
    assert n_rm == n_gm, f"reco-matches ({n_rm}) != gen-matches ({n_gm}); matched-pair order broken"

    data_table = _read_arrow(str(cfg.dataset_root / "features" / feature_mode / "data.arrow"))
    p6_reco = pa.concat_tables((rm, fk))  # [reco-matches ; fakes]
    print(f"[data_reco] data={len(data_table)} p6_reco={len(p6_reco)} "
          f"(reco-matches={n_rm}, fakes={len(fk)})")

    # --- Step 1: reco density ratio (data vs p6 reco). ---
    r_reco, _ = estimate_density_ratio(
        data_table, p6_reco,
        feature_mode=feature_mode, cfg=cfg, num_replicas=num_replicas,
        num_epochs=num_epochs, device=device, cache_dir=cache_dir, label="reco",
    )
    r_reco_match, r_reco_fake = r_reco[:n_rm], r_reco[n_rm:]

    # --- Step 2: gen pull-back (OmniFold step 2). Push the matched-reco ratio onto the
    # gen-match partners, then learn a smooth gen density ratio (pos = gen weighted by the
    # pushed ratio, neg = nominal gen) and predict it on ALL gen jets (incl. misses). ---
    w_gm = gm["weight"].to_numpy().astype(np.float32)
    gm_pushed = replace_table_column(gm, "weight", w_gm * r_reco_match)
    p6_gen_full = pa.concat_tables((gm, mi))  # predict on [gen-matches ; misses]
    r_gen, _ = estimate_density_ratio(
        gm_pushed, gm,
        feature_mode=feature_mode, cfg=cfg, num_replicas=num_replicas,
        num_epochs=num_epochs, device=device, cache_dir=cache_dir, label="gen_pullback",
        predict_table=p6_gen_full, weight_flatten=False,
    )
    r_gm, r_mi = r_gen[:n_gm], r_gen[n_gm:]

    # --- Splice: reco side gets r_reco, gen side gets the pulled-back r_gen, fakes scaled. ---
    rm_out = replace_table_column(rm, "weight", rm["weight"].to_numpy().astype(np.float32) * r_reco_match)
    fk_out = replace_table_column(fk, "weight", fk["weight"].to_numpy().astype(np.float32) * r_reco_fake)
    gm_out = replace_table_column(gm, "weight", w_gm * r_gm)
    mi_out = replace_table_column(mi, "weight", mi["weight"].to_numpy().astype(np.float32) * r_mi)
    _write_four(str(_embedding_dir(cfg, feature_mode, out_sysvar)), gm_out, rm_out, mi_out, fk_out)
    print(f"[data_reco] done -> {_embedding_dir(cfg, feature_mode, out_sysvar)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("gen_prior", "data_reco"))
    parser.add_argument("--generator", choices=sorted(_GEN_PRIOR), default="herwig7",
                        help="gen_prior only: which alternate generator's prior to bake.")
    parser.add_argument("--config", default="runtime-files/config.json")
    parser.add_argument("--feature-mode", default=None,
                        help="Classifier feature set / embedding subdir. Defaults to cfg.feature_mode.")
    parser.add_argument("--num-replicas", type=int, default=1,
                        help="Ensemble size. Default 1 (decoupled from cfg.num_replicas); the "
                        "vmap-undersample memory bottleneck vanishes at 1 so full samples train.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Override cfg.num_epochs for the classifier(s).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_mode = args.feature_mode or cfg.get("feature_mode", "angularities")
    num_replicas = int(args.num_replicas)
    num_epochs = int(args.num_epochs if args.num_epochs is not None else cfg["num_epochs"])
    device = cfg.device
    print(f"[reweight] mode={args.mode} feature_mode={feature_mode} "
          f"num_replicas={num_replicas} num_epochs={num_epochs} device={device}")

    if args.mode == "gen_prior":
        run_gen_prior(cfg, generator=args.generator, feature_mode=feature_mode,
                      num_replicas=num_replicas, num_epochs=num_epochs, device=device)
    else:
        run_data_reco(cfg, feature_mode=feature_mode, num_replicas=num_replicas,
                      num_epochs=num_epochs, device=device)


if __name__ == "__main__":
    main()
