import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    import urllib.request
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Match plot_physics / the poster: reset to matplotlib's default style so text and
    # ticks render black-on-white (the live notebook's dark theme leaves text white,
    # invisible on the white savefig facecolor). Then layer the publication rcParams.
    plt.style.use("default")
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "axes.linewidth": 1.2,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )

    # Reuse the poster's data-point drawer (markers + stat bars + filled systematic
    # boxes when given a sys_err_path). Importing plot_physics runs its app.setup
    # block (loads config.json, sets plot_sys_err=True), so the band is enabled.
    from plot_physics import plot_hist_single
    from systematics import SysVar, get_jet_pt_bins

    return (
        Line2D,
        Patch,
        Path,
        SysVar,
        get_jet_pt_bins,
        json,
        np,
        plot_hist_single,
        plt,
        urllib,
    )


@app.cell
def config(Path, SysVar, get_jet_pt_bins):
    # Overlay our *no-p_T^D* unfolded distributions (feature_mode angularities_noptd,
    # unfolded-data result "nominal") on top of the published STAR pp 200 GeV points.
    # First pass: our existing snapshot bins as-is (no rebinning); overlay every
    # published pT bin that overlaps each of our pT slices. (Reprojection onto
    # HEPData's exact bins is the documented later improvement.)
    HIST_ROOT = Path("outputs/histograms")
    HEP_DIR = Path("runtime-files/hepdata")
    OUT_DIR = Path("outputs/hp2026_poster/pub_comparison")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FEATURE_MODE = "angularities_noptd"
    SYSVAR = "nominal"

    # Our jet-pT slice edges -> jpt0 [10,15] jpt1 [15,20] jpt2 [20,30] jpt3 [30,60].
    JPT_EDGES = tuple(float(x) for x in get_jet_pt_bins(SysVar.NONE))

    # Per-observable HEPData provenance + the published (pt_lo, pt_hi, table_id) bins.
    #   mass  -> ins1853218 (recid 102953, v2): x = low/high edges, y = 1/N dN/dM
    #   zg/Rg -> ins1783875 (recid  93789, v1): x = bin center,    y = 1/N dN/dx
    PUB = {
        "m": dict(
            inspire="ins1853218",
            recid=102953,
            version=2,
            bins=[(20, 25, 1386081), (25, 30, 1386082), (30, 40, 1386083)],
        ),
        "sd_m": dict(
            inspire="ins1853218",
            recid=102953,
            version=2,
            bins=[(20, 25, 1386084), (25, 30, 1386085), (30, 40, 1386086)],
        ),
        "sd_symmetry": dict(
            inspire="ins1783875",
            recid=93789,
            version=1,
            bins=[
                (15, 20, 745595),
                (20, 25, 745596),
                (25, 30, 745597),
                (30, 40, 745598),
                (40, 60, 745599),
            ],
        ),
        "sd_dR": dict(
            inspire="ins1783875",
            recid=93789,
            version=1,
            bins=[
                (15, 20, 745600),
                (20, 25, 745601),
                (25, 30, 745602),
                (30, 40, 745603),
                (40, 60, 745604),
            ],
        ),
    }

    var_xlabel = {
        "m": r"$M_{\rm jet}$ (GeV$/c^2$)",
        "sd_m": r"$M_{\rm jet,\,g}$ (GeV$/c^2$)",
        "sd_dR": r"$R_{g}$",
        "sd_symmetry": r"$z_{g}$",
    }
    var_ylabel = {
        "m": r"$\frac{1}{N_{\rm jets}}\frac{dN_{\rm jets}}{dM}$ ($c^2$/GeV)",
        "sd_m": r"$\frac{1}{N_{\rm jets}}\frac{dN_{\rm jets}}{dM_g}$ ($c^2$/GeV)",
        "sd_dR": r"$\frac{1}{N_{\rm jets}}\frac{dN_{\rm jets}}{dR_g}$",
        "sd_symmetry": r"$\frac{1}{N_{\rm jets}}\frac{dN_{\rm jets}}{dz_g}$",
    }
    var_xlim = {
        "m": (0.0, 12.0),
        "sd_m": (0.0, 12.0),
        "sd_dR": (0.0, 0.45),
        "sd_symmetry": (0.08, 0.52),
    }
    return (
        FEATURE_MODE,
        HEP_DIR,
        HIST_ROOT,
        JPT_EDGES,
        OUT_DIR,
        PUB,
        SYSVAR,
        var_xlabel,
        var_xlim,
        var_ylabel,
    )


@app.cell
def hepdata(HEP_DIR, PUB, json, np, urllib):
    # --- HEPData fetch + local cache -----------------------------------------------
    # Only /record/data/{recid}/{tableid}/{version} passes Cloudflare (the /download/*
    # endpoints 403). The table JSON contains a raw tab char -> json.loads(strict=False).
    # We normalize each row to (x_lo, x_hi, x_ctr, value, stat, syst) and cache a small
    # JSON per (obs, pt bin) so re-runs need no network.
    _UA = {"User-Agent": "Mozilla/5.0"}


    def _normalize(raw):
        x_lo, x_hi, x_ctr, val, stat, syst = [], [], [], [], [], []
        for _r in raw["values"]:
            _x = _r["x"][0]
            if "low" in _x and "high" in _x:
                _lo, _hi = float(_x["low"]), float(_x["high"])
                x_lo.append(_lo)
                x_hi.append(_hi)
                x_ctr.append(0.5 * (_lo + _hi))
            else:  # center-only (zg/Rg record)
                x_ctr.append(float(_x["value"]))
                x_lo.append(None)
                x_hi.append(None)
            _y = _r["y"][0]
            val.append(float(_y["value"]))
            _st = _sy = 0.0
            for _e in _y.get("errors", []):
                _lab = _e.get("label", "").lower()
                _v = float(_e.get("symerror", _e.get("asymerror", {}).get("plus", 0)))
                if "stat" in _lab:
                    _st = _v
                elif "syst" in _lab:
                    _sy = _v
            stat.append(_st)
            syst.append(_sy)
        if x_lo[0] is None:  # derive half-widths from uniform center spacing
            _c = np.asarray(x_ctr)
            _dx = float(np.median(np.diff(_c))) if len(_c) > 1 else 0.0
            x_lo = (_c - _dx / 2).tolist()
            x_hi = (_c + _dx / 2).tolist()
        return dict(x_lo=x_lo, x_hi=x_hi, x_ctr=x_ctr, value=val, stat=stat, syst=syst)


    def load_hep(obs, pt_lo, pt_hi):
        """Cached published distribution for (obs, pt bin) as numpy arrays. Fetches +
        caches from HEPData on first access; reads runtime-files/hepdata/... thereafter."""
        _rec = PUB[obs]
        _tid = next(t for lo, hi, t in _rec["bins"] if (lo, hi) == (pt_lo, pt_hi))
        _cache = HEP_DIR / _rec["inspire"] / f"{obs}_pt{pt_lo}_{pt_hi}.json"
        if _cache.exists():
            _n = json.loads(_cache.read_text())
        else:
            _url = f"https://www.hepdata.net/record/data/{_rec['recid']}/{_tid}/{_rec['version']}"
            _req = urllib.request.Request(_url, headers=_UA)
            with urllib.request.urlopen(_req, timeout=60) as _resp:
                _raw = json.loads(_resp.read().decode("utf-8"), strict=False)
            _n = _normalize(_raw)
            _n.update(
                inspire=_rec["inspire"],
                recid=_rec["recid"],
                version=_rec["version"],
                table_id=_tid,
                obs=obs,
                pt_lo=pt_lo,
                pt_hi=pt_hi,
            )
            _cache.parent.mkdir(parents=True, exist_ok=True)
            _cache.write_text(json.dumps(_n, indent=1))
        _x = np.asarray(_n["x_ctr"], float)
        _lo = np.asarray(_n["x_lo"], float)
        _hi = np.asarray(_n["x_hi"], float)
        _stat = np.asarray(_n["stat"], float)
        _syst = np.asarray(_n["syst"], float)
        return dict(
            x=_x,
            half_width=0.5 * (_hi - _lo),
            value=np.asarray(_n["value"], float),
            stat=_stat,
            syst=_syst,
            total=np.hypot(_stat, _syst),
        )

    return (load_hep,)


@app.cell
def drawers(
    FEATURE_MODE,
    HIST_ROOT,
    JPT_EDGES,
    Line2D,
    OUT_DIR,
    PUB,
    Patch,
    SYSVAR,
    load_hep,
    np,
    plot_hist_single,
    plt,
    var_xlabel,
    var_xlim,
    var_ylabel,
):
    # Distinct color per published pT bin (shared across observables/panels).
    PT_COLORS = {
        (15, 20): "#1f77b4",
        (20, 25): "#2ca02c",
        (25, 30): "#ff7f0e",
        (30, 40): "#9467bd",
        (40, 60): "#8c564b",
    }


    def overlapping_pub_bins(obs, jpt):
        """Published (pt_lo, pt_hi) bins whose range overlaps our pT slice `jpt`."""
        _olo, _ohi = JPT_EDGES[jpt], JPT_EDGES[jpt + 1]
        return [(lo, hi) for lo, hi, _ in PUB[obs]["bins"] if lo < _ohi and hi > _olo]


    def covered_jpts(obs):
        """Our pT slices (jpt index) that have >=1 overlapping published bin."""
        return [j for j in range(len(JPT_EDGES) - 1) if overlapping_pub_bins(obs, j)]


    def plot_hep_points(ax, hep, color, label):
        """Published points: filled systematic box + stat error bars + square markers."""
        _edges = np.append(hep["x"] - hep["half_width"], hep["x"][-1] + hep["half_width"][-1])
        ax.stairs(
            hep["value"] + hep["syst"],
            _edges,
            baseline=hep["value"] - hep["syst"],
            fill=True,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.errorbar(
            hep["x"],
            hep["value"],
            yerr=hep["stat"],
            xerr=hep["half_width"],
            linestyle="none",
            marker="s",
            markersize=5,
            color=color,
            markeredgecolor="white",
            zorder=3,
            label=label,
        )


    def draw_obs(var):
        """One figure for `var`: a panel per covered pT slice, overlaying our no-p_T^D
        unfolded distribution and every overlapping published STAR bin."""
        _jpts = covered_jpts(var)
        _fig, _axs = plt.subplots(1, len(_jpts), figsize=(5.4 * len(_jpts), 5.2), squeeze=False)
        _axs = _axs[0]
        for _ax, _j in zip(_axs, _jpts):
            # --- our no-p_T^D unfolded data (markers + stat bars + filled sys boxes) ---
            _ours = HIST_ROOT / SYSVAR / FEATURE_MODE / var / f"hist_jpt{_j}.pt"
            _sys = HIST_ROOT / "sys_errors" / FEATURE_MODE / var / f"hist_jpt{_j}.pt"
            plot_hist_single(
                _ax,
                "errorbar",
                file_path=_ours,
                sys_err_path=_sys if _sys.exists() else None,
                color="black",
                marker="o",
                label="this analysis",
            )
            # --- every overlapping published STAR bin ---
            for _lo, _hi in overlapping_pub_bins(var, _j):
                plot_hep_points(
                    _ax,
                    load_hep(var, _lo, _hi),
                    PT_COLORS[(_lo, _hi)],
                    rf"STAR pub. ${_lo}\!<\!p_T\!<\!{_hi}$",
                )
            _ax.set_xlim(*var_xlim[var])
            _ax.set_ylim(bottom=0)
            _ax.set_xlabel(var_xlabel[var], fontsize="x-large")
            # legend: our handle + one per published bin, titled with our pT slice
            _handles = [
                Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    linestyle="none",
                    markeredgecolor="white",
                    label=r"this $\pm\,\delta_{\rm sys}$",
                ),
                Patch(facecolor="black", alpha=0.4, label=r""),
            ]
            for _lo, _hi in overlapping_pub_bins(var, _j):
                _handles.append(
                    Line2D(
                        [],
                        [],
                        color=PT_COLORS[(_lo, _hi)],
                        marker="s",
                        linestyle="none",
                        markeredgecolor="white",
                        label=rf"STAR pub. ${_lo}\!<\!p_T\!<\!{_hi}$",
                    )
                )
            _ax.legend(
                handles=_handles,
                loc="upper right",
                frameon=False,
                title=rf"our slice: ${int(JPT_EDGES[_j])}\!<\!p_{{\rm T,jet}}\!<\!"
                rf"{int(JPT_EDGES[_j + 1])}$ GeV/$c$",
            )
        _axs[0].set_ylabel(var_ylabel[var], fontsize="x-large")
        _fig.suptitle(
            r"STAR $pp$ $\sqrt{s}=200$ GeV, anti-$k_T$ $R=0.4$  "
            r"$\cdot$  unfolded (no $p_T^D$) vs published",
            fontsize="medium",
        )
        _fig.tight_layout()
        _fig.savefig(OUT_DIR / f"fig_pub_{var}.pdf", bbox_inches="tight")
        return _fig

    return PT_COLORS, draw_obs, plot_hep_points


@app.cell
def fig_mass(draw_obs):
    fig_m = draw_obs("m")
    fig_m
    return


@app.cell
def fig_groomed_mass(draw_obs):
    fig_sd_m = draw_obs("sd_m")
    fig_sd_m
    return


@app.cell
def fig_zg(draw_obs):
    fig_zg = draw_obs("sd_symmetry")
    fig_zg
    return


@app.cell
def fig_rg(draw_obs):
    fig_rg = draw_obs("sd_dR")
    fig_rg
    return


@app.cell
def regroom_config(OUT_DIR, Path):
    ZCUT_PUB = 0.1  # published STAR RHIC SoftDrop zcut (ins1783875, ins1853218)
    BETA_PUB = 0.0
    ZCUT_OURS = 0.2  # our analysis grooming (already baked into the hist_jpt*.pt snapshots)
    NOPTD_ITER = 2  # no-p_T^D central OmniFold iteration -> gen weights arr_{2*iter}=arr_4

    REGROOM_VARS = ("sd_m", "sd_symmetry", "sd_dR")  # groomed mass, zg, Rg
    REGROOM_CACHE = OUT_DIR / "regroom_zcut010_noptd.npz"
    WUNF = Path(
        "datasets/STAR_pp200GeV_production_2012/features/angularities_noptd/"
        "embedding/nominal/w_unfolding.npz"
    )
    # Stage-1 gen jets (with constituents), in the order the unfolded weights align to:
    # [gen-matches(all pT-hat) ++ misses(all pT-hat)], each pT-hat block ascending.
    JETS_EMB = Path("datasets/STAR_pp200GeV_production_2012/jets/embedding/nominal")
    return (
        BETA_PUB,
        JETS_EMB,
        NOPTD_ITER,
        REGROOM_CACHE,
        WUNF,
        ZCUT_OURS,
        ZCUT_PUB,
    )


@app.cell
def regroom_compute(
    BETA_PUB,
    JETS_EMB,
    NOPTD_ITER,
    REGROOM_CACHE,
    WUNF,
    ZCUT_OURS,
    ZCUT_PUB,
    con,
    np,
):
    # Heavy one-time compute (fastjet re-grooming of ALL gen jets), cached.
    import awkward as ak
    import fastjet as fj
    import pyarrow as pa

    from preprocessing import (
        get_softdrop_groomed_jets,
        jet_r,
        pth_bins,
        to_jet_and_consitit_vectors,
    )


    def _regroom_file(_path):
        # Groom the FULL file exactly like preprocessing.process_table (NO pT pre-mask:
        # boolean-masking the jagged constituents before grooming corrupts the per-jet
        # clustering -> wrong zg/Rg). The pT slice is applied later, at histogram time.
        _buf = pa.memory_map(str(_path), "rb")
        _tbl = pa.ipc.open_file(_buf).read_all()
        _jets, _con = to_jet_and_consitit_vectors(ak.from_arrow(_tbl, generate_bitmasks=True))
        _jet_mass_mask = _jets.m > 1.0
        _jets = _jets[_jet_mass_mask]
        _con = con[_jet_mass_mask]
        _pt = ak.to_numpy(_jets.pt).astype(np.float64)
        _sd, _ = get_softdrop_groomed_jets(
            _con,
            fj.JetDefinition(fj.antikt_algorithm, jet_r, fj.E_scheme),
            beta=BETA_PUB,
            symmetry_cut=ZCUT_PUB,
            R0=jet_r,
        )
        # Failed-SD jets carry sentinel -1 (sym/dR) / ~0 (m); they fall below every
        # published bin range and are dropped by np.histogram -- matching the published
        # SD-tagged definition.
        _m = ak.to_numpy(_sd.m).astype(np.float64)
        _dr = ak.to_numpy(_sd.dR).astype(np.float64)
        _sym = ak.to_numpy(_sd.symmetry).astype(np.float64)
        return _pt, _m, _dr, _sym


    if REGROOM_CACHE.exists():
        _z = np.load(REGROOM_CACHE)
        regroom = {k: _z[k] for k in ("pt", "sd_m", "sd_dR", "sd_symmetry")}
    else:
        _pt_a, _m_a, _dr_a, _sym_a = [], [], [], []
        for _fname in ("gen-matches", "misses"):  # matches block first (weight alignment)
            for _lo, _hi in zip(pth_bins[:-1], pth_bins[1:]):
                _f = JETS_EMB / f"ptHat{_lo}to{_hi}" / f"{_fname}.arrow"
                print(f"  re-grooming {_f} ...")
                _p, _m, _d, _s = _regroom_file(_f)
                _pt_a.append(_p)
                _m_a.append(_m)
                _dr_a.append(_d)
                _sym_a.append(_s)
        regroom = {
            "pt": np.concatenate(_pt_a),
            "sd_m": np.concatenate(_m_a),
            "sd_dR": np.concatenate(_dr_a),
            "sd_symmetry": np.concatenate(_sym_a),
        }
        REGROOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(REGROOM_CACHE, **regroom)

    # Per-replica no-p_T^D unfolded gen weights at iter NOPTD_ITER (arr_{2*iter}).
    regroom_w = np.load(WUNF)[f"arr_{2 * NOPTD_ITER}"]
    assert regroom["pt"].shape[0] == regroom_w.shape[1], (
        f"re-groom length {regroom['pt'].shape[0]} != weights {regroom_w.shape[1]} "
        "-- gen-row ordering mismatch"
    )
    print(f"re-groomed {regroom['pt'].shape[0]} gen jets; weights {regroom_w.shape}")

    # --- zcut=0.2 reference: reuse the already-computed sd_* from the FEATURES arrows
    # (same gen order [gen-matches ++ misses]) so the reference can be re-binned into the
    # published pT bins exactly like the re-groomed (zcut=0.1) result. ---
    _emb = WUNF.parent


    def _ftcol(_c):
        _a = []
        for _f in ("gen-matches.arrow", "misses.arrow"):
            _b = pa.memory_map(str(_emb / _f), "rb")
            _a.append(pa.ipc.open_file(_b).read_all()[_c].to_numpy())
        return np.concatenate(_a).astype(np.float64)


    regroom02 = {_c: _ftcol(_c) for _c in ("pt", "m", "sd_m", "sd_dR", "sd_symmetry")}
    assert np.allclose(regroom02["pt"], regroom["pt"]), (
        "zcut=0.2 features pt misaligned with regroom pt"
    )
    print(f"loaded zcut={ZCUT_OURS} reference sd_* for {regroom02['pt'].shape[0]} gen jets")
    return regroom, regroom02, regroom_w


@app.cell
def regroom_draw(
    Line2D,
    OUT_DIR,
    PT_COLORS,
    PUB,
    ZCUT_OURS,
    ZCUT_PUB,
    load_hep,
    np,
    plot_hep_points,
    plt,
    regroom,
    regroom02,
    regroom_w,
    sysband,
    var_xlabel,
    var_ylabel,
):
    def _pub_edges(var, lo, hi):
        """Published observable bin edges for pT bin (lo,hi)."""
        _h = load_hep(var, lo, hi)
        return np.append(_h["x"] - _h["half_width"], _h["x"][-1] + _h["half_width"][-1])


    def _wt_hist(values, pt, lo, hi, edges):
        """(mean, std) per-replica unit-area histogram of `values` for gen-pT in [lo,hi),
        weighted by the per-replica no-p_T^D unfolded weights."""
        _m = (pt >= lo) & (pt < hi)
        _v = values[_m]
        _w = regroom_w[:, _m]
        _h = np.stack(
            [np.histogram(_v, bins=edges, weights=_w[r], density=True)[0] for r in range(_w.shape[0])]
        )
        return _h.mean(0), _h.std(0)


    def make_regroom_fig(var):
        # One panel per PUBLISHED pT bin -> matched grooming AND matched jet-pT bin.
        _bins = [(lo, hi) for lo, hi, _ in PUB[var]["bins"]]
        _n = len(_bins)
        _ncols = 3 if _n > 3 else _n
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axs = plt.subplots(_nrows, _ncols, figsize=(4.8 * _ncols, 5.0 * _nrows), squeeze=False)
        _axs = _axs.ravel()
        for _ax, (_lo, _hi) in zip(_axs, _bins):
            _edges = _pub_edges(var, _lo, _hi)
            _c = 0.5 * (_edges[:-1] + _edges[1:])
            _hw = 0.5 * np.diff(_edges)
            # faint reference: our original zcut=0.2 result, SAME pT bin + edges
            _r0, _ = _wt_hist(regroom02[var], regroom02["pt"], _lo, _hi, _edges)
            _ax.plot(
                _c, _r0, drawstyle="steps-mid", color="gray", alpha=0.6, linestyle="--", linewidth=2
            )
            # re-groomed at the published zcut, SAME pT bin + edges
            _mean, _std = _wt_hist(regroom[var], regroom["pt"], _lo, _hi, _edges)
            _ax.errorbar(
                _c,
                _mean,
                yerr=_std,
                xerr=_hw,
                linestyle="none",
                marker="o",
                markersize=5,
                color="black",
                markeredgecolor="white",
                zorder=4,
            )
            # published STAR points for this exact pT bin
            if (var, _lo, _hi) in sysband:
                _sb = sysband[(var, _lo, _hi)]
                _se = np.append(_sb["cen"] - _sb["hw"], _sb["cen"][-1] + _sb["hw"][-1])
                _ax.stairs(
                    _sb["nom"] + _sb["total_sys"],
                    _se,
                    baseline=_sb["nom"] - _sb["total_sys"],
                    fill=True,
                    color="black",
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
            plot_hep_points(_ax, load_hep(var, _lo, _hi), PT_COLORS[(_lo, _hi)], "STAR pub.")
            _ax.set_xlim(_edges[0], _edges[-1])
            _ax.set_ylim(bottom=0)
            _ax.set_xlabel(var_xlabel[var], fontsize="x-large")
            _handles = [
                Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    linestyle="none",
                    markeredgecolor="white",
                    label=rf"this (no $p_T^D$), $z_{{\rm cut}}={ZCUT_PUB}$",
                ),
                Line2D(
                    [],
                    [],
                    color="gray",
                    linestyle="--",
                    alpha=0.7,
                    label=rf"this, $z_{{\rm cut}}={ZCUT_OURS}$",
                ),
                Line2D(
                    [],
                    [],
                    color=PT_COLORS[(_lo, _hi)],
                    marker="s",
                    linestyle="none",
                    markeredgecolor="white",
                    label="STAR published",
                ),
            ]
            #if (var, _lo, _hi) in sysband:
            #    _handles.insert(
            #        1, Patch(facecolor="black", alpha=0.25, label=r"this $\pm\,\delta_{\rm sys}$")
            #    )
            _ax.legend(
                handles=_handles,
                loc="upper right",
                frameon=False,
                # title=rf"${_lo}\!<\!p_{{\rm T,jet}}\!<\!{_hi}$ GeV/$c$",
            )
            _ax.set_title(rf"${_lo}\!<\!p_{{\rm T,jet}}\!<\!{_hi}$ GeV/$c$")
        for _r in range(_nrows):
            _axs[_r * _ncols].set_ylabel(var_ylabel[var], fontsize="x-large")
        for _ax in _axs[_n:]:
            _ax.set_axis_off()
        #_fig.suptitle(
        #    r"STAR $pp$ $\sqrt{s}=200$ GeV, anti-$k_T$ $R=0.4$  $\cdot$  "
        #    rf"re-groomed at $z_{{\rm cut}}={ZCUT_PUB}$ vs published (matched $p_T$ bins)",
        #    fontsize="medium",
        #)
        _fig.tight_layout()
        _fig.savefig(OUT_DIR / f"fig_pub_regroom_{var}.pdf", bbox_inches="tight")
        return _fig


    def make_ungroomed_fig(var):
        # Ungroomed observable: no grooming -> no zcut variant. Our unfolded result
        # (zcut-independent jet mass from the features arrows) vs published, one panel
        # per published pT bin.
        _bins = [(lo, hi) for lo, hi, _ in PUB[var]["bins"]]
        _n = len(_bins)
        _ncols = 3 if _n > 3 else _n
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axs = plt.subplots(_nrows, _ncols, figsize=(4.8 * _ncols, 5.0 * _nrows), squeeze=False)
        _axs = _axs.ravel()
        for _ax, (_lo, _hi) in zip(_axs, _bins):
            _edges = _pub_edges(var, _lo, _hi)
            _c = 0.5 * (_edges[:-1] + _edges[1:])
            _hw = 0.5 * np.diff(_edges)
            _mean, _std = _wt_hist(regroom02[var], regroom02["pt"], _lo, _hi, _edges)
            _ax.errorbar(
                _c,
                _mean,
                yerr=_std,
                xerr=_hw,
                linestyle="none",
                marker="o",
                markersize=5,
                color="black",
                markeredgecolor="white",
                zorder=4,
            )
            if (var, _lo, _hi) in sysband:
                _sb = sysband[(var, _lo, _hi)]
                _se = np.append(_sb["cen"] - _sb["hw"], _sb["cen"][-1] + _sb["hw"][-1])
                _ax.stairs(
                    _sb["nom"] + _sb["total_sys"],
                    _se,
                    baseline=_sb["nom"] - _sb["total_sys"],
                    fill=True,
                    color="black",
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
            plot_hep_points(_ax, load_hep(var, _lo, _hi), PT_COLORS[(_lo, _hi)], "STAR pub.")
            _ax.set_xlim(_edges[0], _edges[-1])
            _ax.set_ylim(bottom=0)
            _ax.set_xlabel(var_xlabel[var], fontsize="x-large")
            _handles = [
                    Line2D(
                        [],
                        [],
                        color="black",
                        marker="o",
                        linestyle="none",
                        markeredgecolor="white",
                        label=r"this (no $p_T^D$)",
                    ),
                Line2D(
                    [],
                    [],
                    color=PT_COLORS[(_lo, _hi)],
                    marker="s",
                    linestyle="none",
                    markeredgecolor="white",
                    label="STAR published",
                ),
            ]
            # if (var, _lo, _hi) in sysband:
            #    _handles.insert(
            #        1, Patch(facecolor="black", alpha=0.25, label=r"this $\pm\,\delta_{\rm sys}$")
            #    )
            _ax.legend(
                handles=_handles,
                loc="upper right",
                frameon=False,
                fontsize="small",
                title=rf"${_lo}\!<\!p_{{\rm T,jet}}\!<\!{_hi}$ GeV/$c$",
            )
        for _r in range(_nrows):
            _axs[_r * _ncols].set_ylabel(var_ylabel[var], fontsize="x-large")
        for _ax in _axs[_n:]:
            _ax.set_axis_off()
        _fig.suptitle(
            r"STAR $pp$ $\sqrt{s}=200$ GeV, anti-$k_T$ $R=0.4$  $\cdot$  "
            r"ungroomed jet mass vs published (matched $p_T$ bins)",
            fontsize="medium",
        )
        _fig.tight_layout()
        _fig.savefig(OUT_DIR / f"fig_pub_ptbinned_{var}.pdf", bbox_inches="tight")
        return _fig

    return make_regroom_fig, make_ungroomed_fig


@app.cell
def fig_regroom_mass(make_regroom_fig):
    fig_rg_sd_m = make_regroom_fig("sd_m")
    fig_rg_sd_m
    return


@app.cell
def fig_regroom_zg(make_regroom_fig):
    fig_rg_zg = make_regroom_fig("sd_symmetry")
    fig_rg_zg
    return


@app.cell
def fig_regroom_rg(make_regroom_fig):
    fig_rg_rg = make_regroom_fig("sd_dR")
    fig_rg_rg
    return


@app.cell
def _(make_ungroomed_fig):
    fig_ptb_m = make_ungroomed_fig("m")
    fig_ptb_m
    return


@app.cell
def _(OUT_DIR, PUB, WUNF, load_hep, np, regroom, regroom02, regroom_w):
    # === Full reprojected systematic band on the published bins (mirrors systematics.py) ===
    # total_sys = sqrt(unfolding_sys^2 + track_pruned^2 + tower_pruned^2)
    #   unfolding_sys = per-bin max(|nom-iter|, |nom-JER|, |nom-herwig7|, |nom-nonclosure|) (raw)
    #   track/tower   = |nom-var|, Barlow-pruned (threshold 1.0)
    # noptd central iter2 (arr_4); iter sys iter1/iter3 (arr_2/arr_6); nonclosure like_data iter1
    # (arr_2). herwig7/like_data reuse the nominal gen order; track/tower use their own re-groom.
    _EMB = WUNF.parent.parent
    _SYS_CACHE = OUT_DIR / "sysband_zcut010_noptd.npz"
    _SYS_VARS = ("m", "sd_m", "sd_dR", "sd_symmetry")


    def _obs_edges_sys(var, lo, hi):
        _h = load_hep(var, lo, hi)
        return np.append(_h["x"] - _h["half_width"], _h["x"][-1] + _h["half_width"][-1])


    def _bins_iter_sys():
        for _var in _SYS_VARS:
            for _lo, _hi, _ in PUB[_var]["bins"]:
                yield _var, _lo, _hi


    def _rep_sys(values, pt, w2d, lo, hi, edges):
        _m = (pt >= lo) & (pt < hi)
        _v = values[_m]
        _w = w2d[:, _m]
        _hh = np.stack(
            [np.histogram(_v, bins=edges, weights=_w[r], density=True)[0] for r in range(_w.shape[0])]
        )
        return _hh.mean(0), _hh.std(0)


    def _src_hists(obs, w2d, jer=None):
        _o = {}
        for _var, _lo, _hi in _bins_iter_sys():
            _e = _obs_edges_sys(_var, _lo, _hi)
            _dlo = 1.0 if _lo < 30 else 2.0
            _dhi = 1.0 if _hi < 30 else 2.0
            if jer == "wide":
                _lw, _hi2 = _lo - _dlo, _hi + _dhi
            elif jer == "narrow":
                _lw, _hi2 = _lo + _dlo, _hi - _dhi
            else:
                _lw, _hi2 = _lo, _hi
            _o[(_var, _lo, _hi)] = _rep_sys(obs[_var], obs["pt"], w2d, _lw, _hi2, _e)
        return _o


    def _barlow_np(unc, snom, svar, thr=1.0):
        _s2n = snom**2
        _s2v = svar**2
        _s2u = np.abs(_s2v - _s2n)
        _den = np.maximum(np.sqrt(_s2u), np.finfo(float).eps)
        _sig = np.nan_to_num(np.abs(unc) / _den)
        _sig[_s2u <= 0] = 0.0
        return np.where(_sig >= thr, unc, 0.0)


    if _SYS_CACHE.exists():
        sysband = np.load(_SYS_CACHE, allow_pickle=True)["sysband"].item()
    else:
        _rt = np.load(OUT_DIR / "regroom_zcut010_track_pt_sys.npz")
        _tw = np.load(OUT_DIR / "regroom_zcut010_tower_et_corr_sys.npz")
        _otrk = {k: _rt[k] for k in ("pt", "m", "sd_m", "sd_dR", "sd_symmetry")}
        _otwr = {k: _tw[k] for k in ("pt", "m", "sd_m", "sd_dR", "sd_symmetry")}
        # nominal obs: ungroomed m from features (regroom02), groomed sd_* from the zcut=0.1 re-groom
        _onom = {
            "pt": regroom["pt"],
            "m": regroom02["m"],
            "sd_m": regroom["sd_m"],
            "sd_dR": regroom["sd_dR"],
            "sd_symmetry": regroom["sd_symmetry"],
        }

        _H_nom = _src_hists(_onom, regroom_w)
        _H_j0 = _src_hists(_onom, regroom_w, jer="wide")
        _H_j1 = _src_hists(_onom, regroom_w, jer="narrow")
        _H_i1 = _src_hists(_onom, np.load(_EMB / "nominal" / "w_unfolding.npz")["arr_2"])
        _H_i3 = _src_hists(_onom, np.load(_EMB / "nominal" / "w_unfolding.npz")["arr_6"])
        _H_hw = _src_hists(_onom, np.load(_EMB / "unf_prior_herwig7" / "w_unfolding.npz")["arr_4"])
        _H_nc = _src_hists(_onom, np.load(_EMB / "unf_prior_like_data" / "w_unfolding.npz")["arr_2"])
        _H_tk = _src_hists(_otrk, np.load(_EMB / "track_pt_sys" / "w_unfolding.npz")["arr_4"])
        _H_tw = _src_hists(_otwr, np.load(_EMB / "tower_et_corr_sys" / "w_unfolding.npz")["arr_4"])

        sysband = {}
        for _k in _bins_iter_sys():
            _nom, _nstd = _H_nom[_k]
            _it = np.abs(_nom - 0.5 * (_H_i1[_k][0] + _H_i3[_k][0]))
            _je = np.abs(_nom - 0.5 * (_H_j0[_k][0] + _H_j1[_k][0]))
            _hwd = np.abs(_nom - _H_hw[_k][0])
            _ncd = np.abs(_nom - _H_nc[_k][0])
            _unf = np.maximum.reduce([_it, _je, _hwd, _ncd])
            _tkp = _barlow_np(np.abs(_nom - _H_tk[_k][0]), _nstd, _H_tk[_k][1])
            _twp = _barlow_np(np.abs(_nom - _H_tw[_k][0]), _nstd, _H_tw[_k][1])
            _tot = np.sqrt(_unf**2 + _tkp**2 + _twp**2)
            _var, _lo, _hi = _k
            _e = _obs_edges_sys(_var, _lo, _hi)
            sysband[_k] = dict(
                cen=0.5 * (_e[:-1] + _e[1:]), hw=0.5 * np.diff(_e), nom=_nom, total_sys=_tot
            )
        np.savez(_SYS_CACHE, sysband=np.array(sysband, dtype=object))

    print(
        "sysband keys:",
        len(sysband),
        "example total_sys[(sd_symmetry,15,20)]:",
        np.round(sysband[("sd_symmetry", 15, 20)]["total_sys"], 3),
    )
    return (sysband,)


if __name__ == "__main__":
    app.run()
