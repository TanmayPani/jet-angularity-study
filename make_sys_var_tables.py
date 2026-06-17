"""Generate ``sys_vars.tex`` — a standalone LaTeX (article + longtable) dump of
*all* per-source systematic uncertainties on the 1D distributions, with the
Barlow-pruned cells in red and the Barlow significance ``s`` in parentheses next
to each value.

Run with::

    uv run make_sys_var_tables.py [output.tex]

Default output is ``slides/update-pwg-systematics/sys_vars.tex`` (alongside the
main deck ``updates.tex``). It reads the per-source uncertainty files written by
``systematics.py`` (``outputs/histograms/sys_errors/<mode>/<var>/<hist>.pt``,
holding ``<src>_raw`` and ``<src>_barlow_sig`` for each source) together with the
nominal histograms (``outputs/histograms/nominal/<mode>/<var>/<hist>.pt``, for the
``bin_count`` used to turn the absolute uncertainty into a relative percentage).

Each table cell is ``raw rel. sys. [%] (s)`` and is shaded red when the source is
pruned by the Barlow check (significance ``s`` < threshold). This is the same
information as the representative per-source tables hand-written in
``updates.tex``, but generated for every observable / jet-pT bin.
"""

import math
import sys
from pathlib import Path

import torch

from config import load_config
from systematics import (
    PRIOR_SOURCE_KEYS,
    SysVar,
    angularities,
    common_vars,
    get_jet_pt_bins,
    var_label,
)

# --- where the histograms live (same literal as systematics.main) ---
HIST_ROOT = Path("outputs/histograms")
DEFAULT_OUT = Path("slides/update-pwg-systematics/sys_vars.tex")

# Column order + headers, matching the hand-written tables in updates.tex.
# The unfolding-group components (jet-pT res., N_iter, model priors HERWIG7 /
# PYTHIA8, and the method non-closure) are collapsed by per-bin max into the
# `unf. (env.)` column, which is the single unfolding source feeding the total.
SOURCE_COLUMNS = (
    ("track_pt_sys",        r"track-$p_{\rm T}$"),
    ("tower_et_corr_sys",   r"tower-$E_{\rm T}$"),
    ("jet_pt_res_sys",      r"jet-$p_{\rm T}$ res."),
    ("unf_iter_sys",        r"$N_{\rm iter}$"),
    # --- old: single equal-status combined prior column ---
    # ("unf_prior",           r"prior"),
    # --- old: closure-ratio priors (LIKE-DATA + HERWIG7) ---
    # ("unf_prior_like_data", r"LIKE-DATA"),
    # ("unf_prior_herwig7",   r"HERWIG7"),
    # --- new: model-dependence priors (|nominal - alt|) + non-closure +
    # the unfolding max-envelope column ---
    ("unf_prior_herwig7",   r"HERWIG7"),
    ("unf_prior_pythia8",   r"PYTHIA8"),
    ("unf_nonclosure",      r"non-clos."),
    ("unfolding_sys",       r"unf.\ (env.)"),
)

# The envelope source is a raw per-bin max, never Barlow-pruned, so it has no
# significance and no red shading in the table.
ENVELOPE_KEY = "unfolding_sys"

# Math labels for the non-angularity (common) observables.
COMMON_LABEL = {
    "m":           r"M_{\rm jet}",
    "sd_m":        r"M_{\rm jet}^{\rm SD}",
    "sd_dR":       r"\Delta R_{\rm g}",
    "sd_symmetry": r"z_{\rm g}",
}

# Clean per-angularity unit names (text, with inline math only where needed).
# Avoids systematics.var_unit, whose k2_b0 value embeds bare math outside $...$.
ANG_UNIT = {
    "ch_ang_k1_b0.5": "LHA",
    "ch_ang_k1_b1":   "girth",
    "ch_ang_k1_b2":   "thrust",
    "ch_ang_k2_b0":   r"$(p_{\rm T}^{D})^2$",
}


def _safe_rel(unc, count):
    """|unc / count| * 100, with the count==0 bins scrubbed to NaN."""
    safe = torch.where(count != 0, count, torch.full_like(count, float("nan")))
    return (unc.clone() / safe).abs_().mul_(100)


def build_table_entries():
    """Yield ``(var, section_title, hist_basename)`` for every 1D distribution
    to tabulate, in a sensible reading order."""
    entries = []
    # angularities: inclusive (hist_ang_*) then groomed (hist_sd_ang_*)
    for var in angularities:
        unit = ANG_UNIT[var]
        entries.append((
            var,
            f"${var_label[var]}$ ({unit})",
            "hist_ang",
        ))
        sd_key = f"sd_{var}"
        entries.append((
            var,
            f"${var_label[sd_key]}$ ({unit}, groomed)",
            "hist_sd_ang",
        ))
    # common observables: m, sd_m, sd_dR, sd_symmetry (file basename "hist")
    for var in common_vars:
        entries.append((
            var,
            f"${COMMON_LABEL[var]}$",
            "hist",
        ))
    return entries


def _fmt_num(v):
    """Format with at least one decimal place, auto-increasing precision until a
    nonzero significant figure appears, so small values aren't shown as ``0.0``.
    """
    v = float(v)
    if not math.isfinite(v):
        v = 0.0
    if v == 0.0:
        return "0.0"
    d = 1
    while round(abs(v), d) == 0.0 and d < 6:
        d += 1
    return f"{v:.{d}f}"


def format_cell(raw_rel, sig, threshold):
    """One ``raw rel. sys. [%] (s)`` cell; red when pruned (s < threshold)."""
    body = rf"${_fmt_num(raw_rel)}\,({_fmt_num(sig)})$"
    if float(sig) < threshold:
        return rf"\cellcolor{{red!30}}{body}"
    return body


def emit_longtable(out, var, section_title, hist_basename, feature_mode,
                   jpt_bins, threshold, prior_threshold):
    sys_dir = HIST_ROOT / "sys_errors" / feature_mode / var
    nom_dir = HIST_ROOT / "nominal" / feature_mode / var

    # Header row reused for \endfirsthead and \endhead.
    col_headers = " & ".join(h for _, h in SOURCE_COLUMNS)
    header = (
        "\t\t\\hline\n"
        f"\t\tbin & {col_headers} \\\\\n"
        "\t\t\\hline\n"
    )

    wrote_any = False
    for ijpt in range(len(jpt_bins) - 1):
        sys_path = sys_dir / f"{hist_basename}_jpt{ijpt}.pt"
        nom_path = nom_dir / f"{hist_basename}_jpt{ijpt}.pt"
        if not sys_path.exists() or not nom_path.exists():
            continue

        sdict = torch.load(sys_path, mmap=True)
        ndict = torch.load(nom_path, mmap=True)
        count = ndict["bin_count"]
        centers = ndict["bin_center"]

        jpt_lo, jpt_hi = jpt_bins[ijpt], jpt_bins[ijpt + 1]
        caption = (
            rf"{section_title}, "
            rf"${jpt_lo} < p_{{\rm T, jet}} < {jpt_hi}$~GeV/$c$. "
            rf"Cell $=$ raw rel.\ sys.\ [\%] $(s)$; "
            rf"\textcolor{{red!60!black}}{{red}} $=$ pruned "
            rf"(detector $s<{threshold:g}$, prior $s<{prior_threshold:g}$)."
        )

        # 1 bin column + len(SOURCE_COLUMNS) source columns.
        n_cols = len(SOURCE_COLUMNS) + 1
        out.write("\\begin{longtable}{c|" + "c" * len(SOURCE_COLUMNS) + "}\n")
        out.write(f"\t\\caption{{{caption}}}\\\\\n")
        out.write(header)
        out.write("\t\\endfirsthead\n")
        out.write(
            f"\t\\multicolumn{{{n_cols}}}{{l}}{{\\footnotesize\\itshape "
            f"{section_title} -- $p_{{\\rm T,jet}}\\in({jpt_lo},{jpt_hi})$, continued}}\\\\\n"
        )
        out.write(header)
        out.write("\t\\endhead\n")
        out.write("\t\\hline\n\t\\endfoot\n")

        n_bins = count.shape[0]
        for ib in range(n_bins):
            if float(count[ib]) == 0.0:
                continue
            cells = []
            for src_key, _ in SOURCE_COLUMNS:
                raw = sdict.get(f"{src_key}_raw")
                # The unfolding envelope is a raw per-bin max — never Barlow-pruned,
                # so it has no significance: print the plain rel. value, no (s)/red.
                if src_key == ENVELOPE_KEY:
                    if raw is None:
                        cells.append("--")
                        continue
                    raw_rel = _safe_rel(raw, count)
                    cells.append(rf"${_fmt_num(raw_rel[ib])}$")
                    continue
                sig = sdict.get(f"{src_key}_barlow_sig")
                if raw is None or sig is None:
                    cells.append("--")
                    continue
                # shared-event (prior) sources use their own threshold
                thr = prior_threshold if src_key in PRIOR_SOURCE_KEYS else threshold
                raw_rel = _safe_rel(raw, count)
                cells.append(format_cell(raw_rel[ib], sig[ib], thr))
            row = f"${float(centers[ib]):.3g}$ & " + " & ".join(cells)
            out.write(f"\t\t{row} \\\\\n")

        out.write("\\end{longtable}\n\n")
        wrote_any = True

    return wrote_any


def generate(out_path=None, feature_mode=None, threshold=None, prior_threshold=None):
    """Write the full ``sys_vars.tex`` from the saved per-source uncertainty
    files. Importable so other entrypoints (e.g. ``systematics.main``) can
    regenerate the tables right after recomputing the uncertainties.

    ``threshold`` is the detector-variation Barlow cut; ``prior_threshold`` is
    the separate cut for the shared-event (prior) sources."""
    out_path = Path(out_path) if out_path is not None else DEFAULT_OUT

    cfg = load_config()
    if feature_mode is None:
        feature_mode = cfg["feature_mode"]
    if threshold is None:
        threshold = float(cfg.get("barlow_threshold", 1.0) or 1.0)
    if prior_threshold is None:
        prior_threshold = float(
            cfg.get("barlow_threshold_prior", threshold) or threshold
        )
    jpt_bins = get_jet_pt_bins(SysVar.NONE)

    entries = build_table_entries()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("% !!! AUTO-GENERATED by make_sys_var_tables.py -- do not edit by hand.\n")
        out.write("% Regenerate with:  uv run make_sys_var_tables.py\n")
        out.write("\\documentclass[9pt]{article}\n")
        out.write("\\usepackage[a4paper,landscape,margin=1.2cm]{geometry}\n")
        out.write("\\usepackage[table]{xcolor}\n")
        out.write("\\usepackage{longtable}\n")
        out.write("\\usepackage{amsmath}\n")
        out.write("\\renewcommand{\\arraystretch}{1.2}\n")
        out.write("\\setlength{\\tabcolsep}{4pt}\n")
        out.write(
            f"\\title{{Per-source systematic uncertainties "
            f"(\\texttt{{{feature_mode}}}) with Barlow significance}}\n"
        )
        out.write("\\date{}\n")
        out.write("\\begin{document}\n")
        out.write("\\maketitle\n")
        out.write("\\footnotesize\n")
        out.write(
            "\\noindent Each cell is the raw per-source relative systematic "
            "uncertainty in percent, with the Barlow significance "
            "$s=|\\Delta|/\\sqrt{|\\sigma^2_{\\rm var}-\\sigma^2_{\\rm nom}|}$ in "
            "parentheses. \\textcolor{red!60!black}{Red} cells are pruned by the "
            "Barlow check: detector variations and the model-dependence priors "
            f"(track-$p_{{\\rm T}}$, tower-$E_{{\\rm T}}$, jet-$p_{{\\rm T}}$ res., "
            f"$N_{{\\rm iter}}$, HERWIG7, PYTHIA8) use $s<{threshold:g}$, while the "
            f"shared-event non-closure source uses $s<{prior_threshold:g}$. "
            "The unfolding-group sources (jet-$p_{\\rm T}$ res., $N_{\\rm iter}$, "
            "HERWIG7, PYTHIA8, non-closure) are collapsed into a single unfolding "
            "systematic taken as their per-bin \\emph{maximum} (the \\texttt{unf.\\ (env.)} "
            "column, raw, no Barlow gate); the total combines this envelope with "
            "track-$p_{\\rm T}$ and tower-$E_{\\rm T}$ in quadrature. "
            "Empty (zero-content) bins are omitted.\\par\\medskip\n\n"
        )

        for var, section_title, hist_basename in entries:
            out.write(f"\\section*{{{section_title}}}\n")
            wrote = emit_longtable(
                out, var, section_title, hist_basename, feature_mode,
                jpt_bins, threshold, prior_threshold,
            )
            if not wrote:
                out.write("\\emph{(no histograms found)}\\par\n\n")

        out.write("\\end{document}\n")

    print(f"Wrote {out_path}")
    return out_path


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    generate(out_path=out_path)


if __name__ == "__main__":
    main()
