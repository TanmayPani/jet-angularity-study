import re
import textwrap

with open("/home/tanmaypani/star-workspace/jet-angularity-study/plot_hp2026_prelims.py") as f:
    old_code = f.read()

# Parse the cells
cells = []
for cell_match in re.finditer(r'(@app\.cell\ndef _(?:.*?):\n.*?)(?=\n@app\.cell|\Z)', old_code, re.DOTALL):
    cells.append(cell_match.group(1).strip())

config_cell = cells[0]
helper_cell = cells[1]

# Remove draw_dist_column and draw_profile_column from the helper cell
# They start at `def draw_dist_column` and go to the end of the cell.
# The end of the cell is `return draw_dist_column, draw_profile_column` or `return `
helper_cell = re.sub(r'    def draw_dist_column.*', '    return\n', helper_cell, flags=re.DOTALL)

# Since we removed them, we might need to remove them from the signature of the helper cell?
# Wait, they weren't in the signature, they were DEFINED in the helper cell. So removing them is fine.
# But wait, the previous `generate_unrolled.py` generated `return ` at the end of the helper cell.
# Oh wait, `generate_unrolled.py` just dumped `def draw_dist_column` and didn't `return` them explicitly if we don't need them globally.
# Actually, wait, `plot_hp2026_prelims.py` helper cell signature is:
# `def _(CENTER_JPT, FormatStrFormatter, ...):`
# We can just leave the helper cell signature as is, but we must make sure the 13 cells don't take `draw_dist_column` as argument.

out = []
out.append("import marimo\n\n__generated_with = \"0.23.5\"\napp = marimo.App(width=\"full\")\n\n")

out.append(config_cell + "\n")
out.append(helper_cell + "\n")

dist_template = """
    _var_name = "{var}"
    fig_dist_{var_s} = plt.figure(figsize=(6.5, 7.5))
    _axs = fig_dist_{var_s}.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT
    
    _ax_arts = {{}}
    _ax_arts.update(
        plot_hist_single(
            _ax_main, "errorbar",
            file_path=prefix_dir / str(SysVar.NONE) / feature_mode / _var_name / f"hist_sd_ang_jpt{{_jpt_true}}.pt",
            sys_err_path=prefix_dir / "sys_errors" / feature_mode / _var_name / f"hist_sd_ang_jpt{{_jpt_true}}.pt",
            color="red", label="groomed",
        )
    )
    _ax_arts.update(
        plot_hist_single(
            _ax_main, "errorbar",
            file_path=prefix_dir / str(SysVar.NONE) / feature_mode / _var_name / f"hist_ang_jpt{{_jpt_true}}.pt",
            sys_err_path=prefix_dir / "sys_errors" / feature_mode / _var_name / f"hist_ang_jpt{{_jpt_true}}.pt",
            color="blue", marker="^", label="incl.",
        )
    )

    for _mc in mc_labels:
        plot_hist_single(
            _ax_main, "plot",
            file_path=prefix_dir / _mc / feature_mode / _var_name / f"hist_sd_ang_jpt{{_jpt_true}}.pt",
            color="red", label=f"{{_mc}} (groomed)",
            **(mc_hist_styles[_mc]),
        )
        plot_hist_single(
            _ax_main, "plot",
            file_path=prefix_dir / _mc / feature_mode / _var_name / f"hist_ang_jpt{{_jpt_true}}.pt",
            color="blue", label=f"{{_mc}} (incl.)",
            **(mc_hist_styles[_mc]),
        )
        plot_hist_single(
            _ax_incl, "errorbar",
            file_path=prefix_dir / _mc / feature_mode / _var_name / f"ratio_ang_data_vs_{{_mc}}_jpt{{_jpt_true}}.pt",
            sys_err_path=None,
            color="blue", label=_mc,
            **(mc_hist_styles[_mc]),
        )
        plot_hist_single(
            _ax_sd, "errorbar",
            file_path=prefix_dir / _mc / feature_mode / _var_name / f"ratio_sd_ang_data_vs_{{_mc}}_jpt{{_jpt_true}}.pt",
            sys_err_path=None,
            color="red", label=_mc,
            **(mc_hist_styles[_mc]),
        )

    for _ax_r, _stem, _clr in ((_ax_incl, "hist_ang", "blue"), (_ax_sd, "hist_sd_ang", "red")):
        plot_data_sys_band(
            _ax_r,
            prefix_dir / str(SysVar.NONE) / feature_mode / _var_name / f"{{_stem}}_jpt{{_jpt_true}}.pt",
            prefix_dir / "sys_errors" / feature_mode / _var_name / f"{{_stem}}_jpt{{_jpt_true}}.pt",
            color=_clr,
        )
        _ax_r.axhline(y=1, linewidth=2, color="black", linestyle="--", alpha=0.3)
        _ax_r.set_ylim(0.5, 1.5)

    _ax_sd.set_xlabel(var_xlabel[_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if _var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_var_name])
    if _var_name in var_logy:
        _ax_main.set_yscale("log")

    add_top_headroom(_ax_main)
    annotate_corner(_ax_main, show_prelim=True, pt_true=_jpt_true, show_pt=True)

    _ax_arts.update(mc_proxy_handles)

    _ax_main.legend(
        list(_ax_arts.values()), list(_ax_arts.keys()),
        frameon=False, fontsize="x-small",
        loc="upper right", bbox_to_anchor=(0.99, 0.99),
    )

    prune_ratio_panel_yticks(np.array([[_axs[0]], [_axs[1]], [_axs[2]]]))
    
    _axs[0].set_ylabel(var_hist_ylabel[_var_name], fontsize="xx-large")
    _axs[1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _axs[2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_dist_{var_s}
"""

prof_template_body = """
        _ax_main, _ax_incl, _ax_sd = _col_axs[_k][0], _col_axs[_k][1], _col_axs[_k][2]
        _ijpt = _jpt_true - 1

        _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
        _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

        _ax_art_map = {{}}

        _ax_art_map.update(
            plot_profile_single(
                _ax_main, "errorbar",
                file_path=prefix_dir / str(SysVar.NONE) / feature_mode / _var_name / f"prof_sd_vs_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                sys_err_path=prefix_dir / "sys_errors" / feature_mode / _var_name / f"prof_sd_vs_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                color="red", marker="o", label="{{SD}}",
            )
        )
        _ax_art_map.update(
            plot_profile_single(
                _ax_main, "errorbar",
                file_path=prefix_dir / str(SysVar.NONE) / feature_mode / _var_name / f"prof_incl_vs_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                sys_err_path=prefix_dir / "sys_errors" / feature_mode / _var_name / f"prof_incl_vs_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                color="blue", marker="^", label="{{incl.}}",
            )
        )

        for _mc in mc_labels:
            plot_profile_single(
                _ax_main, "plot",
                file_path=prefix_dir / _mc / feature_mode / _var_name / f"prof_sd_vs_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                label="".join(("{{", _mc, "}}")), color="red", **(mc_hist_styles[_mc]),
            )
            plot_profile_single(
                _ax_main, "plot",
                file_path=prefix_dir / _mc / feature_mode / _var_name / f"prof_incl_vs_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                color="blue", **(mc_hist_styles[_mc]),
            )
            plot_hist_single(
                _ax_sd, "errorbar",
                file_path=prefix_dir / _mc / feature_mode / _var_name / f"ratio_prof_sd_vs_{{_x_var_name}}_data_vs_{{_mc}}_jpt{{_jpt_true}}.pt",
                sys_err_path=None, color="red", label=_mc, **(mc_hist_styles[_mc]),
            )
            plot_hist_single(
                _ax_incl, "errorbar",
                file_path=prefix_dir / _mc / feature_mode / _var_name / f"ratio_prof_incl_vs_{{_x_var_name}}_data_vs_{{_mc}}_jpt{{_jpt_true}}.pt",
                sys_err_path=None, color="blue", label=_mc, **(mc_hist_styles[_mc]),
            )

        for _ax_r, _pfx, _clr in ((_ax_incl, "prof_incl_vs", "blue"), (_ax_sd, "prof_sd_vs", "red")):
            plot_data_sys_band(
                _ax_r,
                prefix_dir / str(SysVar.NONE) / feature_mode / _var_name / f"{{_pfx}}_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                prefix_dir / "sys_errors" / feature_mode / _var_name / f"{{_pfx}}_{{_x_var_name}}_jpt{{_jpt_true}}.pt",
                color=_clr,
            )
            _ax_r.axhline(y=1, linewidth=2, color="black", linestyle="--", alpha=0.3)

        _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
        _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

        if _x_var_name in {{_var_name, "sd_dR"}}:
            if _x_var_name in var_xlim:
                _lims = list(var_xlim[_x_var_name])
            else:
                _lims = [np.min([_ax_main.get_xlim(), _ax_main.get_ylim()]), np.max([_ax_main.get_xlim(), _ax_main.get_ylim()])]
            if _x_var_name == "sd_dR":
                _beta = float(_var_name.split("_b")[-1])
                _xs = np.linspace(_lims[0], _lims[1], 200)
                _ax_main.plot(_xs, _xs ** _beta, "--", color="black", alpha=0.3)
            else:
                _ax_main.plot(_lims, _lims, "--", color="black", alpha=0.3)
            _ax_main.set_xlim(_lims)
            _ax_main.set_ylim(_lims)
        elif _x_var_name in var_xlim:
            _ax_main.set_xlim(*var_xlim[_x_var_name])

        if _prof_ylim_val is not None:
            _ax_main.set_ylim(*_prof_ylim_val)
        _ax_incl.set_ylim(*_ratio_ylim_val)
        _ax_sd.set_ylim(*_ratio_ylim_val)

        _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
        _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

        if _prof_ylim_val is None:
            add_top_headroom(_ax_main)

        annotate_corner(_ax_main, show_prelim=({show_prelim_cond}), pt_true=_jpt_true, show_pt=({show_pt_cond}))

        _ax_art_map.update(mc_proxy_handles)

        _leg_loc, _leg_bbox = {{
            "sd_dR": ("lower right", None),
            "sd_m": ("lower right", None),
            "sd_symmetry": ("upper right", (0.99, 0.99)),
        }}.get(_x_var_name, ("upper right", (0.99, 0.99)))

        if {show_legend_cond}:
            _ax_main.legend(
                list(_ax_art_map.values()), list(_ax_art_map.keys()),
                frameon=False, fontsize="xx-small", loc=_leg_loc, bbox_to_anchor=_leg_bbox,
            )

        prune_ratio_panel_yticks(np.array([[_col_axs[_k][0]], [_col_axs[_k][1]], [_col_axs[_k][2]]]))
"""

for var in ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0"]:
    var_s = var.replace('.', '_')
    code = dist_template.format(var=var, var_s=var_s)
    args = "CENTER_JPT, FormatStrFormatter, SysVar, add_top_headroom, annotate_corner, feature_mode, mc_hist_styles, mc_labels, mc_proxy_handles, np, plot_data_sys_band, plot_hist_single, plt, prefix_dir, prune_ratio_panel_yticks, var_hist_ylabel, var_logy, var_xlabel, var_xlim"
    out.append(f"@app.cell\ndef _({args}):\n{code}\n    return fig_dist_{var_s},\n")

for var in ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2"]:
    var_s = var.replace('.', '_')
    cell_code = f"""
    _var_name = "{var}"
    _x_var_name = "sd_dR"
    _jpt_trues = [CENTER_JPT]
    
    fig_prof_{var_s} = plt.figure(figsize=(6, 7.5))
    _axs_raw = fig_prof_{var_s}.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _col_axs = [_axs_raw]
    
    for _k, _jpt_true in enumerate(_jpt_trues):
{prof_template_body.format(show_prelim_cond="_k == 0", show_pt_cond="_jpt_true != _jpt_trues[0] or _k == 0", show_legend_cond="_k == 0")}
        _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    
    _col_axs[0][1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_prof_{var_s}
    """
    args = "CENTER_JPT, FormatStrFormatter, POSTER_PROF_RATIO_YLIM, POSTER_PROF_YLIM, SysVar, add_top_headroom, annotate_corner, feature_mode, mc_hist_styles, mc_labels, mc_proxy_handles, np, plot_data_sys_band, plot_hist_single, plot_profile_single, plt, prefix_dir, prune_ratio_panel_yticks, resolve_ylim, var_xlabel, var_xlim, var_prof_ratio_ylim, var_prof_ylim, var_prof_ylabel"
    out.append(f"@app.cell\ndef _({args}):\n{cell_code}\n    return fig_prof_{var_s},\n")

for jpts in [[1], [2], [3], [1, 2], [2, 3]]:
    var = "ch_ang_k2_b0"
    suffix = "".join(str(j) for j in jpts)
    cell_code = f"""
    _var_name = "{var}"
    _x_var_name = "sd_symmetry"
    _jpt_trues = {jpts}
    _n = len(_jpt_trues)
    
    fig_zg_{suffix} = plt.figure(figsize=(6 * _n, 7.5))
    if _n == 1:
        _axs_raw = fig_zg_{suffix}.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
        _col_axs = [_axs_raw]
    else:
        _axs_raw = fig_zg_{suffix}.subplots(3, _n, height_ratios=[3, 1, 1], sharey="row", sharex="col", squeeze=False, gridspec_kw=dict(hspace=0, wspace=0))
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]
        
    for _k, _jpt_true in enumerate(_jpt_trues):
{prof_template_body.format(show_prelim_cond="_k == 0", show_pt_cond="_jpt_true != _jpt_trues[0] or _k == 0", show_legend_cond="_k == 0")}
        _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
        
    _col_axs[0][1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_zg_{suffix}
    """
    args = "CENTER_JPT, FormatStrFormatter, POSTER_PROF_RATIO_YLIM, POSTER_PROF_YLIM, SysVar, add_top_headroom, annotate_corner, feature_mode, mc_hist_styles, mc_labels, mc_proxy_handles, np, plot_data_sys_band, plot_hist_single, plot_profile_single, plt, prefix_dir, prune_ratio_panel_yticks, resolve_ylim, var_xlabel, var_xlim, var_prof_ratio_ylim, var_prof_ylim, var_prof_ylabel"
    out.append(f"@app.cell\ndef _({args}):\n{cell_code}\n    return fig_zg_{suffix},\n")

# D. Grid
cell_code = f"""
    _kappa1 = ("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2")
    _n = len(_kappa1)
    fig_grid = plt.figure(figsize=(6 * _n, 7.5))
    _axs = fig_grid.subplots(3, _n, height_ratios=[3, 1, 1], sharex="col", sharey=False, squeeze=False, gridspec_kw=dict(hspace=0, wspace=0.30))
    _col_axs = [_axs[:, _k] for _k in range(_n)]

    for _k, _var_name in enumerate(_kappa1):
        _x_var_name = "sd_dR"
        _jpt_true = CENTER_JPT
{prof_template_body.format(show_prelim_cond="_k == 0", show_pt_cond="_k == 0", show_legend_cond="_k == 0")}
        _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")

    _col_axs[0][1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_grid
"""
args = "CENTER_JPT, FormatStrFormatter, POSTER_PROF_RATIO_YLIM, POSTER_PROF_YLIM, SysVar, add_top_headroom, annotate_corner, feature_mode, mc_hist_styles, mc_labels, mc_proxy_handles, np, plot_data_sys_band, plot_hist_single, plot_profile_single, plt, prefix_dir, prune_ratio_panel_yticks, resolve_ylim, var_xlabel, var_xlim, var_prof_ratio_ylim, var_prof_ylim, var_prof_ylabel"
out.append(f"@app.cell\ndef _({args}):\n{cell_code}\n    return fig_grid,\n")

with open("/home/tanmaypani/star-workspace/jet-angularity-study/plot_hp2026_prelims_unrolled_layout.py", "w") as f:
    f.write("\n\n".join(out))
    f.write("\n\nif __name__ == \"__main__\":\n    app.run()\n")
