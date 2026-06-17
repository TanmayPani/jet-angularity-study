import re
import textwrap

with open("/home/tanmaypani/star-workspace/jet-angularity-study/plot_physics.py") as f:
    plot_physics_code = f.read()

with open("/home/tanmaypani/star-workspace/jet-angularity-study/plot_hp2026_prelims.py") as f:
    hp2026_code = f.read()

# 1. Extract setups
pp_setup = re.search(r'with app\.setup:\n(.*?)(?=\n@app\.function|\n@app\.cell|\Z)', plot_physics_code, re.DOTALL).group(1)
hp_setup = re.search(r'with app\.setup:\n(.*?)(?=\n@app\.function|\n@app\.cell|\Z)', hp2026_code, re.DOTALL).group(1)

# Remove imports from plot_physics in hp_setup
hp_setup = re.sub(r'from plot_physics import \([^)]+\)', '', hp_setup, flags=re.DOTALL)
hp_setup = re.sub(r'from plot_physics import .*', '', hp_setup)

combined_setup = textwrap.dedent(pp_setup).strip() + "\n\n" + textwrap.dedent(hp_setup).strip()

# 2. Extract helper functions
pp_funcs = []
for m in re.finditer(r'@app\.function\n(def .*?(?=\n@app\.function|\n@app\.cell|\Z))', plot_physics_code, re.DOTALL):
    pp_funcs.append(m.group(1).strip())

hp_funcs = []
for m in re.finditer(r'@app\.function\n(def .*?(?=\n@app\.function|\n@app\.cell|\Z))', hp2026_code, re.DOTALL):
    code = m.group(1).strip()
    # Skip drivers
    if code.startswith("def make_single_dist"): continue
    if code.startswith("def make_profile_multi"): continue
    if code.startswith("def make_grid_profiles"): continue
    hp_funcs.append(code)

out = []
out.append(f"""@app.cell\ndef _():\n{textwrap.indent(combined_setup, '    ')}\n    return\n""")

# We will put all helper functions in one big cell so they don't clutter the workspace with 20 cells
all_helpers = "\n\n".join(pp_funcs + hp_funcs)
# For marimo, each cell defines its top-level variables. If we put all defs in one cell, they are globally available.
out.append(f"""@app.cell\ndef _(CENTER_JPT, FormatStrFormatter, Line2D, MaxNLocator, Path, SysVar, feature_mode, figsize, get_jet_pt_bins, jpt_bins, jpt_bins_to_omit, mc_hist_styles, mc_labels, mc_proxy_handles, np, plot_ratio_sys_err, plot_sys_err, prefix_dir, pt_label, pt_true, sharey_for, torch, var_hist_ylabel, var_prof_ratio_ylim, var_prof_ylim, var_xlim):\n{textwrap.indent(all_helpers, '    ')}\n    return\n""")

# 3. Generate individual cells for the 13 plots

# A. Distributions
for var in ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0"]:
    cell_code = f"""
    _var_name = "{var}"
    fig_dist_{var.replace('.', '_')} = plt.figure(figsize=(6.5, 7.5))
    _axs = fig_dist_{var.replace('.', '_')}.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    draw_dist_column(_axs, _var_name, CENTER_JPT, show_legend=True, show_prelim=True, show_pt=True)
    _axs[0].set_ylabel(var_hist_ylabel[_var_name], fontsize="xx-large")
    _axs[1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _axs[2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    # Render figure in notebook
    fig_dist_{var.replace('.', '_')}
    """
    out.append(f"""@app.cell\ndef _(CENTER_JPT, draw_dist_column, plt, var_hist_ylabel):\n{cell_code}\n    return fig_dist_{var.replace('.', '_')},\n""")

# B. dR profiles
for var in ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2"]:
    cell_code = f"""
    _var_name = "{var}"
    _x_var_name = "sd_dR"
    _jpt_trues = [CENTER_JPT]
    
    fig_prof_{var.replace('.', '_')} = plt.figure(figsize=(6, 7.5))
    _axs_raw = fig_prof_{var.replace('.', '_')}.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    _col_axs = [_axs_raw]
    
    for _k, _jpt_true in enumerate(_jpt_trues):
        draw_profile_column(
            _col_axs[_k], _var_name, _x_var_name, _jpt_true,
            show_legend=(_k == 0),
            show_prelim=(_k == 0),
            show_pt=(_jpt_true != _jpt_trues[0] or _k == 0),
        )
        _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    
    _col_axs[0][1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_prof_{var.replace('.', '_')}
    """
    out.append(f"""@app.cell\ndef _(CENTER_JPT, draw_profile_column, plt, var_prof_ylabel):\n{cell_code}\n    return fig_prof_{var.replace('.', '_')},\n""")

# C. zg profiles
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
        _axs_raw = fig_zg_{suffix}.subplots(
            3, 1,
            height_ratios=[3, 1, 1],
            sharex=True,
            gridspec_kw=dict(hspace=0),
        )
        _col_axs = [_axs_raw]
    else:
        _axs_raw = fig_zg_{suffix}.subplots(
            3, _n,
            height_ratios=[3, 1, 1],
            sharey="row",
            sharex="col",
            squeeze=False,
            gridspec_kw=dict(hspace=0, wspace=0),
        )
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]
        
    for _k, _jpt_true in enumerate(_jpt_trues):
        draw_profile_column(
            _col_axs[_k], _var_name, _x_var_name, _jpt_true,
            show_legend=(_k == 0),
            show_prelim=(_k == 0),
            show_pt=(_jpt_true != _jpt_trues[0] or _k == 0),
        )
        _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
        
    _col_axs[0][1].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_zg_{suffix}
    """
    out.append(f"""@app.cell\ndef _(draw_profile_column, plt, var_prof_ylabel):\n{cell_code}\n    return fig_zg_{suffix},\n""")

# D. Grid
cell_code = f"""
    _kappa1 = ("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2")
    _n = len(_kappa1)
    fig_grid = plt.figure(figsize=(6 * _n, 7.5))
    _axs = fig_grid.subplots(
        3, _n,
        height_ratios=[3, 1, 1],
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw=dict(hspace=0, wspace=0.30),
    )

    for _k, _var_name in enumerate(_kappa1):
        draw_profile_column(
            _axs[:, _k], _var_name, "sd_dR", CENTER_JPT,
            show_legend=(_k == 0),
            show_prelim=(_k == 0),
            show_pt=(_k == 0),
        )
        _axs[0, _k].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")

    _axs[1, 0].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(incl.)}}$", fontsize="x-large")
    _axs[2, 0].set_ylabel(r"$\\frac{{\\mathrm{{MC}}}}{{\\mathrm{{Data}}}}\\,\\mathrm{{(SD)}}$", fontsize="x-large")
    
    fig_grid
"""
out.append(f"""@app.cell\ndef _(CENTER_JPT, draw_profile_column, plt, var_prof_ylabel):\n{cell_code}\n    return fig_grid,\n""")

# Output the file
with open("/home/tanmaypani/star-workspace/jet-angularity-study/plot_hp2026_prelims_target.py", "w") as f:
    f.write("import marimo\n\n__generated_with = \"0.23.5\"\napp = marimo.App(width=\"full\")\n\n")
    f.write("\n\n".join(out))
    f.write("\n\nif __name__ == \"__main__\":\n    app.run()\n")
