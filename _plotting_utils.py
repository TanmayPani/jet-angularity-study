# from matplotlib import pyplot as plt
import numpy as np


def plot_hist(ax, bins, counts, errors=None, bin_range=(0, None), **kwargs):
    binWidths = bins[1:] - bins[:-1]
    binCenters = (bins[1:] + bins[:-1]) * 0.5

    if bin_range[0] > 0 or bin_range[1] is not None:
        if bin_range[1] is None:
            binWidths = binWidths[bin_range[0] :]
            binCenters = binCenters[bin_range[0] :]
            counts = counts[bin_range[0] :]
            errors = errors[bin_range[0] :]
        else:
            binCenters = binCenters[bin_range[0] : bin_range[1]]
            binWidths = binWidths[bin_range[0] : bin_range[1]]
            counts = counts[bin_range[0] : bin_range[1]]
            errors = errors[bin_range[0] : bin_range[1]]

    # print(len(binCenters), len(counts), len(binWidths), len(errors))
    # print(binCenters, counts, binWidths, errors)
    ax.errorbar(binCenters, counts, xerr=binWidths * 0.5, yerr=errors, **kwargs)
    return binCenters, binWidths, counts, errors


def plot_ratios(
    fig,
    bins,
    counts1,
    counts2,
    ratios,
    errors1,
    errors2,
    ratio_errs,
    labels1=None,
    labels2=None,
    fill_style="none",
    line_style="none",
    markersize=10,
    markers1=None,
    markers2=None,
):
    axs = fig.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        gridspec_kw={"hspace": 0, "top": 0.85, "bottom": 0.1},
    )

    for iden, (count, error) in enumerate(zip(counts2, errors2)):
        label = str(iden) if labels2 is None else labels2[iden]
        marker = "o" if markers2 is None else markers2[iden]
        #print(iden, label, marker)
        plot_hist(
            axs[0],
            bins,
            count,
            errors=error,
            label=label,
            marker=marker,
            fillstyle=fill_style,
            linestyle=line_style,
            markersize=markersize,
        )

    for inum, (count1, error1, ratios1, ratio_errs1) in enumerate(
        zip(counts1, errors1, ratios, ratio_errs)
    ):
        label1 = str(inum) if labels1 is None else labels1[inum]
        marker1 = "o" if markers1 is None else markers1[inum]
        plot_hist(
            axs[0],
            bins,
            count1,
            errors=error1,
            label=label1,
            marker=marker1,
            fillstyle=fill_style,
            linestyle=line_style,
            markersize=markersize,
        )

        for iden, (count2, error2, ratio, ratio_err) in enumerate(
            zip(counts2, errors2, ratios1, ratio_errs1)
        ):
            label2 = str(iden) if labels2 is None else labels2[iden]
            marker2 = "o" if markers2 is None else markers2[iden]
            #print(inum, iden, label1, label2, marker1, marker2)
            if ratio is None:
                ratio = count1 / count2
                if error1 is not None and error2 is not None:
                    ratio_err = ratio * np.sqrt(
                        (error1 / count1) ** 2 + (error2 / count2) ** 2
                    )
            plot_hist(
                axs[1],
                bins,
                ratio,
                errors=ratio_err,
                label=f"{label1}/{label2}",
                marker=marker2,
                fillstyle=fill_style,
                linestyle=line_style,
                markersize=markersize,
            )
        axs[0].set_yscale("log")
        axs[0].tick_params(axis="x", direction="in")
        axs[1].set_ylim(0, 2)
        axs[1].axhline(y=1, color="red", linestyle="--")
        axs[1].axhspan(0.8, 1.2, color="darkgrey")
    return axs
