import numpy as np
from sklearn.model_selection import train_test_split
from hepstats.modeling import bayesian_blocks


def make_hist(
    array,
    weight=None,
    bins=None,
    range=(None, None),
    size_for_bins=0.02,
    p0=0.05,
    shuffle=False,
    stratify=None
):
    if weight is None:
        weight = np.ones_like(array)
    else:
        weight = weight / np.mean(weight)

    mask = np.full_like(array, True, dtype=np.bool)
    if range[0] is not None:
        mask = mask & (array > range[0])
    if range[1] is not None:
        mask = mask & (array < range[1])

    #print(type(mask), mask.dtype, mask[0:20])

    array = array[mask]
    weight = weight[mask]
    if stratify is not None:
        stratify = stratify[mask]

    jetIndices = np.arange(len(array))
    _, bin_est_ids = train_test_split(jetIndices, test_size=size_for_bins, shuffle=shuffle, stratify=stratify)

    if bins is None:
        bins = bayesian_blocks(
            array[bin_est_ids],
            weights=weight[bin_est_ids],
            p0=p0,
        )  # [1:]
        # print("Bins using bayesian blocks:", bins)
    binWidths = bins[1:] - bins[:-1]

    hCounts, _ = np.histogram(array, bins=bins, weights=weight)
    hSumw2, _ = np.histogram(array, bins=bins, weights=weight * weight)

    hErr = np.sqrt(hSumw2)

    sumBinCounts = np.sum(hCounts)
    binScale = 1.0 / (sumBinCounts * binWidths)
    binCountScaled = hCounts * binScale
    binError = hErr * binScale

    return bins, binCountScaled, binError


def make_profile(
    x, y, weight=None, bins=None, npoint_for_bins=30000, p0=0.05, shuffle=False
):
    if weight is None:
        weight = np.ones_like(x)
    else:
        weight = weight / np.mean(weight)

    jetIndices = np.arange(len(x))
    if shuffle:
        np.random.default_rng().shuffle(jetIndices)

    if bins is None:
        bins = bayesian_blocks(
            x[jetIndices[0:npoint_for_bins]],
            weights=weight[jetIndices[0:npoint_for_bins]],
            p0=p0,
        )  # [1:]

    hSumy, _ = np.histogram(x, bins=bins, weights=y * weight)
    hSumw, _ = np.histogram(x, bins=bins, weights=weight)

    hMeany = hSumy / (hSumw + 1e-8)

    hSumy2, _ = np.histogram(x, bins=bins, weights=weight * y * y)
    hMeany2 = hSumy2 / (hSumw + 1e-8)
    hVary = hMeany2 - hMeany * hMeany

    hErry = np.sqrt(hVary)

    return bins, hMeany, hErry
