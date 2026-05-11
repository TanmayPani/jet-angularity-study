import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import os

    from tqdm import tqdm
    import numba as nb
    import numpy as np

    import awkward as ak
    import pyarrow as pa
    import vector
    import fastjet as fj

    import matplotlib.pyplot as plt

    from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter

    vector.register_awkward()

    jet_r = 0.4

    con_pt_bins = np.asarray((0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0), dtype=np.float32)
    con_dr_bins = np.asarray((0.0,  0.05,  0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0), dtype=np.float32)


@app.cell
def _():
    _binned_vec_sum = np.zeros(
            (10, con_pt_bins.shape[0] - 1, con_dr_bins.shape[0] - 1), 
            dtype=[('px', np.float32), ('py', np.float32), ('pz', np.float32), ('E', np.float32)],
        ).view(vector.MomentumNumpy4D)


    #_binned_vec_sum = _binned_vec_sum_arr.view(vector.MomentumNumpy4D)
    #binned_vec_sum = binned_vec_sum + 1
    _jet = vector.obj(px=np.float32(1.1), py=np.float32(1.1), pz=np.float32(1.1), E=np.float32(1.1))
    _binned_vec_sum[0][0][0] = _binned_vec_sum[0][0][0] + _jet
    #_binned_vec_sum = _binned_vec_sum + _jet

    print(_binned_vec_sum.shape)
    print(np.sum(_binned_vec_sum, axis=(1,2)).pt)
    return


@app.function
@nb.jit(nopython=True)
#@nb.jit(nopython=True, parallel=True)
def get_con_pt_dr_bins(jets, constits):
    px_binned_sum = np.zeros((len(jets), con_pt_bins.shape[0] - 1, con_dr_bins.shape[0] - 1), dtype=np.float32)
    py_binned_sum = np.zeros((len(jets), con_pt_bins.shape[0] - 1, con_dr_bins.shape[0] - 1), dtype=np.float32)
    #out_py = np.zeros((len(jets), con_pt_bins.shape[0] - 1, con_dr_bins.shape[0] - 1), dtype=np.float32)
    for ijet in range(len(jets)):
        tot_ch_px = 0.
        tot_ch_py = 0.
        for iconstit, constit in enumerate(constits[ijet]):
            if constits.charge[ijet][iconstit] != 0:
                dr = jets[ijet].deltaR(constit)
                dr_bin = np.searchsorted(con_dr_bins, dr) - 1
                pt_bin = np.searchsorted(con_pt_bins, constit.pt) - 1 
                px_binned_sum[ijet][pt_bin][dr_bin] += constit.px
                py_binned_sum[ijet][pt_bin][dr_bin] += constit.py
                tot_ch_px += constit.px
                tot_ch_py += constit.py
        tot_ch_pt = np.sqrt(tot_ch_px**2 + tot_ch_py**2)
        px_binned_sum[ijet] /= tot_ch_pt
        py_binned_sum[ijet] /= tot_ch_pt
    pt_binned_sum = np.sqrt(px_binned_sum**2 + py_binned_sum**2)         
    return pt_binned_sum


@app.function
def to_jet_and_consitit_vectors(arr):
    jets = ak.zip(
        dict(
            pt=ak.enforce_type(arr.pt, "float32"),
            eta=ak.enforce_type(arr.eta, "float32"),
            phi=ak.enforce_type(arr.phi, "float32"),
            e=ak.enforce_type(arr.e, "float32"),
            weight=ak.enforce_type(arr.weight, "float32"),
            ncharged=ak.enforce_type(arr.ncharged, "uint8"),
            nconstituents=ak.enforce_type(arr.nconstituents, "uint8"),
        ),
        with_name="Momentum4D",
    )

    jets["m"] = jets.m

    constituents = ak.zip(
        {
            key: ak.enforce_type(
                arr[f"constit_{key}"],
                "var*float32" if key != "charge" else "var*int8",
            )
            for key in ("pt", "eta", "phi", "e", "charge")
        },
        with_name="Momentum4D",
    )

    return jets, constituents


@app.function
def calculate_angularities(jets, constituents):
    is_constit_charged = ak.fill_none(
        ak.mask(ak.ones_like(constituents.pt), constituents.charge != 0), 0
    )

    factors = {}
    factors["k1"] = constituents.pt / jets.pt
    factors["k2"] = constituents.pt2 / jets.pt2
    factors["b2"] = constituents.deltaR2(jets) / (jet_r * jet_r)
    factors["b1"] = np.sqrt(factors["b2"])
    factors["b0.5"] = np.sqrt(factors["b1"])
    factors["b0"] = 1

    jets["nef"] = ak.enforce_type(
        ak.nansum(
            ak.fill_none(ak.mask(factors["k1"], constituents.charge == 0), 0),
            axis=-1,
        ),
        "float32",
    )

    #for kappa, beta in ((1, 0), (1, 0.5), (1, 1), (1, 2), (2, 0)):
    for kappa, beta in ((1, 1),):
        factors[f"k{kappa}_b{beta}"] = (
            #is_constit_charged * factors[f"k{kappa}"] * factors[f"b{beta}"]
            factors[f"k{kappa}"] * factors[f"b{beta}"]
        )
        jets[f"ch_ang_k{kappa}_b{beta}"] = ak.enforce_type(
            ak.nansum(factors[f"k{kappa}_b{beta}"], axis=-1),
            "float32",
        )


    return jets


@app.cell
def _():
    #_py8_dir = "./datasets/STAR_pp200GeV_production_2012"
    #_py8_file = "Pythia8_pp200GeV.arrow"
    _py8_dir = "/home/tanmaypani/star-workspace/mc_generators/pythia8/outputs"
    _py8_file = "pythia8_detroit_tune.arrow"
    _py8_path = os.path.join(_py8_dir, _py8_file)
    _py8_buffer = pa.memory_map(_py8_path, "rb")
    _py8_table = pa.ipc.open_file(_py8_buffer).read_all()

    _py8_ak_array = ak.from_arrow(
            _py8_table,
            generate_bitmasks=True,
        )
    jets, constituents = to_jet_and_consitit_vectors(_py8_ak_array)

    _jetPtCut = jets.pt > 15  # & (jets.pt < jet_pt_max)
    #_jetEtaCut = np.abs(jets.eta) < 0.6
    #jets["ncharged"] = ak.count_nonzero(constituents.charge, axis=-1)
    #jets["nconstituents"] = ak.count(constituents, axis=-1)
    _jetNChargedCut = jets.ncharged > 1
    #_jetCut = _jetPtCut & _jetEtaCut & _jetNChargedCut
    _jetCut = _jetPtCut & _jetNChargedCut
    jets = ak.drop_none(ak.mask(jets, _jetCut), axis=0)
    constituents = ak.drop_none(ak.mask(constituents, _jetCut), axis=0)
    #_jets = calculate_angularities(_jets, _constituents)
    return constituents, jets


@app.cell
def _(constituents, jets):
    #con_pt_dr_bins = get_con_pt_dr_bins(jets, constituents)[(20 < jets.pt) & (jets.pt < 40)]
    con_pt_dr_bins = get_con_pt_dr_bins(jets, constituents)

    jet_wts = jets.weight.to_numpy()
    #


    rho_hist = np.zeros(con_pt_dr_bins.shape[1:], dtype=np.float32)
    for _chunk in np.array_split(np.arange(con_pt_dr_bins.shape[0]), 100000, axis=0):
        rho_hist += np.tensordot(con_pt_dr_bins[_chunk, ...], jet_wts[_chunk], axes=(0, 0)) 

    _con_pt_bin_sizes = con_pt_bins[1:] - con_pt_bins[:-1]
    _con_dr_bin_sizes = con_dr_bins[1:] - con_dr_bins[:-1]
    _bin_sizes = np.ones((_con_pt_bin_sizes.shape[0], 1)) @ _con_dr_bin_sizes[np.newaxis, ...]

    #con_pt_bins = np.asarray((0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, np.inf), dtype=np.float32)

    #rho_hist /= _bin_sizes
    rho_hist /= np.sum(jets.weight)
    #rho_hist /= np.sum(rho_hist)
    return (rho_hist,)


@app.cell
def _(rho_hist):
    _fig = plt.figure(figsize=(8, 8))
    _ax = _fig.add_subplot()
    _dr = 0.5*(con_dr_bins[1:] + con_dr_bins[:-1])
    #for _row in range(rho_hist.shape[0]):
    #    _ax.plot(_dr, rho_hist[_row], label=f"bin #{_row}")
    _ax.plot(_dr[:-2], np.sum(rho_hist[:-1, :-2], axis=0), "o", color="red")
    _ax.stackplot(_dr[:-2], rho_hist[:-1, :-2])
    #_ax.set_yscale("log")
    #con_pt_bins = np.asarray((0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 30.0, 100.0), dtype=np.float32)

    _fig
    return


@app.cell
def _(rho_hist):
    _fig, _axs = plt.subplots(2, 5, figsize=(20, 8))
    _dr = 0.5*(con_dr_bins[1:] + con_dr_bins[:-1])
    _flat_axs = _axs.flatten()
    for _row in range(rho_hist.shape[0]):
        #print(_row//3, _row%3)
        _flat_axs[_row].plot(_dr, rho_hist[_row], label=f"bin #{_row}")
    #_ax.plot(_dr, np.sum(rho_hist, axis=0), "o", color="red")
    #_ax.stackplot(_dr, rho_hist)
    #_axs.legend()
    _fig
    return


@app.cell
def _(jets):
    _fig = plt.figure(figsize=(8, 8))
    _ax = _fig.add_subplot()
    _ax.hist(jets.pt, bins=20, weights=jets.weight, density=True)
    _ax.set_yscale("log")
    #_ax.legend()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
