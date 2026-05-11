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

    con_pt_bins = np.asarray((0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, np.inf), dtype=np.float32)
    con_dr_bins = np.asarray((0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, np.inf), dtype=np.float32)


@app.function
@nb.jit(nopython=True, parallel=True)
def get_con_pt_dr_bins(jets, constituents):  
    out = np.zeros((len(jets), 2, con_pt_bins.shape[0] - 1, con_dr_bins.shape[0] - 1), dtype=np.float32)
    for ijet in nb.prange(len(jets)):
        for constit in constituents[ijet]:
            dr = jets[ijet].deltaR(constit)
            pt_bin = np.searchsorted(con_pt_bins, constit.pt) - 1
            dr_bin = np.searchsorted(con_dr_bins, dr) - 1
            out[ijet][0][pt_bin][dr_bin] += 1
            out[ijet][1][pt_bin][dr_bin] += constit.pt * dr

    return out


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
def _(process_table):
    def preprocess_data(source_dir, file_name, **kwargs):
        # input_path = os.path.join(source_dir, "data.arrow")
        input_path = os.path.join(source_dir, file_name)
        buffer = pa.memory_map(input_path, "rb")
        output_rb = process_table(pa.ipc.open_file(buffer).read_all(), **kwargs)

        output_path = os.path.join(source_dir, f"preproc_{file_name}")
        with pa.OSFile(output_path, "wb") as sink:
            with pa.ipc.new_file(sink, output_rb.schema) as writer:
                writer.write_batch(output_rb)

        # return output_path
    return


@app.cell
def _():
    _py8_dir = "./datasets/STAR_pp200GeV_production_2012"
    _py8_file = "Pythia8_pp200GeV.arrow"
    _py8_path = os.path.join(_py8_dir, _py8_file)
    _py8_buffer = pa.memory_map(_py8_path, "rb")
    _py8_table = pa.ipc.open_file(_py8_buffer).read_all()

    _py8_ak_array = ak.from_arrow(
            _py8_table,
            generate_bitmasks=True,
        )
    _jets, _constituents = to_jet_and_consitit_vectors(_py8_ak_array)

    _jets = calculate_angularities(_jets, _constituents)

    _con_pt_dr_bins = get_con_pt_dr_bins(_jets, _constituents)

    #print( _con_pt_dr_bins[0], _jets[0].deltaR(_constituents[0]), _constituents[0].pt)
    _g = np.sum(_con_pt_dr_bins, axis=(1, 2))/(_jets.pt*jet_r)

    _discrepancies = np.arange(len(_g))[~np.isclose(_jets["ch_ang_k1_b1"], _g, rtol=1e-3)]
    print(_jets["ch_ang_k1_b1"][_discrepancies])
    print(_g[_discrepancies])

    print(_jets["ch_ang_k1_b1"][_discrepancies] - _g[_discrepancies])
    del _jets, _constituents, _con_pt_dr_bins
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
