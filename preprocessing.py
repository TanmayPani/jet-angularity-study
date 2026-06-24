from pathlib import Path

from tqdm import tqdm
import numba as nb
import numpy as np

import awkward as ak
import pyarrow as pa
import vector
import fastjet as fj
import torch
from tensordict import TensorDict

from systematics import SysVar  # , get_jet_pt_bins, get_unfolding_iter

vector.register_awkward()

pth_bins = (
    "11",
    "15",
    "20",
    "25",
    "35",
    "45",
    "55",
    "infty",
)
jet_columns = (
    "pt",
    "eta",
    "phi",
    "m",
    "nef",
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1",
    "ch_ang_k1_b2",
    "ch_ang_k2_b0",
    "leading_constit_pt",
    "leading_constit_eta",
    "leading_constit_phi",
    "subleading_constit_pt",
    "subleading_constit_eta",
    "subleading_constit_phi",
    "sd_pt",
    "sd_eta",
    "sd_phi",
    "sd_m",
    "sd_dR",
    "sd_symmetry",
    "sd_ch_ang_k1_b0.5",
    "sd_ch_ang_k1_b1",
    "sd_ch_ang_k1_b2",
    "sd_ch_ang_k2_b0",
)
# Input-column list for the `angularities_noptd` mode: identical to `jet_columns`
# but with both p_T^D (k2_b0) angularities dropped from the model input. The arrow
# files still carry these columns; this only controls what `to_tensordict` feeds.
jet_columns_noptd = tuple(
    c for c in jet_columns if c not in ("ch_ang_k2_b0", "sd_ch_ang_k2_b0")
)
# Input-column list for the `angularities_minimal` mode: `angularities` minus the
# four observables M (`m`), M_g (`sd_m`), R_g (`sd_dR`) and p_T^D (both the
# ungroomed `ch_ang_k2_b0` and groomed `sd_ch_ang_k2_b0` variants). As with
# `jet_columns_noptd`, the arrow files still carry these columns; this only
# controls what `to_tensordict` feeds the model.
jet_columns_minimal = tuple(
    c
    for c in jet_columns
    if c not in ("m", "sd_m", "sd_dR", "ch_ang_k2_b0", "sd_ch_ang_k2_b0")
)
jet_r = 0.4

con_pt_bins = np.asarray(
    (0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0), dtype=np.float32
)
con_dr_bins = np.asarray(
    (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0), dtype=np.float32
)
N_PT = con_pt_bins.shape[0] - 1
N_DR = con_dr_bins.shape[0] - 1
N_BINS = N_PT * N_DR

# --- old ---
# FEATURE_MODES = ("angularities", "bin_counts", "combined", "kinematics")
# FEATURE_MODES = ("angularities", "angularities_noptd", "bin_counts", "combined", "kinematics")
# `angularities_noptd` / `angularities_minimal` reuse the angularities code paths
# (scalar `to_tensordict` branch, MLP classifier) but drop observables from the model
# input via `jet_columns_noptd` (p_T^D) / `jet_columns_minimal` (p_T^D, R_g, M_g, M).
FEATURE_MODES = (
    "angularities",
    "angularities_noptd",
    "angularities_minimal",
    "bin_counts",
    "combined",
    "kinematics",
)


@nb.jit
def match_sd(builder, sd_jet_constits, jet_constits):
    for sd_constits, constits in zip(sd_jet_constits, jet_constits):
        builder.begin_list()
        sd_constit_idx = list(range(len(sd_constits)))
        constit_idx = list(range(len(constits)))
        for sd_con in sd_constit_idx:
            is_matched = False
            for con in constit_idx:
                if sd_constits[sd_con].isclose(constits[con]):
                    is_matched = True
                    builder.append(con)
                    constit_idx.remove(con)
                    break
        builder.end_list()
    return builder


@nb.jit(nopython=True)
def get_con_pt_dr_bins(jets, constits):
    n_pt = con_pt_bins.shape[0] - 1
    n_dr = con_dr_bins.shape[0] - 1
    px_sum = np.zeros((len(jets), n_pt, n_dr), dtype=np.float32)
    py_sum = np.zeros((len(jets), n_pt, n_dr), dtype=np.float32)
    for ijet in range(len(jets)):
        tot_ch_px = np.float32(0.0)
        tot_ch_py = np.float32(0.0)
        for iconstit, constit in enumerate(constits[ijet]):
            if constits.charge[ijet][iconstit] != 0:
                tot_ch_px += constit.px
                tot_ch_py += constit.py
                dr = jets[ijet].deltaR(constit)
                dr_bin = np.searchsorted(con_dr_bins, dr) - 1
                pt_bin = np.searchsorted(con_pt_bins, constit.pt) - 1
                if 0 <= pt_bin < n_pt and 0 <= dr_bin < n_dr:
                    px_sum[ijet, pt_bin, dr_bin] += constit.px
                    py_sum[ijet, pt_bin, dr_bin] += constit.py
        tot_ch_pt = np.sqrt(tot_ch_px * tot_ch_px + tot_ch_py * tot_ch_py)
        if tot_ch_pt > np.float32(0.0):
            px_sum[ijet] /= tot_ch_pt
            py_sum[ijet] /= tot_ch_pt
    return np.sqrt(px_sum * px_sum + py_sum * py_sum)


def get_softdrop_groomed_jets(constituents, jet_def, **kwargs):
    sd_clus_seq = fj.ClusterSequence(constituents, jet_def)
    sd_jet_data = sd_clus_seq.exclusive_jets_softdrop_grooming(**kwargs)
    sd_constit_data = sd_jet_data.constituents

    sd_jets = ak.zip(
        dict(
            pt=ak.enforce_type(sd_jet_data.ptsoftdrop, "float32"),
            eta=ak.enforce_type(sd_jet_data.etasoftdrop, "float32"),
            phi=ak.enforce_type(sd_jet_data.phisoftdrop, "float32"),
            e=ak.enforce_type(sd_jet_data.Esoftdrop, "float32"),
            m=ak.enforce_type(sd_jet_data.msoftdrop, "float32"),
            pz=ak.enforce_type(sd_jet_data.pzsoftdrop, "float32"),
            dR=ak.enforce_type(sd_jet_data.deltaRsoftdrop, "float32"),
            symmetry=ak.enforce_type(sd_jet_data.symmetrysoftdrop, "float32"),
            nconstituents=ak.enforce_type(ak.count(sd_constit_data.E, axis=1), "uint8"),
        ),
        with_name="Momentum4D",
    )

    sd_constit_vecs = ak.zip(
        dict(
            px=sd_constit_data.px,
            py=sd_constit_data.py,
            pz=sd_constit_data.pz,
            e=sd_constit_data.E,
        ),
        with_name="Momentum4D",
    )

    sd_constit_sorted_idx = ak.argsort(
        sd_constit_vecs.pt,
        axis=-1,
        ascending=False,
    )
    sd_constit_vecs = sd_constit_vecs[sd_constit_sorted_idx]
    sd_constit_indices = match_sd(
        ak.ArrayBuilder(), sd_constit_vecs, constituents
    ).snapshot()
    sd_constituents = constituents[sd_constit_indices]

    sd_jets["ncharged"] = ak.enforce_type(
        ak.count_nonzero(sd_constituents.charge, axis=-1), "uint8"
    )

    return sd_jets, sd_constituents


@nb.jit(nopython=True)
def _build_ang_sums_sparse(
    jets,
    constits,
    bin_idx_builder,
    ang_sums_builder,
    count_builder,
    bin_idx_builder_neutral,
    count_builder_neutral,
):
    n_pt = con_pt_bins.shape[0] - 1
    n_dr = con_dr_bins.shape[0] - 1
    # Single reused buffers — O(1) memory regardless of n_jets.
    # jet_buf channel layout (axis=-1):
    #   0: sum(pT_i)           -> k=1, b=0
    #   1: sum(pT_i * dR^0.5) -> k=1, b=0.5  (LHA)
    #   2: sum(pT_i * dR)     -> k=1, b=1    (girth)
    #   3: sum(pT_i * dR^2)   -> k=1, b=2    (thrust)
    #   4: sum(pT_i^2)        -> k=2, b=0    (p_T^D^2)
    # To recover angularity: sum_bins(channel) / (pT_jet^kappa * jet_r^beta)
    # jet_count is the per-cell CHARGED count; jet_count_neutral the NEUTRAL count
    # (charge == 0). The two feed the two channels of the bin-image input.
    jet_buf = np.zeros((n_pt * n_dr, 5), dtype=np.float32)
    jet_count = np.zeros((n_pt * n_dr,), dtype=np.int32)
    jet_count_neutral = np.zeros((n_pt * n_dr,), dtype=np.int32)
    for ijet in range(len(jets)):
        for cell in range(n_pt * n_dr):
            jet_count[cell] = 0
            jet_count_neutral[cell] = 0
            for ch in range(5):
                jet_buf[cell, ch] = np.float32(0.0)
        for iconstit, constit in enumerate(constits[ijet]):
            dr = jets[ijet].deltaR(constit)
            dr_bin = np.searchsorted(con_dr_bins, dr) - 1
            pt_bin = np.searchsorted(con_pt_bins, constit.pt) - 1
            if not (0 <= pt_bin < n_pt and 0 <= dr_bin < n_dr):
                continue
            cell = pt_bin * n_dr + dr_bin
            if constits.charge[ijet][iconstit] != 0:
                jet_count[cell] += 1
                cpt = constit.pt
                jet_buf[cell, 0] += cpt
                jet_buf[cell, 1] += cpt * np.sqrt(dr)
                jet_buf[cell, 2] += cpt * dr
                jet_buf[cell, 3] += cpt * dr * dr
                jet_buf[cell, 4] += cpt * cpt
            else:
                jet_count_neutral[cell] += 1
        bin_idx_builder.begin_list()
        ang_sums_builder.begin_list()
        count_builder.begin_list()
        for cell in range(n_pt * n_dr):
            if jet_count[cell] > 0:
                bin_idx_builder.integer(cell)
                count_builder.integer(jet_count[cell])
                for ch in range(5):
                    ang_sums_builder.real(jet_buf[cell, ch])
        bin_idx_builder.end_list()
        ang_sums_builder.end_list()
        count_builder.end_list()
        bin_idx_builder_neutral.begin_list()
        count_builder_neutral.begin_list()
        for cell in range(n_pt * n_dr):
            if jet_count_neutral[cell] > 0:
                bin_idx_builder_neutral.integer(cell)
                count_builder_neutral.integer(jet_count_neutral[cell])
        bin_idx_builder_neutral.end_list()
        count_builder_neutral.end_list()
    return (
        bin_idx_builder,
        ang_sums_builder,
        count_builder,
        bin_idx_builder_neutral,
        count_builder_neutral,
    )


def get_con_pt_dr_bins_sparse(jets, constituents):
    (
        bin_idx_builder,
        ang_sums_builder,
        count_builder,
        bin_idx_builder_neutral,
        count_builder_neutral,
    ) = _build_ang_sums_sparse(
        jets,
        constituents,
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
    )

    jets["bin_index"] = bin_idx_builder.snapshot()
    jets["bin_count"] = count_builder.snapshot()
    jets["bin_sum_wts"] = ang_sums_builder.snapshot()
    jets["bin_index_neutral"] = bin_idx_builder_neutral.snapshot()
    jets["bin_count_neutral"] = count_builder_neutral.snapshot()

    return jets


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

    for kappa, beta in ((1, 0), (1, 0.5), (1, 1), (1, 2), (2, 0)):
        factors[f"k{kappa}_b{beta}"] = (
            is_constit_charged * factors[f"k{kappa}"] * factors[f"b{beta}"]
        )
        jets[f"ch_ang_k{kappa}_b{beta}"] = ak.enforce_type(
            ak.nansum(factors[f"k{kappa}_b{beta}"], axis=-1),
            "float32",
        )
    return jets


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


def process_table(table, feature_mode, **extra_fields):
    ak_array = ak.from_arrow(
        table,
        generate_bitmasks=True,
    )
    jets, constituents = to_jet_and_consitit_vectors(ak_array)
    sd_jets, sd_constituents = (
        get_softdrop_groomed_jets(
            constituents,
            fj.JetDefinition(fj.antikt_algorithm, jet_r, fj.E_scheme),
            symmetry_cut=0.2,
            R0=jet_r,
        )
        if feature_mode != "bin_counts"
        else (None, None)
    )

    for coord in ("pt", "eta", "phi"):
        jets[f"leading_constit_{coord}"] = getattr(constituents, coord)[:, 0]
        jets[f"subleading_constit_{coord}"] = getattr(constituents, coord)[:, 1]

    match feature_mode:
        case "bin_counts":
            jets = get_con_pt_dr_bins_sparse(jets, constituents)
            # sd_jets = get_con_pt_dr_bins_sparse(sd_jets, sd_constituents)
        case "angularities":
            jets = calculate_angularities(jets, constituents)
            sd_jets = calculate_angularities(sd_jets, sd_constituents)
        case "combined":
            jets = get_con_pt_dr_bins_sparse(jets, constituents)
            sd_jets = get_con_pt_dr_bins_sparse(sd_jets, sd_constituents)
            # TODO: Add summing over jets["bin_sum_wts"] to calculate angularities here
        case _:
            raise ValueError(
                f"feature_mode input must be [bin_counts, angularities, combined] but got {feature_mode}!"
            )

    out_dict = {key: pa.array(getattr(jets, key)) for key in jets.fields}
    if sd_jets is not None:
        for key in sd_jets.fields:
            # print(key)
            out_dict[f"sd_{key}"] = pa.array(getattr(sd_jets, key))

    if len(extra_fields) > 0:
        print("------> Adding extra fields:", extra_fields)
        for key, val in extra_fields.items():
            broadcasted_arr, _ = ak.broadcast_arrays(val, jets.pt)
            out_dict[key] = pa.array(broadcasted_arr)

    return pa.RecordBatch.from_pydict(out_dict)


def preprocess_data(input_dir, output_dir, file_name, feature_mode, **kwargs):
    buffer = pa.memory_map(str(input_dir / file_name), "rb")
    output_rb = process_table(
        pa.ipc.open_file(buffer).read_all(), feature_mode, **kwargs
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output_dir / file_name), "wb") as sink:
        with pa.ipc.new_file(sink, output_rb.schema) as writer:
            writer.write_batch(output_rb)


def _densify_bin_counts(chunk):
    idxs = chunk.column("bin_index").to_pylist()
    cnts = chunk.column("bin_count").to_pylist()
    out = np.zeros((len(idxs), N_BINS), dtype=np.float32)
    for i, (ix, ct) in enumerate(zip(idxs, cnts)):
        if ix:
            out[i, ix] = ct
    return torch.from_numpy(out)


def _densify_bin_image(chunk):
    """Sparse (charged, neutral) per-cell counts -> dense (N, 2, N_PT, N_DR).

    Channel 0 is the charged constituent count (`bin_index`/`bin_count`), channel
    1 the neutral count (`bin_index_neutral`/`bin_count_neutral`). The flat cell
    index is `pt_bin * N_DR + dr_bin`, so reshaping the 81-vector to
    (N_PT, N_DR) places pT along the rows and dR along the columns.
    """
    ch_idx = chunk.column("bin_index").to_pylist()
    ch_cnt = chunk.column("bin_count").to_pylist()
    ne_idx = chunk.column("bin_index_neutral").to_pylist()
    ne_cnt = chunk.column("bin_count_neutral").to_pylist()
    n = len(ch_idx)
    # Per-cell constituent counts are small non-negative integers (observed max 8),
    # so store the (2,9,9) bin image as uint8: the part_lvl memmap is 18.8 GB in
    # float32 but only 4.7 GB in uint8 -> it stays in page cache instead of being
    # re-read from disk every undersampled epoch. The training/inference loop casts
    # the gathered batch to the compute dtype on-device (omnitrain._input_to_device).
    out = np.zeros((n, 2, N_BINS), dtype=np.uint8)
    for i in range(n):
        if ch_idx[i]:
            out[i, 0, ch_idx[i]] = ch_cnt[i]
        if ne_idx[i]:
            out[i, 1, ne_idx[i]] = ne_cnt[i]
    return torch.from_numpy(out.reshape(n, 2, N_PT, N_DR))


def to_tensordict(
    data_like,
    sim_like,
    columns=None,
    prefix=None,
    max_chunksize=None,
    feature_mode="angularities",
):
    if feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}"
        )

    table = pa.concat_tables((data_like, sim_like))
    target = torch.concatenate(
        (
            torch.ones(len(data_like), dtype=torch.float32),
            torch.zeros(len(sim_like), dtype=torch.float32),
        )
    )

    # `bin_index` and `bin_count` are list<int>-typed (variable per-jet length) and
    # break `to_pydict + zip` if mixed with scalar columns, so iterate them via a
    # separate sub-table and densify per chunk. The jet (scalar) columns iterate via
    # a fast `to_numpy` + np.stack path — both yield (chunk_size, k) float32 tensors
    # that get concatenated for the combined mode.
    # bin_counts is now a 2-channel (charged, neutral) 9x9 "image" fed to a CNN;
    # combined keeps the flat single-channel charged bin block.
    use_bin_image = feature_mode == "bin_counts"
    if feature_mode == "bin_counts":
        jet_input_columns: tuple[str, ...] = ()
        use_bin_block = True
        n_features = N_BINS
    elif feature_mode == "combined":
        jet_input_columns = (
            tuple(columns)
            if columns is not None
            else tuple(
                c for c in table.column_names if c not in ("bin_index", "bin_count")
            )
        )
        use_bin_block = True
        n_features = len(jet_input_columns) + N_BINS
    else:  # "angularities" / "kinematics"
        jet_input_columns = (
            tuple(columns) if columns is not None else tuple(table.column_names)
        )
        use_bin_block = False
        n_features = len(jet_input_columns)

    input_shape = (2, N_PT, N_DR) if use_bin_image else (n_features,)
    # The pure bin_counts image is small integer counts -> uint8 (see
    # _densify_bin_image); the combined/angularity modes mix in float features.
    input_dtype = torch.uint8 if use_bin_image else torch.float32
    tdict = (
        TensorDict(
            dict(
                input=torch.zeros(input_shape, dtype=input_dtype),
                target=torch.zeros((), dtype=torch.float32),
                weight=torch.ones((), dtype=torch.float32),
                is_data=torch.zeros((), dtype=torch.bool),
                is_matched=torch.zeros((), dtype=torch.int64),
                pth_bin=torch.zeros((), dtype=torch.int64),
            ),
            batch_size=[],
        )
        .expand(len(table))
        .memmap_like(prefix=prefix)
    )

    jet_table = table.select(list(jet_input_columns)) if jet_input_columns else None
    bin_columns = (
        ["bin_index", "bin_count", "bin_index_neutral", "bin_count_neutral"]
        if use_bin_image
        else ["bin_index", "bin_count"]
    )
    bin_table = table.select(bin_columns) if use_bin_block else None
    _target = torch.as_tensor(target, dtype=torch.float32)
    _weight = torch.as_tensor(table["weight"].to_numpy(), dtype=torch.float32)
    _is_data = torch.as_tensor(table["is_data"].to_numpy(), dtype=torch.bool)
    _is_matched = torch.as_tensor(table["is_matched"].to_numpy(), dtype=torch.int64)
    _pth_bin = torch.as_tensor(table["pth_bin"].to_numpy(), dtype=torch.int64)

    jet_batches = (
        jet_table.to_batches(max_chunksize=max_chunksize)
        if jet_table is not None
        else None
    )
    bin_batches = (
        bin_table.to_batches(max_chunksize=max_chunksize)
        if bin_table is not None
        else None
    )

    pos = 0
    pbar = tqdm(total=len(table))
    for batch_idx in range(
        max(
            len(jet_batches) if jet_batches is not None else 0,
            len(bin_batches) if bin_batches is not None else 0,
        )
    ):
        parts = []
        if jet_batches is not None:
            jc = jet_batches[batch_idx]
            cols = [
                jc.column(name)
                .to_numpy(zero_copy_only=False)
                .astype(np.float32, copy=False)
                for name in jet_input_columns
            ]
            parts.append(torch.from_numpy(np.stack(cols, axis=-1)))
        if bin_batches is not None:
            parts.append(
                _densify_bin_image(bin_batches[batch_idx])
                if use_bin_image
                else _densify_bin_counts(bin_batches[batch_idx])
            )
        chunk_size = parts[0].shape[0]
        input_tensor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

        _tdict = TensorDict(
            dict(
                input=input_tensor,
                target=_target[pos : pos + chunk_size],
                weight=_weight[pos : pos + chunk_size],
                is_data=_is_data[pos : pos + chunk_size],
                is_matched=_is_matched[pos : pos + chunk_size],
                pth_bin=_pth_bin[pos : pos + chunk_size],
            ),
            batch_size=(chunk_size,),
            device="cpu",
        )

        tdict[pos : pos + chunk_size] = _tdict
        pos += chunk_size
        pbar.update(chunk_size)

    return tdict


def preprocess_embedding_file(input_dir, output_dir, feature_mode, file_name):
    extra_fields = {}
    extra_fields["is_data"] = False
    extra_fields["is_matched"] = 1 if "matches" in file_name else 0
    outfile = output_dir / file_name
    print("Writing to file:", outfile)
    sink = pa.OSFile(str(outfile), "wb")
    writer = None
    njets = 0
    nbytes = 0
    for ipth, (pth_low, pth_high) in enumerate(zip(pth_bins[:-1], pth_bins[1:])):
        infile = input_dir / f"ptHat{pth_low}to{pth_high}" / file_name
        print("> Reading embedding file from:", infile)
        buffer = pa.memory_map(str(infile), "rb")
        extra_fields["pth_bin"] = ipth
        output_rb = process_table(
            pa.ipc.open_file(buffer).read_all(),
            feature_mode,
            **extra_fields,
        )
        writer = writer or pa.ipc.new_file(sink, output_rb.schema)
        writer.write_batch(output_rb)
        njets += len(output_rb)
        nbytes += output_rb.nbytes
        print(
            f"---> Processed {njets} jets, wrote {nbytes / (1024 * 1024):.2f} mb to file..."
        )
    if writer is not None:
        writer.close()


def preprocess_embedding(input_dir, output_dir, sysvar, feature_mode):
    _input_dir = input_dir / "embedding" / str(sysvar)
    if not _input_dir.exists():
        print(
            f"[preprocess_embedding] no embedding dir at {_input_dir}; "
            f"skipping {sysvar}. Run cluster_embedding.py for this sysvar first."
        )
        return

    _output_dir = output_dir / "embedding" / str(sysvar)
    _output_dir.mkdir(parents=True, exist_ok=True)
    for infile in ("gen-matches", "reco-matches", "misses", "fakes"):
        preprocess_embedding_file(
            _input_dir, _output_dir, feature_mode, f"{infile}.arrow"
        )


def replace_table_column(table, name, array, new_name=None, **kwargs):
    col_index = table.schema.get_field_index(name)
    col_name = new_name or name
    column = pa.array(array, **kwargs)
    return table.set_column(col_index, col_name, column)


def make_datasets_for_unfolding(input_dir, output_dir, sysvar, feature_mode):

    buffers = []

    # The embedding tree for a sysvar is OPTIONAL input — e.g. the PYTHIA8 prior
    # may not be produced yet. Skip gracefully (matching preprocess_embedding /
    # the embedding-is-optional convention) so a top-level `for sys_var in SysVar`
    # loop can run before every variation has been clustered/reweighted, and so
    # no script makes PYTHIA8 (or any unproduced variant) compulsory.
    _embed_dir = input_dir / "embedding" / str(sysvar)
    if not (_embed_dir / "reco-matches.arrow").exists():
        print(
            f"[make_datasets_for_unfolding] no embedding arrows at {_embed_dir}; "
            f"skipping {sysvar}. Produce them first (cluster_embedding.py + "
            f"preprocessing; prior variants also need omnisequential.py / "
            f"reverse_omnisequential.py)."
        )
        return None, None

    # Only LIKE_DATA stays a closure test (pseudo-data = reweighted reco vs
    # nominal sim) -> its residual non-closure becomes the "non-closure"
    # systematic. HERWIG7 / PYTHIA8 are now genuine model/prior-dependence
    # systematics: they unfold the REAL data using their own reweighted
    # embedding as the simulation/prior, so they take the normal `else` branch
    # below (data = data.arrow, sim = embedding/<sysvar>/) and produce an
    # alternate unfolded *data* spectrum, not a closure ratio.
    _PRIOR_VARIANTS = (
        SysVar.UNFOLDING_PRIOR_LIKE_DATA,  # TODO: Make sure paths are set properly for .arrow outputs of omnisequential.py
        # --- old: HERWIG7/PYTHIA8 also ran the pseudo-data closure path; they
        # are now model-dependence variations unfolding real data instead. ---
        # SysVar.UNFOLDING_PRIOR_HERWIG7,
        # SysVar.UNFOLDING_PRIOR_PYTHIA8,
    )

    if sysvar in _PRIOR_VARIANTS:
        # Prior-systematic closure variant (LIKE_DATA only): the "data" side is
        # reco-level pseudo-data (reweighted reco-matches + fakes), the "sim"
        # side stays nominal so the closure test is meaningful.
        # omnisequential.py has already baked the reweighted weights into
        # embedding/<sysvar>/{reco-matches,fakes}.arrow.

        embedding_input_path = input_dir / "embedding" / str(SysVar.NONE)
        buffers.append(
            pa.memory_map(
                str(input_dir / "embedding" / str(sysvar) / "reco-matches.arrow")
            )
        )
        data_match_table = pa.ipc.open_file(buffers[-1]).read_all()
        buffers.append(
            pa.memory_map(str(input_dir / "embedding" / str(sysvar) / "fakes.arrow"))
        )
        data_fake_table = pa.ipc.open_file(buffers[-1]).read_all()
        data_table = pa.concat_tables((data_match_table, data_fake_table))

        is_data_col = np.full_like(data_table["is_data"].to_numpy(), True)
        is_matched_col = np.full_like(data_table["is_matched"].to_numpy(), -1)
        data_table = replace_table_column(data_table, "is_data", is_data_col)
        data_table = replace_table_column(data_table, "is_matched", is_matched_col)
    else:
        embedding_input_path = input_dir / "embedding" / str(sysvar)
        buffers.append(pa.memory_map(str(input_dir / "data.arrow")))
        data_table = pa.ipc.open_file(buffers[-1]).read_all()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Tensordicts will be written to", output_dir)

    # The `angularities_*` subset modes feed the same scalar columns as `angularities`
    # minus a few observables: `angularities_noptd` drops the two p_T^D (k2_b0)
    # angularities; `angularities_minimal` additionally drops M / M_g / R_g. Other
    # modes are unaffected: bin_counts ignores `columns` (uses the bin block) and
    # combined keeps jet_columns. Extend this map to add another subset mode.
    # --- old ---
    # _input_columns = (
    #     jet_columns_noptd if feature_mode == "angularities_noptd" else jet_columns
    # )
    _INPUT_COLUMNS = {
        "angularities_noptd": jet_columns_noptd,
        "angularities_minimal": jet_columns_minimal,
    }
    _input_columns = _INPUT_COLUMNS.get(feature_mode, jet_columns)

    buffers.append(pa.memory_map(str(embedding_input_path / "reco-matches.arrow")))
    reco_match_table = pa.ipc.open_file(buffers[-1]).read_all()
    buffers.append(pa.memory_map(str(embedding_input_path / "fakes.arrow")))
    reco_fakes_table = pa.ipc.open_file(buffers[-1]).read_all()
    reco_table = pa.concat_tables((reco_match_table, reco_fakes_table))

    detlvl_td = to_tensordict(
        data_table,
        reco_table,
        columns=_input_columns,
        prefix=output_dir / "det_lvl",
        max_chunksize=100000,
        feature_mode=feature_mode,
    )

    buffers.append(pa.memory_map(str(embedding_input_path / "gen-matches.arrow")))
    gen_match_table = pa.ipc.open_file(buffers[-1]).read_all()
    buffers.append(pa.memory_map(str(embedding_input_path / "misses.arrow")))
    gen_misses_table = pa.ipc.open_file(buffers[-1]).read_all()
    gen_table = pa.concat_tables((gen_match_table, gen_misses_table))

    is_data_col = np.full_like(gen_table["is_data"].to_numpy(), True)
    is_matched_col = np.full_like(gen_table["is_matched"].to_numpy(), -1)
    gen_table_data_like = replace_table_column(gen_table, "is_data", is_data_col)
    gen_table_data_like = replace_table_column(
        gen_table_data_like, "is_matched", is_matched_col
    )

    partlvl_td = to_tensordict(
        gen_table_data_like,
        gen_table,
        columns=_input_columns,
        prefix=output_dir / "part_lvl",
        max_chunksize=100000,
        feature_mode=feature_mode,
    )

    return partlvl_td, detlvl_td


# def make_datasets_for_closure(source_dir, feature_mode="angularities"):
#    if feature_mode not in FEATURE_MODES:
#        raise ValueError(
#            f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}"
#        )
#
#    root_dir = source_dir / "embedding" / str(SysVar.NONE)
#    if not root_dir.exists():
#        print(
#            f"[make_datasets_for_closure] no embedding dir at {root_dir}; skipping. "
#            f"Run cluster_embedding.py + preprocessing for {SysVar.NONE} first."
#        )
#        return
#
#    buffers = []
#
#    buffers.append(pa.memory_map(str(root_dir / "reco-matches.arrow")))
#    reco_match_table = pa.ipc.open_file(buffers[-1]).read_all()
#    buffers.append(pa.memory_map(str(root_dir / "fakes.arrow")))
#    reco_fakes_table = pa.ipc.open_file(buffers[-1]).read_all()
#    reco_table = pa.concat_tables((reco_match_table, reco_fakes_table))
#
#    # Pseudo-data side: identical row content to sim, relabeled is_data=True / is_matched=-1.
#    # weight stays as the original per-pthat MC weight — NO omniseq overwrite. This is what
#    # makes this a true MC self-closure rather than a prior-systematic restriction.
#    data_table = pa.concat_tables((reco_match_table, reco_fakes_table))
#    data_table = replace_table_column(
#        data_table, "is_data", np.full_like(data_table["is_data"].to_numpy(), True)
#    )
#    data_table = replace_table_column(
#        data_table, "is_matched", np.full_like(data_table["is_matched"].to_numpy(), -1)
#    )
#
#    out_dir = source_dir / "closure" / "tensordicts" / feature_mode
#    out_dir.mkdir(parents=True, exist_ok=True)
#    print("Closure tensordicts will be written to", out_dir)
#
#    to_tensordict(
#        data_table,
#        reco_table,
#        columns=jet_columns,
#        prefix=out_dir / "det_lvl",
#        max_chunksize=100000,
#        feature_mode=feature_mode,
#    )
#
#    buffers.append(pa.memory_map(str(root_dir / "gen-matches.arrow")))
#    gen_match_table = pa.ipc.open_file(buffers[-1]).read_all()
#    buffers.append(pa.memory_map(str(root_dir / "misses.arrow")))
#    gen_misses_table = pa.ipc.open_file(buffers[-1]).read_all()
#    gen_table = pa.concat_tables((gen_match_table, gen_misses_table))
#
#    gen_table_data_like = replace_table_column(
#        gen_table, "is_data", np.full_like(gen_table["is_data"].to_numpy(), True)
#    )
#    gen_table_data_like = replace_table_column(
#        gen_table_data_like,
#        "is_matched",
#        np.full_like(gen_table["is_matched"].to_numpy(), -1),
#    )
#
#    to_tensordict(
#        gen_table_data_like,
#        gen_table,
#        columns=jet_columns,
#        prefix=out_dir / "part_lvl",
#        max_chunksize=100000,
#        feature_mode=feature_mode,
#    )


def main(
    root_dir,
    sysvar,
    feature_mode="angularities",
    redo_preprocessing=True,
    redo_datasets=False,
):
    if redo_preprocessing:
        print("Preprocessing data...")
        preprocess_data(
            root_dir / "jets",
            root_dir / "features" / feature_mode,
            "data.arrow",
            feature_mode,
            is_data=True,
            is_matched=-1,
            pth_bin=-1,
        )
        print("Preprocessing embedding...")
        preprocess_embedding(
            root_dir / "jets",
            root_dir / "features" / feature_mode,
            sysvar,
            feature_mode,
        )
    if redo_datasets:
        print("Making tensordict for ML datasets", str(sysvar), feature_mode, "...")
        _, _ = make_datasets_for_unfolding(
            root_dir / "features" / feature_mode,
            root_dir / "features" / feature_mode / "tensordicts" / str(sysvar),
            sysvar,
            feature_mode,
        )


if __name__ == "__main__":
    from config import load_config, config_path_from_argv

    # Optional positional config path (see config.config_path_from_argv) lets a driver
    # point this at a private config copy for concurrent runs; defaults to the shared
    # runtime-files/config.json otherwise.
    cfg = load_config(config_path_from_argv())
    root_dir = cfg.dataset_root
    sys_var = cfg.sys_var

    feature_mode = cfg.get("feature_mode", "angularities")
    redo_preprocessing = cfg.get("redo_preprocessing", False)
    redo_datasets = cfg.get("redo_datasets", True)
    redo_closure_datasets = cfg.get("redo_closure_datasets", False)
    if feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"cfg['feature_mode'] must be one of {FEATURE_MODES}, got {feature_mode!r}"
        )
    print(
        f"feature_mode={feature_mode} redo_preprocessing={redo_preprocessing} "
        f"redo_datasets={redo_datasets} redo_closure_datasets={redo_closure_datasets}"
    )

    if sys_var is None:
        for _sys_var in SysVar:
            main(
                root_dir,
                _sys_var,
                feature_mode=feature_mode,
                redo_preprocessing=redo_preprocessing,
                redo_datasets=redo_datasets,
            )
    else:
        main(
            root_dir,
            sys_var,
            feature_mode=feature_mode,
            redo_preprocessing=redo_preprocessing,
            redo_datasets=redo_datasets,
        )

    # if redo_closure_datasets:
    #    print("Making closure-test tensordicts ...")
    #    make_datasets_for_closure(source_dir, feature_mode=feature_mode)
