import gc
import json
import re
from pathlib import Path
from concurrent.futures import as_completed

import numpy as np
import pyarrow as pa

from preprocessing import (
    jet_columns,
    process_table,
    pth_bins as _PTH_BINS,
    replace_table_column,
)
from bounded_pool_executor import BoundedProcessPoolExecutor


_PTH_FOLDER_RE = re.compile(r"ptHat(\d+)-(\d+|infty|150)")
_PTH_LOWER_TO_IDX = {edge: i for i, edge in enumerate(_PTH_BINS[:-1])}


def resolve_pth_bin(folder_name: str) -> int:
    m = _PTH_FOLDER_RE.search(folder_name)
    if m is None:
        raise ValueError(f"can't extract pt-hat range from folder: {folder_name}")
    lo = m.group(1)
    if lo not in _PTH_LOWER_TO_IDX:
        raise ValueError(
            f"folder {folder_name} has pt-hat low edge {lo!r}, "
            f"not in canonical pth_bins {_PTH_BINS}"
        )
    return _PTH_LOWER_TO_IDX[lo]


def _iter_subbatches(batch: pa.RecordBatch, max_chunksize):
    if max_chunksize is None or batch.num_rows <= max_chunksize:
        yield batch
        return
    for off in range(0, batch.num_rows, max_chunksize):
        yield batch.slice(off, min(max_chunksize, batch.num_rows - off))


def process_herwig_file(
    arrow_file_path: Path,
    pth_bin: int,
    out_dir: Path,
    *,
    max_chunksize: int = 50_000,
):
    processed_path = out_dir / f"processed_{arrow_file_path.name}"
    json_path = arrow_file_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"missing xsec sidecar: {json_path}")
    with open(json_path) as f:
        xsec_meta = json.load(f)
    hist_scale = float(xsec_meta["histogramScale"])
    # `m` and `sd_m` are not in preprocessing.jet_columns but are required by
    # reverse_omnisequential.py's local jet_columns list — keep them too.
    # keep_cols = list(jet_columns) + ["m", "sd_m", "pth_bin", "weight"]

    buffer = pa.memory_map(str(arrow_file_path), "rb")
    reader = pa.ipc.open_file(buffer)

    sink = pa.OSFile(str(processed_path), "wb")
    writer = None
    try:
        for ib in range(reader.num_record_batches):
            raw_batch = reader.get_batch(ib)
            for sub in _iter_subbatches(raw_batch, max_chunksize):
                scaled_w = (sub["weight"].to_numpy() * hist_scale).astype(np.float32)
                in_tab = replace_table_column(
                    pa.Table.from_batches([sub]),
                    "weight",
                    scaled_w,
                )
                out_rb = process_table(in_tab, pth_bin=pth_bin)
                # out_rb = out_rb.select(keep_cols)
                if writer is None:
                    writer = pa.ipc.new_file(sink, out_rb.schema)
                writer.write_batch(out_rb)
                del in_tab, out_rb, scaled_w, sub
                gc.collect()
            del raw_batch
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
        sink.close()
        del reader, buffer
    return processed_path, hist_scale, xsec_meta


def preprocess_herwig(
    input_root_dir,
    pattern,
    *,
    max_workers: int = 2,
    max_chunksize: int = 50_000,
):
    results = []
    for folder in sorted(Path(input_root_dir).glob(pattern)):
        pth_bin = resolve_pth_bin(folder.name)
        in_files = sorted((folder / "out").glob("*.arrow"))
        if not in_files:
            print(f"[skip] no raw arrows under {folder}/out")
            continue
        print(f"[{folder.name}] pth_bin={pth_bin}  n_raw={len(in_files)}")
        with BoundedProcessPoolExecutor(
            max_workers=max_workers,
            max_tasks_per_child=1,
        ) as exe:
            futures = {
                exe.submit(
                    process_herwig_file,
                    in_file,
                    pth_bin,
                    folder,
                    max_chunksize=max_chunksize,
                ): in_file
                for in_file in in_files
            }
            for future in as_completed(futures):
                processed_path, hist_scale, xsec_meta = future.result()
                print(
                    f"  {processed_path.name}  "
                    f"xsec={xsec_meta['crossSection']:.3g}  scale={hist_scale:.4g}"
                )
                results.append((processed_path, pth_bin, xsec_meta))
                del futures[future]
                gc.collect()
    return results


def combine_processed_arrows(processed_entries, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    sink = pa.OSFile(str(output_path), "wb")
    total_rows, total_w, per_bin_rows = 0, 0.0, {}
    try:
        for processed_path, pth_bin, _meta in processed_entries:
            buf = pa.memory_map(str(processed_path), "rb")
            t = pa.ipc.open_file(buf).read_all()
            for batch in t.to_batches():
                if writer is None:
                    writer = pa.ipc.new_file(sink, batch.schema)
                writer.write_batch(batch)
                total_rows += batch.num_rows
                total_w += float(batch["weight"].to_numpy().sum())
                per_bin_rows[pth_bin] = per_bin_rows.get(pth_bin, 0) + batch.num_rows
            del t, buf
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
        sink.close()
    print(f"[combine] wrote {output_path}")
    print(f"  total_rows={total_rows}  sum(weight)={total_w:.4g}")
    print(
        "  per-pth_bin rows: "
        + ", ".join(f"{b}:{n}" for b, n in sorted(per_bin_rows.items()))
    )


if __name__ == "__main__":
    redo_processing = True
    input_root = (
        "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs"
    )
    pattern = "HwJets_RHIC_ptHat*-*_nEv500000"
    output_path = Path(
        "./datasets/STAR_pp200GeV_production_2012/clustered_jets/alt_gen/herwig7.arrow"
    )

    if redo_processing:
        entries = preprocess_herwig(input_root, pattern)
    else:
        entries = []
        for folder in sorted(Path(input_root).glob(pattern)):
            pth_bin = resolve_pth_bin(folder.name)
            for processed in sorted(folder.glob("processed_*.arrow")):
                entries.append((processed, pth_bin, None))
    combine_processed_arrows(entries, output_path)
