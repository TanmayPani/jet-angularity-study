import gc
from pathlib import Path
from multiprocessing import Array
from concurrent.futures import as_completed

import pyarrow as pa
import pyarrow.compute as pc
from matplotlib import pyplot as plt

from preprocessing import process_table, replace_table_column
from bounded_pool_executor import BoundedProcessPoolExecutor


def process_herwig_file(arrow_file_path, out_dir):
    processed_path = out_dir / f"processed_{arrow_file_path.name}"
    buffer = pa.memory_map(str(arrow_file_path), "rb")
    output = process_table(pa.ipc.open_file(buffer).read_all())
    with pa.OSFile(str(processed_path), "wb") as sink:
        with pa.ipc.new_file(sink, output.schema) as writer:
            writer.write_batch(output)

    del output
    del buffer
    json_path = arrow_file_path.with_suffix(".json")
    return processed_path, json_path if json_path.exists() else None


def preprocess_herwig(input_root_dir, pattern, **kwargs):
    for ifolder, folder in enumerate(Path(input_root_dir).glob(pattern)):
        with BoundedProcessPoolExecutor(max_workers=5) as exe:
            futures = {}
            for input_file in (folder / "out").glob("*.arrow"):
                futures[exe.submit(process_herwig_file, input_file, folder)] = (
                    input_file
                )

            for future in as_completed(futures):
                processed_path, xsec_data_path = future.result()
                del futures[future]
                print(
                    "Output file:", processed_path, ", XSec data path:", xsec_data_path
                )
                gc.collect()


if __name__ == "__main__":
    num_events_per_batch = 500000
    preprocess_herwig(
        "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs",
        f"HwJets_RHIC_ptHat*-*_nEv{num_events_per_batch}",
    )
