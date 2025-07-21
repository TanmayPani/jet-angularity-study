from typing import Optional, List
from collections.abc import Sequence
import numpy as np
from numpy import typing as npt

import torch 
import pyarrow as pa
from pyarrow import compute as pc 

def get_column_mean(data : pa.Table, col : str):
    if pa.types.is_list(data[col].type):
        return pc.mean(pc.list_flatten(data[col])).as_py()
    else:
        return pc.mean(data[col]).as_py()

def get_column_stddev(data : pa.Table, col : str):
    if pa.types.is_list(data[col].type):
        return pc.stddev(pc.list_flatten(data[col])).as_py()
    else:
        return pc.stddev(data[col]).as_py()

def get_mean_stddev_tensors(data: pa.Table, column_names: Optional[Sequence[str]]):
    column_names = column_names or data.column_names 
    mean_tensor = torch.as_tensor([get_column_mean(data, col) for col in column_names], dtype=torch.float32)
    stddev_tensor = torch.as_tensor([get_column_stddev(data, col) for col in column_names], dtype=torch.float32)
    #mean_list.append(pc.mean(data[col]).as_py())
    #stddev_list.append(pc.stddev(data[col]).as_py())
    return mean_tensor, stddev_tensor

def add_constit_slice_column(jet_table, consit_col_name, new_col_name, start, stop=None):
    if stop is None:
        stop = start+1
    carr = pc.list_slice(jet_table[consit_col_name], start, stop=stop).combine_chunks().flatten()
    return jet_table.append_column(new_col_name, carr)

def add_leading_constit_column(jet_table): 
    constit_pt_arr = jet_table["constit_pt"].combine_chunks().values
    constit_jet_indices = jet_table["constit_pt"].combine_chunks().value_parent_indices()
    constit_table = pa.table({"constit_pt":constit_pt_arr, "jet_index":constit_jet_indices})
    agg = constit_table.group_by("jet_index").aggregate([("constit_pt", "max")])

    return jet_table.append_column("leading_constit_pt", agg["constit_pt_max"])

def pa_table(source : str , label : Optional[npt.ArrayLike] = None, label_key:str="label"):
    buffer = pa.memory_map(source, "rb")
    table = pa.ipc.open_file(buffer).read_all()
    _len = len(table)
    label_arr = None
    if isinstance(label, (int, np.number)):
        label_arr = np.full(_len, label, dtype=np.int_)
    elif isinstance(label, np.ndarray):
        assert len(label) == _len
        label_arr = np.asarray(label, dtype=np.int_)
    else: 
        label_arr = np.empty(0)

    table = add_extra_columns(table.append_column(label_key, pa.array(label_arr, type=pa.int32())))

    return buffer, table

def pa_concated_table(source : List[str], label : Optional[List[npt.ArrayLike]] = None, label_key:str="label"):
    n_tables = len(source)
    label_iter = [None]*n_tables
    
    if label:
        assert len(label) == n_tables
        label_iter = label

    buffer_list = []
    table_list = []
    for _source, _label in zip(source, label_iter):
        if not isinstance(_source, str):
            raise TypeError("Can't use sources other than path strings for pa.Table!")
        buffer, table = pa_table(_source, label=_label, label_key=label_key)
        buffer_list.append(buffer)
        table_list.append(table)

    return buffer_list, pa.concat_tables(table_list)

def add_extra_columns(table : pa.Table) -> pa.Table:
    table = add_constit_slice_column(table, "constit_pt", "leading_constit_pt", 0)
    table = add_constit_slice_column(table, "constit_eta", "leading_constit_eta", 0)
    table = add_constit_slice_column(table, "constit_phi", "leading_constit_phi", 0)

    table = add_constit_slice_column(table, "constit_pt", "subleading_constit_pt", 1)
    table = add_constit_slice_column(table, "constit_eta", "subleading_constit_eta", 1)
    table = add_constit_slice_column(table, "constit_phi", "subleading_constit_phi", 1)

    return table

