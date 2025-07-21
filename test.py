import numpy as np
from torchmodel.datasets import Batch, collate_fn, ArrowDataset, SubsetSequentialSampler
import torch 
from torch.utils.data import DataLoader, SequentialSampler

from _unfolding_utils import pa_table, pa_concated_table, get_mean_stddev_tensors

jet_columns = [
        "pt", "eta", "phi", "nef",
        "ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0",
        "leading_constit_pt", "leading_constit_eta", "leading_constit_phi",
        "subleading_constit_pt", "subleading_constit_eta", "subleading_constit_phi",
        "hc_pt", "hc_eta", "hc_phi",
        "hc_ch_ang_k1_b0.5", "hc_ch_ang_k1_b1", "hc_ch_ang_k1_b2", "hc_ch_ang_k2_b0",
        ]

buffer, table = pa_table("outputs/jets_conPtMin0.2_wTrackPtSys.arrow", label=0, label_key="stratify_labels")

l = len(table)
indices = list(range(10))
#table = table.take(indices)

dataset = ArrowDataset(table, np.ones(l), sample_weights=np.ones(l), column_names=jet_columns)

sampler = SubsetSequentialSampler(indices)

btsize = 8

batch = Batch(batch_size=btsize)

dl = DataLoader(dataset, batch_size=8, collate_fn=batch.__call__, pin_memory=True, sampler=sampler)

for bt in dl:
    print(bt.unpack_data()[0])

dataset.sample_weights = np.zeros(l)

for bt in dl:
    print(bt.unpack_data()[0])
