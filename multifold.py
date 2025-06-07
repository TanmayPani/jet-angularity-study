import os 
from functools import singledispatch
from collections.abc import Sequence, Sized, Iterable
from datetime import datetime
from typing import Optional, Union, List, Any, Generic
import pickle

import numpy as np
from numpy import typing as npt
from sklearn.model_selection import train_test_split

import pyarrow as pa
from pyarrow import compute as pc 
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy

from torchmodel import archs
from torchmodel.torchmodel import TensorDictModel
from torchmodel.callbacks import EarlyStopping
from torchmodel.datasets import JetDataset, batch_collate

def get_model(dnn_layers, in_keys, out_key, device=torch.device("cpu"), name="", do_debug=False, learning_rate=0.01, sched_patience=5):
    classifier = TensorDictModel(
        archs.DNN(dnn_layers, dropout_prob=0.2, do_batch_norm=True), in_keys, out_key, device=device, name=name, do_debug=do_debug
    )
    optimizer = torch.optim.Adam(classifier().parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=sched_patience
    )
    classifier.compile(
        criterion=torch.nn.BCEWithLogitsLoss(reduction="none"),
        optimizer=optimizer,
        compile_mode="default",
        scheduler=lr_scheduler,
    )

    return classifier

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

def pa_table(source : str , label : Optional[npt.ArrayLike] = None):
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

    return buffer, add_extra_columns(table), label_arr

def pa_concated_table(source : Sequence[str], label : Optional[Sequence[npt.ArrayLike]] = None):
    n_tables = len(source)
    label_iter = label if label is not None else [None]*n_tables
    assert len(label) == n_tables
    buffer_list = []
    table_list = []
    label_list = []
    for _source, _label in zip(source, label_iter):
        if not isinstance(_source, str):
            raise TypeError("Can't use sources other than path strings for pa.Table!")
        buffer, table, label_arr = pa_table(_source, label=_label)
        buffer_list.append(buffer)
        table_list.append(table)
        label_list.append(label_arr)

    return buffer_list, pa.concat_tables(table_list), np.concatenate(label_list)

def add_extra_columns(table : pa.Table) -> pa.Table:
    table = add_constit_slice_column(table, "constit_pt", "leading_constit_pt", 0)
    table = add_constit_slice_column(table, "constit_eta", "leading_constit_eta", 0)
    table = add_constit_slice_column(table, "constit_phi", "leading_constit_phi", 0)

    table = add_constit_slice_column(table, "constit_pt", "subleading_constit_pt", 1)
    table = add_constit_slice_column(table, "constit_eta", "subleading_constit_eta", 1)
    table = add_constit_slice_column(table, "constit_phi", "subleading_constit_phi", 1)

    return table

def get_train_val_dataloaders(table, labels, batch_size, validation_size=0.2, test_size=None, do_scale=True, columns=None, stratify=None, num_dl_workers=0, persistent_dl_workers=False):
    if stratify is None:
        stratify = labels
    elif len(stratify) == 0:
        stratify = labels

    all_indices = np.arange(len(table))

    test_indices = None 
    n_test = 0
    if test_size is not None:
        all_indices, test_indices = train_test_split(all_indices, test_size=test_size, stratify=stratify, shuffle=True)
        stratify = stratify[all_indices]
        n_test = len(test_indices)

    train_indices, val_indices = train_test_split(all_indices, test_size=validation_size, stratify=stratify, shuffle=True)
    print(f"---Got {len(train_indices)} for training, {len(val_indices)} for validation, and {n_test} for testing...")

    if do_scale:
        train_table = table.take(train_indices)
        dataset = JetDataset(table, labels, column_names=columns, scale_from=train_table)
    else:
        dataset = JetDataset(table, labels, column_names=columns)

    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers, persistent_workers=persistent_dl_workers)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers, persistent_workers=persistent_dl_workers)

    test_loader = None 
    if test_indices is not None:
        test_sampler = SequentialSampler(test_indices)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers, persistent_workers=persistent_dl_workers)
    
    print(f"---Expecting {len(train_indices)//batch_size} training batches, {len(val_indices)//batch_size} validation batches, {n_test//batch_size} testing batches")
    return dataset, train_loader, val_loader, test_loader

def main(device: torch.device = torch.device("cuda")):
    jet_columns = [
        "pt",
        "eta",
        "phi",
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
        "hc_pt",
        "hc_eta",
        "hc_phi",
        "hc_ch_ang_k1_b0.5",
        "hc_ch_ang_k1_b1",
        "hc_ch_ang_k1_b2",
        "hc_ch_ang_k2_b0",
        ]

    emb_data_folder = "outputs/30May25-1147"
    pth_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
    pth_bin_folders = [f"{emb_data_folder}/ptHat{pth_low}to{pth_high}" for pth_low, pth_high in zip(pth_bins[:-1], pth_bins[1:])]
    n_pth_bins = len(pth_bin_folders)
    pth_label = list(range(1, n_pth_bins+1))
    print(len(pth_label), n_pth_bins)

    gen_match_buffers, gen_match_table, gen_match_stratify_label = pa_concated_table([f"{folder}/gen-matches.arrow" for folder in pth_bin_folders], label=pth_label)
    gen_miss_buffers, gen_miss_table, gen_miss_stratify_label = pa_concated_table([f"{folder}/misses.arrow" for folder in pth_bin_folders], label=pth_label) 
    reco_match_buffers, reco_match_table, reco_match_stratify_label = pa_concated_table([f"{folder}/reco-matches.arrow" for folder in pth_bin_folders], label=pth_label)
    reco_fake_buffers, reco_fake_table, reco_fake_stratify_label = pa_concated_table([f"{folder}/fakes.arrow" for folder in pth_bin_folders], label=pth_label)

    data_buffer, data_table, data_stratify_label = pa_table("outputs/jets-conPtMin0.2.arrow", label=0)

    n_data = len(data_table)

    n_gen_matches = len(gen_match_table)
    n_gen_misses = len(gen_miss_table)
    n_reco_matches = len(reco_match_table)
    n_reco_fakes = len(reco_fake_table)

    assert n_gen_matches == n_reco_matches

    n_matches = n_gen_matches 
    n_gen = n_matches + n_gen_misses
    n_reco = n_matches + n_reco_fakes

    print("Number of matched gen jets, matched reco jets:", n_gen_matches, n_reco_matches, n_matches)
    print("Number of missed gen jets, fake reco jets:", n_gen_misses, n_reco_fakes)
    print("Number of data jets:", n_data)
    
    in_keys = [("features")]
    out_key = "targets"

    detlvl_model_layers = [len(jet_columns), 256, 256, 256, 1]
    fake_scaler_model_layers = [len(jet_columns), 256, 256, 256, 1]
    genlvl_model_layers = [len(jet_columns), 256, 256, 256, 1]
    miss_scaler_model_layers = [len(jet_columns), 256, 256, 256, 1]
    
    val_size = 0.4

    detlvl_batch_size = 10000
    genlvl_batch_size = 10000
    num_dl_workers = 10 

    do_debug = False

    use_missed_jets_for_push = False

    #################################################################################################################################################################################
    #################################################################################################################################################################################

    detlvl_table = pa.concat_tables([data_table, reco_match_table, reco_fake_table])
    detlvl_labels = np.concatenate([np.ones(n_data), np.zeros(n_reco_matches + n_reco_fakes)])
    detlvl_stratify_labels = np.concatenate([data_stratify_label, reco_match_stratify_label, reco_fake_stratify_label])
    print("Splitting detlvl...")
    detlvl_ds, train_detlvl_loader, val_detlvl_loader, _ = get_train_val_dataloaders(detlvl_table, detlvl_labels, detlvl_batch_size, test_size=None, validation_size=val_size, stratify=detlvl_stratify_labels, columns=jet_columns, num_dl_workers=num_dl_workers)

    reco_match_ds = JetDataset(reco_match_table, np.zeros(n_reco_matches), column_names=jet_columns, scale_from=detlvl_ds)
    reco_match_sampler = SequentialSampler(reco_match_ds)
    reco_match_loader = DataLoader(reco_match_ds, batch_size=detlvl_batch_size, sampler=reco_match_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers)

    det_lvl_model = get_model(detlvl_model_layers, in_keys, out_key, device=device, name="detector_level", do_debug=do_debug)

    #################################################################################################################################################################################

    fake_scaler_table = pa.concat_tables([reco_match_table, reco_match_table])
    fake_scaler_labels = np.concatenate([np.ones(n_reco_matches), np.zeros(n_reco_matches)])
    fake_scaler_stratify_labels = np.concatenate([reco_match_stratify_label, -reco_match_stratify_label])
    print("Splitting fake scaler lvl...")
    fake_scaler_ds, train_fake_scaler_loader, val_fake_scaler_loader, _ = get_train_val_dataloaders(fake_scaler_table, fake_scaler_labels, detlvl_batch_size, test_size=None, validation_size=val_size, do_scale=True, stratify=fake_scaler_stratify_labels, columns=jet_columns, num_dl_workers=num_dl_workers)
    
    reco_fake_ds = JetDataset(reco_fake_table, np.zeros(n_reco_fakes), column_names=jet_columns, scale_from=fake_scaler_ds)
    reco_fake_sampler = SequentialSampler(reco_fake_ds)
    reco_fake_loader = DataLoader(reco_fake_ds, batch_size=detlvl_batch_size, sampler=reco_fake_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers)

    fake_scaler_model = get_model(fake_scaler_model_layers, in_keys, out_key, device=device, name="fake_scaler", do_debug=do_debug)

    #################################################################################################################################################################################
    #################################################################################################################################################################################
    
    if use_missed_jets_for_push:
        genlvl_table = pa.concat_tables([gen_match_table, gen_miss_table, gen_match_table, gen_miss_table])
        genlvl_labels = np.concatenate([np.ones(n_gen_matches + n_gen_misses), np.zeros(n_gen_matches + n_gen_misses)])
        genlvl_stratify_labels = np.concatenate([gen_match_stratify_label, gen_miss_stratify_label, -gen_match_stratify_label, -gen_miss_stratify_label])
    else:
        genlvl_table = pa.concat_tables([gen_match_table, gen_match_table])
        genlvl_labels = np.concatenate([np.ones(n_gen_matches), np.zeros(n_gen_matches)])
        genlvl_stratify_labels = np.concatenate([gen_match_stratify_label, -gen_match_stratify_label])
    
    print("Splitting genlvl...")
    genlvl_ds, train_genlvl_loader, val_genlvl_loader, _ = get_train_val_dataloaders(genlvl_table, genlvl_labels, genlvl_batch_size, test_size=None, validation_size=val_size, do_scale=True, stratify=genlvl_stratify_labels, columns=jet_columns, num_dl_workers=num_dl_workers)

    gen_match_ds = JetDataset(gen_match_table, np.zeros(n_gen_matches), column_names=jet_columns, scale_from=genlvl_ds)
    gen_match_sampler = SequentialSampler(gen_match_ds)
    gen_match_loader = DataLoader(gen_match_ds, batch_size=genlvl_batch_size, sampler=gen_match_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers)

    gen_lvl_model = get_model(genlvl_model_layers, in_keys, out_key, device=device, name="particle_level", do_debug=do_debug)

    #################################################################################################################################################################################
 
    miss_scaler_table = pa.concat_tables([gen_match_table, gen_match_table])
    miss_scaler_labels = np.concatenate([np.ones(n_gen_matches), np.zeros(n_gen_matches)])
    miss_scaler_stratify_labels = np.concatenate([gen_match_stratify_label, -gen_match_stratify_label])
    print("Splitting miss scaler lvl...")
    miss_scaler_ds, train_miss_scaler_loader, val_miss_scaler_loader, _ = get_train_val_dataloaders(miss_scaler_table, miss_scaler_labels, genlvl_batch_size, test_size=None, validation_size=val_size, stratify=miss_scaler_stratify_labels, do_scale=True, columns=jet_columns, num_dl_workers=num_dl_workers)
    
    gen_miss_ds = JetDataset(gen_miss_table, np.zeros(n_gen_misses), column_names=jet_columns, scale_from=miss_scaler_ds)
    gen_miss_sampler = SequentialSampler(gen_miss_ds)
    gen_miss_loader = DataLoader(gen_miss_ds, batch_size=genlvl_batch_size, sampler=gen_miss_sampler, pin_memory=True, collate_fn=batch_collate, num_workers=num_dl_workers)
    
    miss_scaler_model = get_model(miss_scaler_model_layers, in_keys, out_key, device=device, name="miss_scaler", do_debug=do_debug)
    
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    
    n_iterations = 5 
    num_epochs = 50
    patience = 10

    #output_folder=f"{emb_data_folder}/multifolding-{datetime.now().strftime("%d-%b-%y-%H-%M")}"
    output_folder=f"{emb_data_folder}/multifolding_1"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    checkpoint_folder=f"{output_folder}/checkpoints"
    
    epsilon = 1e-20

    metrics = MetricCollection({ "f1_score" : BinaryF1Score(), "precision" : BinaryPrecision(), "recall" : BinaryRecall(), "accuracy" : BinaryAccuracy()})

    gen_match_weights = gen_match_table["weight"].to_numpy()
    reco_match_weights = reco_match_table["weight"].to_numpy()
    assert np.allclose(gen_match_weights, reco_match_weights)
    
    match_weights = gen_match_weights

    gen_miss_weights = gen_miss_table["weight"].to_numpy()
    reco_fake_weights = reco_fake_table["weight"].to_numpy()
   

    data_weights = np.ones(len(data_table))
    
    gen_weights = np.concatenate([match_weights, gen_miss_weights]) if use_missed_jets_for_push else match_weights
    gen_wts_scale_factor = np.sum(data_weights)/np.sum(gen_weights)
    
    reco_weights = np.concatenate([match_weights, reco_fake_weights])
    reco_wts_scale_factor = np.sum(data_weights)/np.sum(reco_weights)

    w_unfolding = [gen_weights*gen_wts_scale_factor, reco_weights*reco_wts_scale_factor]

    for iteration in range(n_iterations):
        print("###############################################################################################")
        print(f"Iteration: {iteration+1}/{n_iterations}")
        print("###############################################################################################")

        print("---Setting reco weights...")
        detlvl_weights = np.concatenate([data_weights, w_unfolding[-1]])

        print("---Calculating pull weights for gen matches...")
        det_lvl_earlystop = EarlyStopping(patience=patience, checkpointFolder=f"{checkpoint_folder}/det_lvl", checkPointFileName=f"model-iter{iteration}-epoch[EPOCH].pth")
        det_lvl_history = det_lvl_model.fit(train_detlvl_loader, val_detlvl_loader, epochs=num_epochs, sample_weights=detlvl_weights, early_stopping=det_lvl_earlystop, metrics=metrics)
        predictions = det_lvl_model.predict(reco_match_loader, out_activation=torch.nn.Sigmoid()).cpu().detach().numpy()
        gen_match_reweight = predictions / (1. - predictions + epsilon)

        match_weights = match_weights*gen_match_reweight
        
        with open(f"{output_folder}/detlvl-train-iter{iteration+1}.pkl", "wb") as f:
            pickle.dump(det_lvl_history, f)
        
        #################################################################################################################################################################################
        if use_missed_jets_for_push:
            print("---Calculating pull weights for gen misses...")
            gen_miss_scaler_weights = np.concatenate([gen_match_reweight, np.ones(n_gen_matches)])
            miss_scaler_ds.set_sample_weights(gen_miss_scaler_weights)

            miss_scaler_earlystop = EarlyStopping(patience=patience, checkpointFolder=f"{checkpoint_folder}/miss_scaler", checkPointFileName=f"model-iter{iteration}-epoch[EPOCH].pth")
            miss_scaler_history = miss_scaler_model.fit(train_miss_scaler_loader, val_miss_scaler_loader, epochs=num_epochs, sample_weights=gen_miss_scaler_weights, early_stopping=miss_scaler_earlystop, metrics=metrics)
            predictions = miss_scaler_model.predict(gen_miss_loader, out_activation=torch.nn.Sigmoid()).cpu().detach().numpy()
            gen_miss_reweight = predictions / (1. - predictions + epsilon)

            gen_miss_weights = gen_miss_weights*gen_miss_reweight

            gen_weights = np.concatenate([match_weights, gen_miss_weights])

            with open(f"{output_folder}/miss-scaler-train-iter{iteration+1}.pkl", "wb") as f:
                pickle.dump(miss_scaler_history, f)
        else:
            gen_weights = match_weights
        #################################################################################################################################################################################        
        gen_wts_scale_factor = np.sum(data_weights)/np.sum(gen_weights)
        w_unfolding.append(gen_weights*gen_wts_scale_factor)
        #################################################################################################################################################################################
        
        print("---Setting gen weights...")
        genlvl_weights = np.concatenate([w_unfolding[-1], w_unfolding[-3]])
        genlvl_ds.set_sample_weights(genlvl_weights)
        
        print("---Calculating push weights for reco matches...")
        gen_lvl_earlystop = EarlyStopping(patience=patience, checkpointFolder=f"{checkpoint_folder}/gen_lvl", checkPointFileName=f"model-iter{iteration}-epoch[EPOCH].pth")
        gen_lvl_history = gen_lvl_model.fit(train_genlvl_loader, val_genlvl_loader, epochs=num_epochs, sample_weights=genlvl_weights, early_stopping=gen_lvl_earlystop, metrics=metrics)
        predictions = gen_lvl_model.predict(gen_match_loader, out_activation=torch.nn.Sigmoid()).cpu().detach().numpy()
        reco_match_reweight = predictions / (1. - predictions + epsilon)

        match_weights = match_weights*reco_match_reweight
        with open(f"{output_folder}/genlvl-train-iter{iteration+1}.pkl", "wb") as f:
            pickle.dump(gen_lvl_history, f)
 
        #################################################################################################################################################################################
        
        print("---Calculating push weights for reco fakes...")
        reco_fake_scaler_weights = np.concatenate([reco_match_reweight, np.ones(n_reco_matches)])
        fake_scaler_ds.set_sample_weights(reco_fake_scaler_weights)

        fake_scaler_earlystop = EarlyStopping(patience=patience, checkpointFolder=f"{checkpoint_folder}/fake_scaler", checkPointFileName=f"model-iter{iteration}-epoch[EPOCH].pth")
        fake_scaler_history = fake_scaler_model.fit(train_fake_scaler_loader, val_fake_scaler_loader, epochs=num_epochs, sample_weights=reco_fake_scaler_weights, early_stopping=fake_scaler_earlystop, metrics=metrics)
        predictions = fake_scaler_model.predict(reco_fake_loader, out_activation=torch.nn.Sigmoid()).cpu().detach().numpy()
        reco_fake_reweight = predictions / (1. - predictions + epsilon)

        reco_fake_weights = reco_fake_weights*reco_fake_reweight
        with open(f"{output_folder}/fake-scaler-train-iter{iteration+1}.pkl", "wb") as f:
            pickle.dump(fake_scaler_history, f)

        #################################################################################################################################################################################
        reco_weights = np.concatenate([match_weights, reco_fake_weights])
        reco_wts_scale_factor = np.sum(data_weights)/np.sum(reco_weights)
        w_unfolding.append(reco_weights*reco_wts_scale_factor)
        #################################################################################################################################################################################
       
        with open(f"{output_folder}/multifolded-wts-iter{iteration+1}.npz", "wb") as f:
            np.savez(f, *w_unfolding)

    print("Done!")
if __name__ == "__main__":
    #os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    main()
