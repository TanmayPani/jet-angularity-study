import os
from copy import deepcopy
from typing import Optional
from functools import partial

import numpy as np
from numpy import float64, typing as npt
from sklearn.model_selection import train_test_split

import pyarrow as pa
import pyarrow.compute as pc

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, SequentialSampler
from torch.nn.functional import binary_cross_entropy_with_logits

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy

from torchmodel.archs import DNN
from torchmodel.transforms import Normalize
from torchmodel.torchmodel import NeuralNet
from torchmodel.callbacks import EarlyStopping
from torchmodel.callbacks import LRScheduler
from torchmodel.callbacks import Checkpoint 
from torchmodel.callbacks import ProgressBar 
from torchmodel.datasets import ArrowDataset, TorchDataset, ValidSplit, Batch, collate_fn
from torchmodel.utils import set_global_random_seed

from _unfolding_utils import pa_table, pa_concated_table, get_mean_stddev_tensors

INTMAX = 4294967295

def undersample_single(X, max_size = -1, stratify = None, random_state=None):
    indices = np.arange(len(X))
    
    if max_size < 0 or max_size >= len(X):
        return indices

    train_indices, _ = train_test_split(indices, train_size=max_size, stratify=stratify, random_state=random_state)
    #print(train_indices[0:5])
    return train_indices

def undersample(X_pos : pa.Table, X_neg : pa.Table, max_class_size : int = -1, stratify_pos = None, stratify_neg = None, random_state=None):
    pos_size = len(X_pos)
    neg_size = len(X_neg)

    max_class_size = min(pos_size, neg_size) if max_class_size < 0 else min(pos_size, neg_size, max_class_size)

    pos_train_indices = undersample_single(X_pos, max_size=max_class_size, stratify=stratify_pos, random_state=random_state)
    neg_train_indices = undersample_single(X_neg, max_size=max_class_size, stratify=stratify_neg, random_state=random_state)

    return X_pos.take(pos_train_indices), X_neg.take(neg_train_indices), pos_train_indices, neg_train_indices

#def get_input_normalizer(X : pa.Table, column_names=None, as_tensor_dtype=torch.float32, stddev_ddof=1):
#    print("Creating input transform...")
#    column_names = column_names or X.column_names
#    mean = torch.as_tensor([pc.mean(X[col]).as_py() for col in column_names], dtype=as_tensor_dtype)
#    stddev = torch.as_tensor([pc.stddev(X[col], ddof=stddev_ddof).as_py() for col in column_names], dtype=as_tensor_dtype)
#
#    print(mean, stddev)
#    return Normalize(mean, stddev)

def get_input_normalizer(X : Dataset, ncols, stddev_ddof=1, dtype=torch.float32, **kwargs):
    sum = torch.zeros(ncols, dtype=dtype)
    sum_sq = torch.zeros(ncols, dtype=dtype)
    n_samples = len(X)
    loader = DataLoader(X, **kwargs)
    for batch in loader:
        x, _, _, _ = batch.unpack_data()
        sum += torch.sum(x, dim=0)
        sum_sq += torch.sum(x.pow(2), dim=0)

    mean = sum.div_(n_samples)
    sum_sq.div_(n_samples).sub_(mean.pow(2))
    stddev = sum_sq.mul_(n_samples/float(n_samples-stddev_ddof)).sqrt_()

    #print(mean, stddev)
    return Normalize(mean, stddev)


def get_reweighting(model : NeuralNet, X : Optional[pa.Table]=None, epsilon:float=1e-20) -> npt.NDArray[np.float64]:
    if X is None:
        return np.asarray([])
    pred = model.predict(X).squeeze()
    reweights = pred / (1. - pred + epsilon)
    return reweights

def get_split_datasets(X, y=None, sample_weights=None, valid_size=0.2, stratify = None, random_state=None, **kwargs):
    indices = np.arange(len(X))

    #print(train_indices[0:5])

    dataset = None
    if isinstance(X, Dataset):
        dataset = X
    elif isinstance(X , pa.Table):
        dataset = ArrowDataset(X, targets=y, sample_weights=sample_weights, **kwargs)
    else:
        raise TypeError(f"Can't convert input of type {type(X)} into Dataset!")

    if isinstance(dataset, ArrowDataset):
        stratify = dataset.stratification

    train_indices, valid_indices = train_test_split(indices, test_size=valid_size, stratify=stratify, random_state=random_state)

    return Subset(dataset, train_indices), Subset(dataset, valid_indices), dataset

def _make_split(X, y=None, valid_ds=None, **kwargs):
    """Used by ``predefined_split`` to allow for pickling"""
    return X, valid_ds

def predefined_split(dataset):
    return partial(_make_split, valid_ds=dataset)


def multifold(n_iterations,
              data_table : pa.Table, 
              reco_match_table : pa.Table, 
              gen_match_table : pa.Table, 
              reco_fake_table : Optional[pa.Table] = None, 
              gen_miss_table : Optional[pa.Table] = None, 
              output_folder:str="outputs", 
              init_kwargs : dict = {}, 
              train_class_size=-1,
              undersample_rng=None,
              model_rng=None,
              column_names : Optional[list[str]] = None,
              ):

    column_names = column_names or data_table.column_names
    n_features = len(column_names)
    #column_names = init_kwargs["dataset__column_names"] or data_table.column_names
    #print(n_features)
    #pull_layer_sizes = [n_features, 256, 256, 256, 1]
    #push_layer_sizes = [n_features, 256, 256, 256, 1]
    #push_fake_layer_sizes = [n_features, 256, 256, 256, 1]
    #pull_miss_layer_sizes = [n_features, 256, 256, 256, 1]

    batch_size = init_kwargs["batch_size"] or 10000
    #detlvl_batch_size = init_kwargs["batch_size"] or 10000
    #genlvl_batch_size = 2*detlvl_batch_size

    sample_weights = {}

    gen_table = pa.concat_tables([gen_match_table, gen_miss_table]) if gen_miss_table is not None else gen_match_table
    reco_table = pa.concat_tables([reco_match_table, reco_fake_table]) if reco_fake_table is not None else reco_match_table

    gen_match_weights = gen_match_table["weight"].to_numpy()
    reco_match_weights = reco_match_table["weight"].to_numpy()
    assert np.allclose(gen_match_weights, reco_match_weights)
    
    sample_weights["match"] = gen_match_weights
    sample_weights["gen_miss"] = gen_miss_table["weight"].to_numpy() if gen_miss_table is not None else np.asarray([])
    sample_weights["reco_fake"] = reco_fake_table["weight"].to_numpy() if reco_fake_table is not None else np.asarray([])

    data_weights = data_table["weight"].to_numpy()# if "omniseq_weight" in data_table.column_names else np.ones(len(data_table))
    sample_weights["data"] = data_weights/np.mean(data_weights) 
    
    gen_weights = np.concatenate([sample_weights["match"], sample_weights["gen_miss"]])
    sample_weights["gen"] = gen_weights*(np.sum(sample_weights["data"])/np.sum(gen_weights))

    reco_weights = np.concatenate([sample_weights["match"], sample_weights["reco_fake"]])
    sample_weights["reco"] = reco_weights*(np.sum(sample_weights["data"])/np.sum(reco_weights))

    reco_stratify = reco_table["stratification_labels"].to_numpy()
    gen_stratify = gen_table["stratification_labels"].to_numpy()

    undsmp_rnd_sts = [None, None]
    if undersample_rng is not None:
        undsmp_rnd_sts = undersample_rng.integers(INTMAX, size=2).tolist()

    data_train, reco_train, data_train_indices, reco_train_indices = undersample(data_table, reco_table, 
                                                                            max_class_size=train_class_size, 
                                                                            stratify_pos=None, 
                                                                            stratify_neg=reco_stratify,
                                                                            random_state=undsmp_rnd_sts[0])
    sample_weights["data_train"]= sample_weights["data"][data_train_indices]
    sample_weights["data_train"]= sample_weights["data_train"]/np.mean(sample_weights["data_train"])
    
    sample_weights["reco_train"] = sample_weights["reco"][reco_train_indices]
    sample_weights["reco_train"] = sample_weights["reco_train"]/np.mean(sample_weights["reco_train"])


    gen_train_pos, gen_train_neg, gen_train_pos_indices, gen_train_neg_indices = undersample(gen_table, gen_table, 
                                                                                max_class_size=train_class_size, 
                                                                                stratify_pos=gen_stratify, 
                                                                                stratify_neg=gen_stratify,
                                                                                random_state=undsmp_rnd_sts[1])
    
    sample_weights["gen_train_pos"]= sample_weights["gen"][gen_train_pos_indices]
    sample_weights["gen_train_pos"]= sample_weights["gen_train_pos"]/np.mean(sample_weights["gen_train_pos"])
    
    sample_weights["gen_train_neg"]= sample_weights["gen"][gen_train_neg_indices]
    sample_weights["gen_train_neg"]= sample_weights["gen_train_neg"]/np.mean(sample_weights["gen_train_neg"])

    valid_size=0.2 

    checkpoint_dir=f"{output_folder}/checkpoints"
    n_seeds_needed=8
    model_seeds = [None]*n_seeds_needed
    iseed = 0
    if model_rng is not None:
        model_seeds = model_rng.integers(INTMAX, size=n_seeds_needed).tolist()

    #################################################################################################################################################################################
    print(f"{iseed} seeds used, Initializing datasets to pull reco level...")
    pull_inputs = pa.concat_tables([data_train, reco_train])
    pull_targets = np.concatenate([np.ones(len(data_train)), np.zeros(len(reco_train))])
    pull_sample_weights = np.concatenate([sample_weights["data_train"], sample_weights["reco_train"]])
    pull_train_ds, pull_valid_ds, pull_ds = get_split_datasets(pull_inputs, 
                                                               y=pull_targets, 
                                                               sample_weights=pull_sample_weights, 
                                                               column_names=column_names, 
                                                               random_state=model_seeds[iseed])
    iseed += 1
    #get_input_normalizer(pull_inputs, column_names=column_names)
    #fit_normalizer(pull_ds, ncols=len(column_names), batch_size=100000, collate_fn=Batch())
    #return
    pull_init_kwargs = deepcopy(init_kwargs)
    pull_init_kwargs["seed"] = model_seeds[iseed]
    pull_init_kwargs["train_split"] = predefined_split(pull_valid_ds)
    pull_init_kwargs["iterator_train__batch_size"] = batch_size
    pull_init_kwargs["iterator_valid__batch_size"] = 5*batch_size
    pull_init_kwargs["callbacks"] += [("checkpoint", Checkpoint(dirname=f"{checkpoint_dir}/pull", load_best=True))]
    pull_init_kwargs["module__input_transform"] = get_input_normalizer(pull_train_ds, ncols=n_features, batch_size=100000, collate_fn=Batch())
    pull_classifier = NeuralNet(**pull_init_kwargs)

    iseed += 1
    #################################################################################################################################################################################   
    print(f"{iseed} seeds used, Initializing datasets to push gen level...")
    push_inputs = pa.concat_tables([gen_train_pos, gen_train_neg])
    push_targets = np.concatenate([np.ones(len(gen_train_pos)), np.zeros(len(gen_train_neg))])
    push_sample_weights = np.concatenate([sample_weights["gen_train_pos"], sample_weights["gen_train_neg"]])
    push_train_ds, push_valid_ds, push_ds = get_split_datasets(push_inputs, 
                                                               y=push_targets, 
                                                               sample_weights=push_sample_weights, 
                                                               column_names=column_names, 
                                                               random_state=model_seeds[iseed])
    iseed += 1

    push_init_kwargs = deepcopy(init_kwargs)
    push_init_kwargs["seed"] = model_seeds[iseed]
    push_init_kwargs["train_split"] = predefined_split(push_valid_ds)
    push_init_kwargs["iterator_train__batch_size"] = batch_size
    push_init_kwargs["iterator_valid__batch_size"] = 5*batch_size 
    push_init_kwargs["callbacks"] += [("checkpoint", Checkpoint(dirname=f"{output_folder}/checkpoints/push", load_best=True))]
    push_init_kwargs["module__input_transform"] = get_input_normalizer(push_train_ds, ncols=n_features, batch_size=100000, collate_fn=Batch())
    push_classifier = NeuralNet(**push_init_kwargs)

    iseed += 1

    #################################################################################################################################################################################
    if gen_miss_table is not None:
        print(f"{iseed} seeds used, Initializing datasets to pull misses...")
        pull_miss_inputs = pa.concat_tables([gen_match_table, gen_match_table])
        pull_miss_targets = np.concatenate([np.ones(len(gen_match_table)), np.zeros(len(gen_match_table))])
        pull_miss_sample_weights = np.concatenate([sample_weights["match"], sample_weights["match"]])
        pull_miss_train_ds, pull_miss_valid_ds, pull_miss_ds = get_split_datasets(pull_miss_inputs, 
                                                               y=pull_miss_targets, 
                                                               sample_weights=pull_miss_sample_weights, 
                                                               column_names=column_names, 
                                                               random_state=model_seeds[iseed])

        iseed += 1
        
        pull_miss_init_kwargs = deepcopy(init_kwargs)
        pull_miss_init_kwargs["seed"] = model_seeds[iseed]
        pull_miss_init_kwargs["train_split"] = predefined_split(pull_miss_valid_ds)
        pull_miss_init_kwargs["iterator_train__batch_size"] = batch_size
        pull_miss_init_kwargs["iterator_valid__batch_size"] = 5*batch_size 
        pull_miss_init_kwargs["callbacks"] += [("checkpoint", Checkpoint(dirname=f"{output_folder}/checkpoints/pull_miss", load_best=True))]
        pull_miss_init_kwargs["module__input_transform"] = get_input_normalizer(pull_miss_train_ds, ncols=n_features, batch_size=100000, collate_fn=Batch())
        pull_miss_classifier = NeuralNet(**pull_miss_init_kwargs)

        iseed += 1

    #################################################################################################################################################################################
    if reco_fake_table is not None:
        print(f"{iseed} seeds used, Initializing datasets to push fakes...")
        push_fake_inputs = pa.concat_tables([reco_match_table, reco_match_table])
        push_fake_targets = np.concatenate([np.ones(len(reco_match_table)), np.zeros(len(reco_match_table))])
        push_fake_sample_weights = np.concatenate([sample_weights["match"], sample_weights["match"]])
        push_fake_train_ds, push_fake_valid_ds, push_fake_ds = get_split_datasets(push_fake_inputs, 
                                                               y=push_fake_targets, 
                                                               sample_weights=push_fake_sample_weights, 
                                                               column_names=column_names, 
                                                               random_state=model_seeds[iseed])
        iseed += 1

        push_fake_init_kwargs = deepcopy(init_kwargs)
        push_fake_init_kwargs["seed"] = model_seeds[iseed]
        push_fake_init_kwargs["train_split"] = predefined_split(push_fake_valid_ds)
        push_fake_init_kwargs["iterator_train__batch_size"] = batch_size
        push_fake_init_kwargs["iterator_valid__batch_size"] = 5*batch_size 
        push_fake_init_kwargs["callbacks"] += [("checkpoint", Checkpoint(dirname=f"{output_folder}/checkpoints/push_fakes", load_best=True))]
        
        push_fake_init_kwargs["module__input_transform"] = get_input_normalizer(push_fake_train_ds, ncols=n_features, batch_size=100000, collate_fn=Batch())
        push_fake_classifier = NeuralNet(**push_fake_init_kwargs)

        iseed += 1
    ################################################################################################################################################################################# 
    w_unfolding = [sample_weights["gen"], sample_weights["reco"]]

    for iteration in range(n_iterations):
        print("###############################################################################################")
        print(f"Iteration: {iteration+1}/{n_iterations}")
        print("###############################################################################################")

        print("---Setting reco weights...")
        sample_weights["reco_train"] = w_unfolding[-1][reco_train_indices]
        sample_weights["reco_train"] = sample_weights["reco_train"]/np.mean(sample_weights["reco_train"])
        pull_ds.sample_weights =np.concatenate([sample_weights["data_train"], sample_weights["reco_train"]])

        print("---Calculating pull weights for gen matches...")
        pull_classifier.fit(pull_train_ds)
        match_reweight = get_reweighting(pull_classifier, reco_match_table)
        reco_fake_reweight = get_reweighting(pull_classifier, reco_fake_table)

        #print(sample_weights["match"].shape, match_reweight.shape)

        sample_weights["match"] *= match_reweight
        sample_weights["reco_fake"] *= reco_fake_reweight

        #################################################################################################################################################################################
        if gen_miss_table is not None: 
            print("---Calculating pull weights for gen misses...")
            pull_miss_ds.sample_weights = np.concatenate([match_reweight, np.ones(len(gen_match_table))])
            pull_miss_classifier.fit(pull_miss_train_ds)
            gen_miss_reweight = get_reweighting(pull_miss_classifier, gen_miss_table)
            sample_weights["gen_miss"] *= gen_miss_reweight
        #################################################################################################################################################################################        
        gen_weights = np.concatenate([sample_weights["match"], sample_weights["gen_miss"]])
        sample_weights["gen"] = gen_weights*np.sum(sample_weights["data"])/np.sum(gen_weights) 
        w_unfolding.append(sample_weights["gen"])
        #################################################################################################################################################################################
        
        print("---Setting gen weights...")
        #genlvl_ds.set_sample_weights(genlvl_weights)
        sample_weights["gen_train_pos"] = w_unfolding[-1][gen_train_pos_indices]
        sample_weights["gen_train_pos"] = sample_weights["gen_train_pos"]/np.mean(sample_weights["gen_train_pos"])

        sample_weights["gen_train_neg"] = w_unfolding[-3][gen_train_neg_indices]
        sample_weights["gen_train_neg"] = sample_weights["gen_train_neg"]/np.mean(sample_weights["gen_train_neg"])

        #genlvl_weights = np.concatenate([w_unfolding[-1], w_unfolding[-3]])
        push_ds.sample_weights = np.concatenate([sample_weights["gen_train_pos"], sample_weights["gen_train_neg"]])
        
        print("---Calculating push weights for reco matches...")
        push_classifier.fit(push_train_ds)
        match_reweight = get_reweighting(push_classifier, gen_match_table)
        gen_miss_reweight = get_reweighting(push_classifier, gen_miss_table)
        
        sample_weights["match"] *= match_reweight
        sample_weights["gen_miss"] *= gen_miss_reweight
        #################################################################################################################################################################################
        if reco_fake_table is not None:
            print("---Calculating push weights for reco fakes...")
            push_fake_ds.sample_weights = np.concatenate([match_reweight, np.ones(len(reco_match_table))])
            push_fake_classifier.fit(push_fake_train_ds)
            reco_fake_reweight = get_reweighting(push_fake_classifier, reco_fake_table)

            sample_weights["reco_fake"] *= reco_fake_reweight
        #################################################################################################################################################################################
        reco_weights = np.concatenate([sample_weights["match"], sample_weights["reco_fake"]])
        sample_weights["reco"] = reco_weights*np.sum(sample_weights["data"])/np.sum(reco_weights)         
        w_unfolding.append(sample_weights["reco"])
        #################################################################################################################################################################################

        with open(f"{output_folder}/multifolded-wts-iter{iteration+1}.npz", "wb") as f:
            np.savez(f, *w_unfolding)
    
    print("Done!")
    
    return w_unfolding
       

if __name__ == "__main__":
    #os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    jet_columns = [
        "pt", "eta", "phi", "nef",
        "ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0",
        "leading_constit_pt", "leading_constit_eta", "leading_constit_phi",
        "subleading_constit_pt", "subleading_constit_eta", "subleading_constit_phi",
        "hc_pt", "hc_eta", "hc_phi",
        "hc_ch_ang_k1_b0.5", "hc_ch_ang_k1_b1", "hc_ch_ang_k1_b2", "hc_ch_ang_k2_b0",
        ]

    emb_data_folder = "outputs/30May25-1147"
    #output_folder=f"{emb_data_folder}/multifolding-{datetime.now().strftime("%d-%b-%y-%H-%M")}"
    #output_folder=f"{emb_data_folder}/multifolding_wsys_hadr_corr"

    pth_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
    pth_bin_folders = [f"{emb_data_folder}/ptHat{pth_low}to{pth_high}" for pth_low, pth_high in zip(pth_bins[:-1], pth_bins[1:])]
    n_pth_bins = len(pth_bin_folders)
    matched_pth_label = list(range(1, n_pth_bins+1))
    unmatched_pth_label = list(range(n_pth_bins+1, 2*n_pth_bins+1))
    #print(matched_pth_label, unmatched_pth_label, n_pth_bins)

    stratification_label_key = "stratification_labels"
    gen_match_buffers, gen_match_table = pa_concated_table([f"{folder}/gen-matches.arrow" for folder in pth_bin_folders], label=matched_pth_label, label_key=stratification_label_key)
    gen_miss_buffers, gen_miss_table = pa_concated_table([f"{folder}/misses.arrow" for folder in pth_bin_folders], label=unmatched_pth_label, label_key=stratification_label_key) 
    reco_match_buffers, reco_match_table = pa_concated_table([f"{folder}/reco-matches.arrow" for folder in pth_bin_folders], label=matched_pth_label, label_key=stratification_label_key)
    reco_fake_buffers, reco_fake_table = pa_concated_table([f"{folder}/fakes.arrow" for folder in pth_bin_folders], label=unmatched_pth_label, label_key=stratification_label_key)

    #data_buffer, data_table = pa_table("outputs/jets-conPtMin0.2.arrow", label=0, label_key="stratify_labels")
    #data_buffer, data_table = pa_table("outputs/jets_conPtMin0.2_wTowerHadrCorrSys.arrow", label=0, label_key="stratify_labels")
    data_buffer, data_table = pa_table("outputs/jets-conPtMin0.2.arrow", label=0, label_key=stratification_label_key)

    #data_table = pa.concat_tables([reco_match_table, reco_fake_table])
    #omniseq_wt_list = np.load(f"{emb_data_folder}/omnisequential_1/omniseq-wts-iter10.npz")
    #omniseq_iter = 9
    #omniseq_wt = omniseq_wt_list[f"arr_{2*omniseq_iter+1}"]
    #data_table = data_table.set_column(0, "weight", pa.array(omniseq_wt, pa.float32()))

    #print(data_table.take([0,1]))
    #print(reco_match_table.take([0,1]))

    n_data = len(data_table)

    n_gen_matches = len(gen_match_table)
    n_gen_misses = len(gen_miss_table) if gen_miss_table is not None else 0
    n_reco_matches = len(reco_match_table)
    n_reco_fakes = len(reco_fake_table) if reco_fake_table is not None else 0

    assert n_gen_matches == n_reco_matches

    n_matches = n_gen_matches 
    n_gen = n_matches + n_gen_misses
    n_reco = n_matches + n_reco_fakes

    print("Number of matched gen jets, matched reco jets:", n_gen_matches, n_reco_matches, n_matches)
    print("Number of missed gen jets, fake reco jets:", n_gen_misses, n_reco_fakes)
    print("Number of data jets:", n_data)

    device = "cuda"
    n_features = len(jet_columns) 
    
    num_epochs = 50
    patience = 10
    valid_size = 0.4
    batch_size = 10000
    
    n_iterations = 10
    train_class_size = 500000

    init_kwargs = {}
    common_dl_kwargs = {"collate_fn" : Batch(), 
                        "num_workers" : 10, 
                        "pin_memory" : device == "cuda",
                        "persistent_workers" : False
                        }

    early_stop_kwargs = {"patience" : 10,
                         "load_best" : True
                         }

    lr_scheduler_kwargs = {"policy" : "ReduceLROnPlateau",
                           "monitor" : "valid_loss",
                           "patience" :5,
                           "factor" : 0.5,
                           }
    
    init_kwargs["module"] = DNN
    init_kwargs["module__layer_sizes"] = [n_features, 256, 256, 256, 1] 
    
    init_kwargs["criterion"] = binary_cross_entropy_with_logits
    init_kwargs["optimizer"] = torch.optim.Adam
    
    init_kwargs["dataset"] = ArrowDataset
    init_kwargs["dataset__column_names"] = jet_columns
    
    init_kwargs["batch_size"] = batch_size
    init_kwargs["lr"] = 0.01
    init_kwargs["callbacks"] = [
        ("early_stopping", EarlyStopping(**early_stop_kwargs)),
        ("lr_scheduler", LRScheduler(**lr_scheduler_kwargs))
        #("progress_bar", ProgressBar()),
    ] 
    #init_kwargs["seed"] = model_seed
    init_kwargs["device"] = device
    init_kwargs["compile"] = True
    init_kwargs["max_epochs"]=num_epochs
    init_kwargs["predict_nonlinearity"]=torch.sigmoid
    init_kwargs.update({f"iterator_train__{key}" : deepcopy(value) for key, value in common_dl_kwargs.items()})
    init_kwargs["iterator_train__shuffle"] = True
    init_kwargs.update({f"iterator_valid__{key}" : deepcopy(value) for key, value in common_dl_kwargs.items()})
    #init_kwargs["train_split__cv"] = valid_size
 
#    print(callable(binary_cross_entropy_with_logits), type(binary_cross_entropy_with_logits))
#    print(init_kwargs["criterion"], callable(init_kwargs["criterion"]))
  
    model_seed = 0
    for undersample_rnd_seed in range(10):
        undersample_rng = np.random.default_rng(seed=undersample_rnd_seed)
        model_rng = np.random.default_rng(seed=model_seed)
        set_global_random_seed(model_seed, make_cuda_deterministic = device=="cuda")
        output_folder = f"outputs/seed_var/mltfld_dataseed{undersample_rnd_seed}_torchseed{model_seed}"
        #output_folder=f"{emb_data_folder}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        w_unfolding = multifold(n_iterations, 
                            data_table, 
                            reco_match_table, 
                            gen_match_table, 
                            reco_fake_table=reco_fake_table, 
                            gen_miss_table=gen_miss_table, 
                            output_folder=output_folder, 
                            init_kwargs=init_kwargs,
                            train_class_size=train_class_size,
                            undersample_rng=undersample_rng,
                            model_rng=model_rng,
                            column_names=jet_columns,
                            )
