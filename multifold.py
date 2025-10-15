import os

import numpy as np

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from tensordict import TensorDict

from churten.ensemble import Ensemble
from churten.optimizer import Adam

from churten.nn.archs import MLP
from churten.nn.transform import Normalize

from utils.data import TensorDictDataset
from utils.data import train_test_multi_loaders, reweight_inference_loaders

if __name__ == "__main__":
    num_replicas = 10
    num_data_subsample = 500000
    batch_size = 10000
    train_size = 0.6
    num_iterations = 10 
    num_epochs = 20
    dataseed = 42
    modelseed = 0
    filename_stub_fmt = "nreplicas{}_sub{}_batch_size{}_niter{}_nepochs{}_trainsplit{}_dataseed{}_modelseed{}"
    filename_stub = filename_stub_fmt.format(
        num_replicas, num_data_subsample, batch_size, num_iterations, num_epochs, 
        f"{int(train_size*100)}{int((1-train_size)*100)}", dataseed, modelseed,
    )
    print("Associated files will be saved with suffix:", filename_stub)

    generator = torch.Generator().manual_seed(dataseed)
    
    src = "partitioned_datasets/nominal"
    
    detlvl_src = f"{src}/det_lvl/all"
    detlvl_td = TensorDict.load_memmap(detlvl_src)
    print("Loaded (data, reco) TensorDict from:", detlvl_src)

    detlvl_ds = TensorDictDataset(detlvl_td, is_categorical=True, num_replicas=num_replicas)
    detlvl_pos_mask = detlvl_ds.target > 0.5
    detlvl_neg_mask = detlvl_ds.target < 0.5

    detlvl_matched_indices = detlvl_ds.indices[detlvl_td["is_matched"] == 1]
    fake_scaling_indices = torch.cat([detlvl_matched_indices]*2)
    fake_scaling_targets = torch.cat([
        torch.ones_like(detlvl_matched_indices, dtype=torch.float32), 
        torch.zeros_like(detlvl_matched_indices, dtype=torch.float32),
    ])
    fake_scaling_weights = torch.ones_like(fake_scaling_indices, dtype=torch.float32)
    fake_scaling_ds = TensorDictDataset(
        detlvl_td,
        indices=fake_scaling_indices,
        target=fake_scaling_targets,
        sample_weight=fake_scaling_weights,
        is_categorical=True, 
        num_replicas=num_replicas,
    )
    fake_scaling_pos_mask = fake_scaling_ds.target > 0.5
    fake_scaling_neg_mask = fake_scaling_ds.target < 0.5

    detlvl_train_loader, detlvl_valid_loader = train_test_multi_loaders(
        detlvl_ds,
        train_size = train_size,
        undersample_size=num_data_subsample,
        batch_size=batch_size,
        num_replicas=num_replicas,
        generator=generator,
        stratifys=(False, True),
    )
    fake_scaling_train_loader, fake_scaling_valid_loader = train_test_multi_loaders(
        fake_scaling_ds,
        train_size = train_size,
        undersample_size=num_data_subsample,
        batch_size=batch_size,
        num_replicas=num_replicas,
        generator=generator,
        stratifys=(True, True),
    )
    reco_match_loader, reco_fake_loader = reweight_inference_loaders(
        detlvl_ds,
        batch_size=batch_size*5,
        num_replicas=num_replicas,
    )
    print(
        "Initialized loaders to get"
        " f : p(reco) --> p(data) and"
        " g : p(reco fake) --> p(reco match)"
    )

    partlvl_src = f"{src}/part_lvl/all"
    partlvl_td = TensorDict.load_memmap(partlvl_src)
    print("Loaded particle level TensorDict from:", partlvl_src)
    
    partlvl_ds = TensorDictDataset(partlvl_td, is_categorical=True, num_replicas=num_replicas)
    partlvl_pos_mask = partlvl_ds.target > 0.5
    partlvl_neg_mask = partlvl_ds.target < 0.5

    partlvl_matched_indices = partlvl_ds.indices[partlvl_td["is_matched"] == 1]
    miss_scaling_indices = torch.cat([partlvl_matched_indices]*2)
    miss_scaling_targets = torch.cat([
        torch.ones_like(partlvl_matched_indices, dtype=torch.float32), 
        torch.zeros_like(partlvl_matched_indices, dtype=torch.float32),
    ])
    miss_scaling_weights = torch.ones_like(miss_scaling_indices, dtype=torch.float32)
    miss_scaling_ds = TensorDictDataset(
        partlvl_td,
        indices=miss_scaling_indices,
        target=miss_scaling_targets,
        sample_weight=miss_scaling_weights,
        is_categorical=True, 
        num_replicas=num_replicas,
    )
    miss_scaling_pos_mask = miss_scaling_ds.target > 0.5
    miss_scaling_neg_mask = miss_scaling_ds.target < 0.5


    partlvl_train_loader, partlvl_valid_loader = train_test_multi_loaders(
        partlvl_ds,
        train_size = train_size,
        undersample_size=num_data_subsample,
        batch_size=batch_size,
        num_replicas=num_replicas,
        generator=generator,
        stratifys=(True, True),
    ) 
    miss_scaling_train_loader, miss_scaling_valid_loader = train_test_multi_loaders(
        miss_scaling_ds,
        train_size = train_size,
        undersample_size=num_data_subsample,
        batch_size=batch_size,
        num_replicas=num_replicas,
        generator=generator,
        stratifys=(True, True),
    ) 
    gen_match_loader, gen_miss_loader = reweight_inference_loaders(
        partlvl_ds,
        batch_size=batch_size*5,
        num_replicas=num_replicas,
    )
    print(
        "Initialized loaders to get"
        " f : p(gen) --> p(unfolded) and"
        " g : p(gen miss) --> p(gen match)"
    )

    n_data = detlvl_ds.td[detlvl_ds.td["is_data"]].size(0)
    n_reco = detlvl_ds.td[torch.logical_not(detlvl_ds.td["is_data"])].size(0)
    n_gen = partlvl_ds.td.size(0)

    print("Number of data jets:", n_data)
    print("Number of gen jets, reco jets:", n_gen, n_reco)

    num_features = 21
    layer_sizes = [num_features, 256, 256, 256, 1]
    device = "cuda"
    torch.manual_seed(modelseed)
    
    detlvl_std, detlvl_mean = torch.std_mean(detlvl_ds.input)
    detlvl_optimizer = Adam(lr = 1e-3, batch_size=[num_replicas], device=device)
    detlvl_ensemble = Ensemble(
        MLP, model_init_kwargs={
            "layer_sizes" : layer_sizes,
            "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform" : Normalize(detlvl_mean.to(device), detlvl_std.to(device)), 
            "dropout_prob" : 0.2,
        },
        criterion=binary_cross_entropy_with_logits,
        num_replicas=num_replicas,
        device=device,
        model_init_randomness="different",
    )
    detlvl_optimizer.init(detlvl_ensemble.params_dict)

    gen_match_std, gen_match_mean = torch.std_mean(partlvl_ds.input[partlvl_matched_indices])
    miss_scaling_optimizer = Adam(lr = 1e-3, batch_size=[num_replicas], device=device)
    miss_scaling_ensemble = Ensemble(
        MLP, model_init_kwargs={
            "layer_sizes" : layer_sizes,
            "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform" : Normalize(gen_match_mean.to(device), gen_match_std.to(device)), 
            "dropout_prob" : 0.2,
        },
        criterion=binary_cross_entropy_with_logits,
        num_replicas=num_replicas,
        device=device,
        model_init_randomness="different",
    )
    miss_scaling_optimizer.init(miss_scaling_ensemble.params_dict)

    partlvl_std, partlvl_mean = torch.std_mean(partlvl_ds.input)
    partlvl_optimizer = Adam(lr = 1e-3, batch_size=[num_replicas], device=device)
    partlvl_ensemble = Ensemble(
        MLP, model_init_kwargs={
            "layer_sizes" : layer_sizes,
            "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform" : Normalize(partlvl_mean.to(device), partlvl_std.to(device)),
            "dropout_prob" : 0.2,
        },
        criterion=binary_cross_entropy_with_logits,
        num_replicas=num_replicas,
        device=device,
        model_init_randomness="different",
    )
    partlvl_optimizer.init(partlvl_ensemble.params_dict)

    reco_match_std, reco_match_mean = torch.std_mean(detlvl_ds.input[detlvl_matched_indices])
    fake_scaling_optimizer = Adam(lr = 1e-3, batch_size=[num_replicas], device=device)
    fake_scaling_ensemble = Ensemble(
        MLP, model_init_kwargs={
            "layer_sizes" : layer_sizes,
            "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform" : Normalize(reco_match_mean.to(device), reco_match_std.to(device)), 
            "dropout_prob" : 0.2,
        },
        criterion=binary_cross_entropy_with_logits,
        num_replicas=num_replicas,
        device=device,
        model_init_randomness="different",
    )
    fake_scaling_optimizer.init(fake_scaling_ensemble.params_dict)


    w_unfolding = [
        partlvl_ds.sample_weight[..., partlvl_neg_mask].detach().numpy(), 
        detlvl_ds.sample_weight[..., detlvl_neg_mask].detach().numpy(),
    ]

    dir_prefix = f"./outputs/unfolding_{filename_stub}"
    if not os.path.exists(dir_prefix):
        os.makedirs(dir_prefix)

    for iteration in range(num_iterations):
        print("###############################################################################################")
        print(f"Iteration: {iteration+1}/{num_iterations}")
        print("###############################################################################################")

        detlvl_history = detlvl_ensemble.fit(
            detlvl_optimizer,
            detlvl_train_loader, 
            detlvl_valid_loader,
            num_epochs=num_epochs,
        )

        reco_match_preds = detlvl_ensemble.predict(reco_match_loader, non_linearity = sigmoid)
        reco_match_reweights = (reco_match_preds/(1. - reco_match_preds + 1e-20)).squeeze_()
        fake_preds = detlvl_ensemble.predict(reco_fake_loader, non_linearity = sigmoid)
        fake_reweights = (fake_preds/(1. - fake_preds + 1e-20)).squeeze_()
        reco_reweights = torch.cat([reco_match_reweights, fake_reweights], dim=1)

        detlvl_wts = detlvl_ds.sample_weight.clone()
        detlvl_wts[..., detlvl_neg_mask] *= reco_reweights
        detlvl_ds.sample_weight = detlvl_wts

        miss_scaling_ds.sample_weight[..., miss_scaling_pos_mask] = reco_match_reweights
        miss_scaling_history = miss_scaling_ensemble.fit(
            miss_scaling_optimizer,
            miss_scaling_train_loader, 
            miss_scaling_valid_loader,
            num_epochs=num_epochs,
        )

        miss_preds = miss_scaling_ensemble.predict(gen_miss_loader, non_linearity = sigmoid)
        miss_reweights = (miss_preds/(1. - miss_preds + 1e-20)).squeeze_()
        gen_reweights = torch.cat([reco_match_reweights, miss_reweights], dim=1)

        partlvl_wts = partlvl_ds.sample_weight.clone()
        partlvl_wts[..., partlvl_pos_mask] *= gen_reweights
        partlvl_ds.sample_weight = partlvl_wts
        w_unfolding.append(partlvl_ds.sample_weight[..., partlvl_pos_mask].detach().numpy())

        partlvl_history = partlvl_ensemble.fit(
            partlvl_optimizer,
            partlvl_train_loader,
            partlvl_valid_loader,
            num_epochs=num_epochs,
        )
        gen_match_preds = partlvl_ensemble.predict(gen_match_loader, non_linearity = sigmoid)
        gen_match_reweights = (gen_match_preds/(1. - gen_match_preds + 1e-20)).squeeze_()
        miss_preds = partlvl_ensemble.predict(gen_miss_loader, non_linearity = sigmoid)
        miss_reweights = (miss_preds/(1. - miss_preds + 1e-20)).squeeze_()
        gen_reweights = torch.cat([gen_match_reweights, miss_reweights], dim=1)
        
        partlvl_wts = partlvl_ds.sample_weight.clone()
        partlvl_wts[..., partlvl_neg_mask] *= gen_reweights
        partlvl_ds.sample_weight = partlvl_wts

        fake_scaling_ds.sample_weight[..., fake_scaling_pos_mask] = gen_match_reweights
        fake_scaling_history = fake_scaling_ensemble.fit(
            fake_scaling_optimizer,
            fake_scaling_train_loader, 
            fake_scaling_valid_loader,
            num_epochs=num_epochs,
        )

        fake_preds = miss_scaling_ensemble.predict(reco_fake_loader, non_linearity = sigmoid)
        fake_reweights = (fake_preds/(1. - fake_preds + 1e-20)).squeeze_()
        reco_reweights = torch.cat([gen_match_reweights, fake_reweights], dim=1)

        detlvl_wts = detlvl_ds.sample_weight.clone()
        detlvl_wts[..., detlvl_neg_mask] *= reco_reweights
        detlvl_ds.sample_weight = detlvl_wts
        w_unfolding.append(detlvl_ds.sample_weight[..., detlvl_neg_mask].detach().numpy())

        with open(f"{dir_prefix}/w_unfolding_{iteration+1}.npz", "wb") as f:
            np.savez(f, *w_unfolding)

    print("Done !")


























    
    


    
    
    
        
        



    





