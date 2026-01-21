import os
import json
from typing import no_type_check

import numpy as np

import matplotlib.pyplot as plt 

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from tensordict import TensorDict

from churten.stateless import StatelessModule
from churten.optimizer import Adam
from churten.utils.nn.archs import MLP
from churten.callbacks import Checkpoint, EarlyStopping

from utils.data import TensorDictDataset
from utils.data import train_test_multi_loaders, reweight_inference_loaders

from utils import zjets

keys_to_stack = ["ms", "mults", "tau21s", "widths", "lnrhos", "zgs"]

def preprocessed_data_inputs(dataset, data_prefix, axs = None):
    data_prefix = os.path.join(
        data_prefix, zjets.FILENAME_PREFIX[dataset], "sim",
    )
    
    ds = TensorDict.load_memmap(data_prefix).to(dtype=torch.float32)
    ds_good = ds[ds["zgs"] > 0].unlock_()
    ds_good["lnrhos"]= torch.log(ds_good["sdms"]**2/ds_good["pts"]**2)
    ds_good["tau21s"] = ds_good["tau2s"]/ds_good["widths"]

    ds_std, ds_mean = ds_good.std(), ds_good.mean()
    ds_norm = ds_good.sub(ds_mean).div_(ds_std)
    
    if axs is not None:
        axs_flat = axs.flatten()
        iax = 0
        for key in keys_to_stack:
            #print(ds_std[key], ds_mean[key])
            axs_flat[iax].hist(
                ds_norm[key], density=True, histtype="step", label=f"{dataset}, sim"
            )
            axs_flat[iax].set_yscale("log")
            axs_flat[iax].set_xlabel(key)
            axs_flat[iax].legend()
            iax += 1 
 
    return torch.stack([ds_norm[key] for key in keys_to_stack], dim=1).to(torch.float32)

def preprocessed_sim_inputs(dataset, data_prefix, axs = None):
    data_prefixes = {
        lvl : os.path.join(
            data_prefix, zjets.FILENAME_PREFIX[dataset], lvl,
        ) for lvl in ("sim", "gen")
    }

    ds = {
        "sim" : TensorDict.load_memmap(data_prefixes["sim"]).to(dtype=torch.float32),
        "gen" : TensorDict.load_memmap(data_prefixes["gen"]).to(dtype=torch.float32),
    }
    sel = (ds["sim"]["zgs"] > 0) & (ds["gen"]["zgs"] > 0)
    ds_new = {}
    for lvl in ("sim", "gen"): 
        ds_new[lvl] = ds[lvl][sel].unlock_()
        ds_new[lvl]["lnrhos"]= torch.log(ds_new[lvl]["sdms"]**2/ds_new[lvl]["pts"]**2)
        ds_new[lvl]["tau21s"] = ds_new[lvl]["tau2s"]/ds_new[lvl]["widths"]

        ds_std, ds_mean = ds_new[lvl].std(), ds_new[lvl].mean()
        ds_new[lvl] = ds_new[lvl].sub(ds_mean).div_(ds_std)
    
        if axs is not None:
            axs_flat = axs.flatten()
            iax = 0
            for key in keys_to_stack:
                #print(ds_std[key], ds_mean[key])
                axs_flat[iax].hist(
                    ds_new[lvl][key], density=True, histtype="step", label=f"{dataset}, {lvl}"
                )
                axs_flat[iax].set_yscale("log")
                axs_flat[iax].set_xlabel(key)
                axs_flat[iax].legend()
                iax += 1 
 
    return (
        torch.stack([ds_new["sim"][key] for key in keys_to_stack], dim=1).to(torch.float32),
        torch.stack([ds_new["gen"][key] for key in keys_to_stack], dim=1).to(torch.float32),
    )


def preprocessed_tds(sim_dataset="pythia26", obj_dataset="herwig", cache_dir="."):

    data_prefix = os.path.join(cache_dir, 'datasets', 'ZjetsDelphes')
    #fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    axs = None
    sim_reco_input, sim_gen_input = preprocessed_sim_inputs(
        sim_dataset, data_prefix, axs=axs,
    )
    data_input = preprocessed_data_inputs(
        obj_dataset, data_prefix, axs=axs,
    )

    sim_input = torch.cat((data_input, sim_reco_input,))
    sim_target = torch.cat((
        torch.ones(data_input.shape[0], dtype=torch.float32), 
        torch.zeros(sim_reco_input.shape[0], dtype=torch.float32),
    ))
    sim_weight = torch.ones_like(sim_target, dtype=torch.float32)
    
    preproc_sim_prefix = os.path.join(data_prefix, "sim")
    
    sim_td = TensorDict(
        input = sim_input,
        target = sim_target,
        weight = sim_weight,
    ).auto_batch_size_().memmap_(preproc_sim_prefix)

    gen_input = torch.cat((sim_gen_input, sim_gen_input,))
    gen_target = torch.cat((
        torch.ones(sim_gen_input.shape[0], dtype=torch.float32), 
        torch.zeros(sim_gen_input.shape[0], dtype=torch.float32),
    ))
    gen_weight = torch.ones_like(gen_target, dtype=torch.float32)
    preproc_gen_prefix = os.path.join(data_prefix, "gen")
    gen_td = TensorDict(
        input  = gen_input,
        target = gen_target,
        weight = gen_weight,
    ).auto_batch_size_().memmap_(preproc_gen_prefix)

    #plt.show()
    
    return sim_td, gen_td

#@no_type_check
def main():
    cfg = {}
    with open('config.json', 'r') as config_file:
        cfg.update(json.load(config_file))

    filename_stub = cfg["filename_stub_fmt"].format(
        cfg["num_replicas"], 
        cfg["num_data_subsample"], 
        cfg["batch_size"], 
        cfg["num_iterations"], 
        cfg["num_epochs"], 
        f"{int(cfg["train_size"]*100)}{int((1-cfg["train_size"])*100)}", 
        cfg["dataseed"], 
        cfg["modelseed"],
    )
    print("Associated files will be saved with suffix:", filename_stub)

    generator = torch.Generator().manual_seed(cfg["dataseed"])

    sim_td, gen_td = preprocessed_tds()

    sim_ds = TensorDictDataset(sim_td, is_categorical=True, num_replicas=cfg["num_replicas"])
    sim_train_loader, sim_valid_loader = train_test_multi_loaders(
        sim_ds,
        train_size = cfg["train_size"],
        undersample_size = cfg["num_data_subsample"],
        batch_size = cfg["batch_size"],
        num_replicas = cfg["num_replicas"],
        generator = generator,
    )

    sim_rewt_loader = reweight_inference_loaders(
        sim_ds,
        batch_size=cfg["batch_size"]*5,
        num_replicas=cfg["num_replicas"],
    )
    
    gen_ds = TensorDictDataset(gen_td, is_categorical=True, num_replicas=cfg["num_replicas"])
    gen_train_loader, gen_valid_loader = train_test_multi_loaders(
        gen_ds,
        train_size = cfg["train_size"],
        undersample_size = cfg["num_data_subsample"],
        batch_size = cfg["batch_size"],
        num_replicas = cfg["num_replicas"],
        generator = generator,
    )

    gen_rewt_loader = reweight_inference_loaders(
        gen_ds,
        batch_size=cfg["batch_size"]*5,
        num_replicas=cfg["num_replicas"],
    )

    num_features = sim_td["input"].shape[-1]
    layer_sizes = [num_features, 100, 100, 1]
    print("Model layer sizes:", layer_sizes)

    device = "cuda"
    torch.manual_seed(cfg["modelseed"])
    
    optimizer_kwargs = dict(
        lr = 0.001, 
        eps=1e-7,
        decoupled_weight_decay=False,
        weight_decay=0.0, 
    )

    sim_callbacks = [
        ("early_stopping", EarlyStopping()),
        #("checkpoint", Checkpoint(root_dir="outputs/checkpoint_z_jets/sim"))
    ]
    sim_ensemble, _, sim_state = StatelessModule.init(
        MLP, 
        Adam,
        model_init_kwargs={
            "layer_sizes" : layer_sizes,
            #"activation" : torch.nn.Mish,
            #"batch_norm" : torch.nn.BatchNorm1d,
            #"dropout_prob" : 0.3,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different", 
        **optimizer_kwargs,
    )

    gen_callbacks = [
        ("early_stopping", EarlyStopping()),
        #("checkpoint", Checkpoint(root_dir="outputs/checkpoint_z_jets/gen"))
    ]

    gen_ensemble, _, gen_state = StatelessModule.init(
        MLP, 
        Adam,
        model_init_kwargs={
            "layer_sizes" : layer_sizes,
            #"activation" : torch.nn.Mish,
            #"batch_norm" : torch.nn.BatchNorm1d,
            #"dropout_prob" : 0.3,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        **optimizer_kwargs,
    )

    w_data = torch.ones(
        int(sim_ds.target.sum().item()), 
        dtype=torch.float32,
    ).expand(cfg["num_replicas"], -1)
    w_unf = [sim_ds.weight[..., sim_ds.target < 0.5].clone()]
    num_epochs = cfg["num_epochs"]
    dir_prefix = f"./outputs/cms_z_jets/unfolding_{filename_stub}"
    if not os.path.exists(dir_prefix):
        os.makedirs(dir_prefix)

    epsilon = 1e-20

    for iter in range(cfg["num_iterations"]):
        print(f"Iteration : {iter+1}/{cfg["num_iterations"]}")
        sim_ds.sample_weight = torch.cat((w_data, w_unf[-1]), dim=1)
        sim_history = sim_ensemble.fit(
            Adam, 
            binary_cross_entropy_with_logits,
            sim_state,
            sim_train_loader, 
            valid_iterator = sim_valid_loader, 
            num_epochs=num_epochs,
            callbacks = sim_callbacks,
            randomness="different",
        )
        preds = torch.clip(
            sim_ensemble.predict(sim_state, sim_rewt_loader, non_linearity=sigmoid),
            min=epsilon,
            max=1-epsilon,
        )
        rewts = (preds/(1. - preds + epsilon)).squeeze_()
        sim_state.reset_status()
       
        _w_unf = w_unf[-1]*rewts
        w_unf.append(_w_unf/_w_unf.sum()*w_data.sum())

        gen_ds.sample_weight = torch.cat((w_unf[-1], w_unf[-2]), dim=1)
        gen_history = gen_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits, 
            gen_state, 
            gen_train_loader, 
            valid_iterator = gen_valid_loader, 
            num_epochs=num_epochs,
            callbacks = gen_callbacks,
            randomness="different",
        )
        preds = torch.clip(
                gen_ensemble.predict(gen_state, gen_rewt_loader, non_linearity=sigmoid),
                min=epsilon,
                max=1-epsilon,
        )
        rewts = (preds/(1. - preds + epsilon)).squeeze_()
        gen_state.reset_status()

        _w_unf = w_unf[-1]*rewts
        w_unf.append(_w_unf/_w_unf.sum()*w_data.sum())

        with open(f"{dir_prefix}/w_unfolding_{iter+1}.npz", "wb") as f:
            np.savez(f, *[w.numpy() for w in w_unf])

    print("Done !")


if __name__ == "__main__":
    main()





     


    
    
    


