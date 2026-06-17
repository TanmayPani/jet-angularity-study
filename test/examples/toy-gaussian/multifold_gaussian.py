import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid
from torch.utils.data import Dataset
from torch.distributions import Binomial

from tensordict import TensorDict

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.callbacks import EarlyStopping
from torchstrap.utils.nn.archs import MLP

from utils.data import TensorDictDataset
from utils.data import train_test_multi_loaders, reweight_inference_loaders

def drop_random(prob, num_samples):
    dist = Binomial(1, prob)
    return dist.sample((num_samples,)).to(torch.bool)

def random_data(
    mu, 
    sigma, 
    epsilon,
    *, 
    num_samples,
    efficiency = None,
    fake_rate = None,
    dummy_val = -10 ,
):
    x_part= torch.normal(mu, sigma, (num_samples,))
    x_det = torch.normal(x_part, epsilon)

    if efficiency is not None:
        is_part_match = drop_random(1.-efficiency, num_samples)
        x_det[is_part_match.logical_not_()] = dummy_val
    
    if fake_rate is not None:
        is_det_match = drop_random(1.-fake_rate, num_samples)
        x_part[is_det_match.logical_not_()] = dummy_val

    return x_part, x_det

def make_dataset(pos_data, neg_data, num_replicas, *, dummy_val=-10):
    _pos_data = pos_data[pos_data != dummy_val]
    _neg_data = neg_data[neg_data != dummy_val]
    x = torch.cat((_pos_data, _neg_data))
    td = TensorDict( 
        input = x.unsqueeze_(-1),
        target = torch.cat(
            (torch.ones(_pos_data.shape[0]), torch.zeros(_pos_data.shape[0]))
        ),
        weight = torch.ones(x.shape[0])
    ).auto_batch_size_()

    return TensorDictDataset(td, is_categorical=True, num_replicas=num_replicas)

if __name__ == "__main__":
    torch.manual_seed(0)
    num_samples = 150000
    num_replicas = 50
    train_size = 0.75
    num_data_subsample = 10**5
    num_iterations = 6
    num_epochs = 200
    layer_sizes = [1, 50, 50, 50, 1]
    device="cuda"
    optimizer_kwargs = dict(
        lr = 0.001, 
        eps=1e-7,
        decoupled_weight_decay=False,
        weight_decay=0.0, 
    )

    x_gen, x_reco = random_data(0., 1., 0.5, num_samples=num_samples)    
    x_truth, x_data = random_data(0.2, 0.8, 0.5, num_samples=num_samples)  

    detlvl_ds =  make_dataset(x_data, x_reco, num_replicas)
    partlvl_ds =  make_dataset(x_gen, x_gen, num_replicas)
    
    detlvl_train_loader, detlvl_valid_loader = train_test_multi_loaders(
        detlvl_ds,
        train_size = train_size,
        undersample_size = num_data_subsample,
        batch_size = 10000,
        num_replicas = num_replicas,
    )

    detlvl_rewt_loader = reweight_inference_loaders(
        detlvl_ds,
        batch_size=50000,
        num_replicas=num_replicas,
    )

    detlvl_callbacks = [
        ("early_stopping", EarlyStopping(patience=10)),
    ]

    detlvl_ensemble, _, detlvl_state = StatelessModule.init(
        MLP, 
        Adam,
        model_init_kwargs={
            "layer_sizes" : layer_sizes,
        },
        num_replicas=num_replicas,
        device=device,
        init_randomness="different",
        **optimizer_kwargs,
    )

    partlvl_train_loader,partlvl_valid_loader = train_test_multi_loaders(
        partlvl_ds,
        train_size = train_size,
        undersample_size = num_data_subsample,
        batch_size = 2000,
        num_replicas = num_replicas,
    )

    partlvl_rewt_loader = reweight_inference_loaders(
        partlvl_ds,
        batch_size=50000,
        num_replicas=num_replicas,
    )

    partlvl_callbacks = [
        ("early_stopping", EarlyStopping(patience=10)),
    ]

    partlvl_ensemble, _, partlvl_state = StatelessModule.init(
        MLP, 
        Adam,
        model_init_kwargs={
            "layer_sizes" : layer_sizes,
        },
        num_replicas=num_replicas,
        device=device,
        init_randomness="different",
        **optimizer_kwargs,
    )

    w_data = torch.ones(
        int(detlvl_ds.target.sum().item()), 
        dtype=torch.float32,
    ).expand(num_replicas, -1)
    w_unf = [detlvl_ds.weight[..., detlvl_ds.target < 0.5].clone()]
    num_epochs = num_epochs
    dir_prefix = f"./outputs/gaussian/unfolding_"
    if not os.path.exists(dir_prefix):
        os.makedirs(dir_prefix)

    epsilon = 1e-20

    for iter in range(num_iterations):
        print(f"Iteration : {iter+1}/{num_iterations}")
        detlvl_ds.sample_weight = torch.cat((w_data, w_unf[-1]), dim=1)
        detlvl_history = detlvl_ensemble.fit(
            Adam, 
            binary_cross_entropy_with_logits,
            detlvl_state,
            detlvl_train_loader, 
            valid_iterator = detlvl_valid_loader, 
            num_epochs=num_epochs,
            callbacks = detlvl_callbacks,
            randomness="different",
        )
        preds = torch.clip(
            detlvl_ensemble.predict(detlvl_state, detlvl_rewt_loader, non_linearity=sigmoid),
            min=epsilon,
            max=1-epsilon,
        )
        rewts = (preds/(1. - preds + epsilon)).squeeze_()
        detlvl_state.reset_status()
       
        _w_unf = w_unf[-1]*rewts
        w_unf.append(_w_unf/_w_unf.sum()*w_data.sum())

        partlvl_ds.sample_weight = torch.cat((w_unf[-1], w_unf[-2]), dim=1)
        gen_history = partlvl_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits, 
            partlvl_state, 
            partlvl_train_loader, 
            valid_iterator = partlvl_valid_loader, 
            num_epochs=num_epochs,
            callbacks = partlvl_callbacks,
            randomness="different",
        )
        preds = torch.clip(
                partlvl_ensemble.predict(partlvl_state, partlvl_rewt_loader, non_linearity=sigmoid),
                min=epsilon,
                max=1-epsilon,
        )
        rewts = (preds/(1. - preds + epsilon)).squeeze_()
        partlvl_state.reset_status()

        _w_unf = w_unf[-1]*rewts
        w_unf.append(_w_unf/_w_unf.sum()*w_data.sum())

    print("Done !")


    bins = torch.linspace(-5, 5, 20)
    x = 0.5*(bins[1:] + bins[:-1])
    dx = 0.5*(bins[1:] - bins[:-1])

    h_gen, _= torch.histogram(x_gen  , bins=bins,  density=True)
    h_reco, _= torch.histogram(x_reco , bins=bins,  density=True)
    h_data, _ = torch.histogram(x_data , bins=bins,  density=True)
    h_truth, _ = torch.histogram(x_truth, bins=bins,  density=True)

    common_kwargs = dict(
        xerr=dx,  
        markersize=10,
        markeredgecolor="white",
        linestyle="none"
    )
    
    ax = plt.figure().add_subplot(projection='3d')
    
    for iter in range(num_iterations+1):
        y = torch.full_like(x, iter)
        h_unf_stacked = torch.empty((num_replicas, len(bins)-1), dtype=torch.float32)
        h_unf_list = h_unf_stacked.unbind()
        for ireplica in range(num_replicas):
            h_unf_list[ireplica].copy_(
                torch.histogram(
                x_gen, bins=bins, weight=w_unf[2*iter][ireplica], density=True,
                )[0]
            )

        h_unf_std, h_unf_mean = torch.std_mean(h_unf_stacked, dim=0)
        ax.fill_between(x, iter, h_unf_mean - h_unf_std, x, iter, h_unf_mean + h_unf_std, alpha=0.4, color="red")
        if iter == num_iterations:
            gen_kwargs = dict(label="gen", **common_kwargs)
            truth_kwargs = dict(label="truth", **common_kwargs)
        else:
            gen_kwargs = common_kwargs
            truth_kwargs = common_kwargs
        gen_plot = ax.errorbar(x, y, h_gen, marker="^", markerfacecolor="blue", **gen_kwargs)
        truth_plot = ax.errorbar(x, y, h_truth, marker="s", markerfacecolor="red", **truth_kwargs)
        #ax.errorbar(x, y, h_reco, label="reco", marker="v", **common_kwargs)
        #ax.errorbar(x, y, h_data, label="data", marker="o", **common_kwargs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()

    ax.legend()
    plt.show()







