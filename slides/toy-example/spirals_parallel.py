from copy import deepcopy
from functools import partial
from math import pi

from awkward import num
from churten import optimizer
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import torch 
from torch.func import grad_and_value, vmap, functional_call
from torch.nn import Sequential, Linear, ReLU, LogSoftmax
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from torch.distributions import Categorical

from churten.ensemble import Ensemble
from churten.optimizer import Adam

def single_mlp_fit_predict(
    points, 
    labels,
    batch_size=32,
    num_batches = 100, 
    layer_sizes=(2, 512, 512, 1),
):
    mlp = make_classifier_module(*layer_sizes, device="cuda")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    indices = torch.arange(batch_size*num_batches)%(points.shape[0])
    loss_list = []
    for i in range(0, batch_size*num_batches, batch_size):
        optimizer.zero_grad()
        batch_indices = indices[i:i+batch_size]
        preds = mlp(points[batch_indices, ...])
        loss = binary_cross_entropy_with_logits(preds, labels[batch_indices, ...])
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().requires_grad_(False))

    losses = torch.stack(loss_list).cpu()

    fig_loss, ax_loss = plt.subplots() 
    #ax_loss.set_title("Cross entropy loss vs epochs", weight="bold") 
    ax_loss.set_xlabel("optimizer steps", weight="bold") 
    ax_loss.set_ylabel("cross entropy loss", weight="bold")
    ax_loss.plot(torch.arange(num_batches), losses, "-", label="loss")
    ax_loss.legend()
    fig_loss.savefig("loss_single_model.pdf")

    return single_predict_on_mesh(mlp, device="cuda")

def single_predict_on_mesh(model, device="cuda", width=2., steps=50):
    with torch.inference_mode():
        xs = torch.linspace(-width, width, steps=steps, device=device)
        ys = torch.linspace(-width, width, steps=steps, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        points = torch.stack([xx.ravel(), yy.ravel()], dim=1)

        z = sigmoid(model(points)).reshape_as(xx)
        return xx.detach().cpu(), yy.detach().cpu(), z.detach().cpu()
 
def parallel_mlp_fit_predict(
    points,
    labels,
    layer_sizes = (2, 512, 512, 1),
    num_replicas = 100,
    batch_size = 32,
    num_batches = 100,
    device="cuda",
    do_bootstrap = False,
):
   
    ensemble = Ensemble(
        make_classifier_module,
        criterion=binary_cross_entropy_with_logits,
        num_replicas=num_replicas,
        model_init_args=layer_sizes,
        device = device,
        model_init_randomness="different",
    )

    print("Initialized ensemble for bootstrapping ...")
    
    optimizer = Adam(lr=1e-3, batch_size=(num_replicas,), device=device)
    optimizer.init(ensemble._params_dict)
    ensemble.train(True)

    data_iterator = parallel_batch_iterator(
        points, labels, 
        num_replicas=num_replicas, 
        batch_size=batch_size, 
        num_batches=num_batches,
        do_bootstrap=do_bootstrap,
    )

    losses = ensemble.fit_step(
        optimizer, 
        data_iterator, 
        in_dims=0 if do_bootstrap else (0, 0, None, None),
    ).detach_()
    print(f"Training ensemble {"with" if do_bootstrap else "without"} bootstrapping done ...")

    dy, y = torch.std_mean(losses.cpu(), dim=0)
    x = torch.arange(num_batches)
    
    fig0, ax0 = plt.subplots() 
    #ax0.set_title("Cross entropy loss vs # minibatch iterations", weight="bold") 
    ax0.set_xlabel("optimizer steps", weight="bold") 
    ax0.set_ylabel("cross entropy loss", weight="bold")
    ax0.plot(x, y, "-", label="loss")
    ax0.fill_between(x, y-dy, y+dy, alpha=0.2, label=r"$\Delta$(loss)")
    ax0.legend()
    fig0.savefig("loss_parallel.pdf" if do_bootstrap else "loss_parallel_no_bootstrap.pdf")

    return parallel_predict_on_mesh(ensemble)

def make_classifier_module(*layer_sizes, device = "cpu", dtype = torch.float32):
    layer_sizes, output_size = layer_sizes[:-1], layer_sizes[-1]
    module = Sequential()
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        module.extend(
            Sequential(
                Linear(in_size, out_size), 
                ReLU(),
            )
        )
    module.append(Linear(layer_sizes[-1], output_size))
    #module.append(LogSoftmax(dim=1))
    return module.to(device=device, dtype=dtype)

def parallel_batch_iterator(
    X, y, *, 
    num_replicas = 100,
    batch_size = 32,
    num_batches = 100,
    do_bootstrap = False,
):
    total_samples = batch_size*num_batches

    if do_bootstrap:
        indices = vmap(
            make_batched_indices, 
            randomness="different"
        )(
            torch.arange(num_replicas), 
            dataset_size=X.shape[0], 
            sample_size=(total_samples,),
        )

        print(indices.shape)
        for i in range(0, total_samples, batch_size):
            batch_indices = indices[:, i:i+batch_size]
            yield (
                X[batch_indices.ravel()].reshape(*(batch_indices.shape), *(X.shape[1:])), 
                y[batch_indices.ravel()].reshape(*(batch_indices.shape), 1), 
                None,
            )

    else:
        indices = torch.arange(total_samples)%(X.shape[0])
        print(indices.shape)
        for i in range(0, total_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield (X[batch_indices], y[batch_indices], None,)

    
def make_batched_indices(seed, *, dataset_size, sample_size=()):
    dist = Categorical(torch.ones(dataset_size))
    return dist.sample(sample_size)


def parallel_predict_on_mesh(ensemble, width=2, steps=50):
    with torch.inference_mode():
        xs = torch.linspace(-width, width, steps=steps, device=ensemble.device)
        ys = torch.linspace(-width, width, steps=steps, device=ensemble.device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    
        points = torch.stack([xx.ravel(), yy.ravel()], dim=1)

        def predict_fn(model, params, buffers, inputs):
            return sigmoid(functional_call(model, (params, buffers), inputs))

        fpred = vmap(
            partial(predict_fn, ensemble._base_model),
            in_dims=(0, 0, None)
        )
        z = fpred(ensemble._params_dict, ensemble._buffers_dict, points)

        z_mean = z.mean(dim=0).reshape_as(xx)

        return xx.detach().cpu(), yy.detach().cpu(), z_mean.detach().cpu()
 
def make_spirals(n_samples, noise_std=0., rotations=1.):
    ts = torch.linspace(0, 1, n_samples)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * pi
    signs = torch.randint(0, 2, (n_samples,)) * 2 - 1
    labels = (signs > 0).to(torch.int8)

    xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
    ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
    points = torch.stack([xs, ys], dim=1)
    return points, labels.unsqueeze_(-1)


   
def plot_predictions(ax, xx, yy, z):
    return ax.imshow(
        z, 
        extent=(
            xx.min(), xx.max(), 
            yy.min(), yy.max(),
        ), 
        origin="lower", 
        cmap=LinearSegmentedColormap.from_list(
            "blueorange", 
            ["#014182", "white", "#c45508"],
            #["tab:blue", "white", "tab:orange"],
        ), 
        vmin=0, vmax=1, 
        aspect="equal",
    )


def plot_spirals(ax, points, labels):
    sc = ax.scatter(
        points[:, 0], 
        points[:, 1], 
        c=labels,
        cmap = ListedColormap([
            #"xkcd:darkblue", "xkcd:orangered", 
            "tab:blue", "tab:orange",
        ]),
        edgecolors = "white",
    )
    lim = 2.0 
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("x", weight="bold")
    ax.set_ylabel("y", weight="bold")
    ax.legend(
        sc.legend_elements()[0], 
        ["0", "1"], 
        title = "label",
        title_fontproperties = FontProperties(weight="bold"), 
        loc="lower right",
        alignment = "center",
    )

    return ax


if __name__ == "__main__":
    device = "cuda"
    num_replicas = 100
    num_samples = 100
    batch_size = 32 
    num_batches = 100
    do_bootstrap = True  
    show_plot = True

    torch.manual_seed(0)
    
    points, labels = make_spirals(num_samples, noise_std=0.05)
    fig_0 = plt.figure(figsize=(5, 5))
    ax_0 = fig_0.add_subplot()
    ax_0 = plot_spirals(ax_0, points, labels)
    fig_0.savefig("data.pdf")

    fig = plt.figure()
    ax = fig.add_subplot()
    
    if num_replicas <= 1:
        xx, yy, z = single_mlp_fit_predict(
            points.to(device=device), labels.to(device=device, dtype=torch.float32),
            num_batches=num_batches,
            batch_size=batch_size,
        )
        ax.set_title(f"Predictions from single MLP", weight="bold", y=0.95, pad = 30)
        ax = plot_spirals(ax, points, labels)
        im = plot_predictions(ax, xx, yy, z)
        fig.colorbar(im, ax=ax, label="prediction")
        fig.savefig("predictions_single.pdf")

    else:
        xx, yy, z = parallel_mlp_fit_predict(
            points, labels, 
            num_replicas=num_replicas,
            num_batches=num_batches,
            batch_size=batch_size,
            device=device, 
            do_bootstrap=do_bootstrap
        )
        if do_bootstrap:
            ax.set_title(f"Predictions from bootstrap ensemble ({num_replicas} models)", weight="bold", y=0.95, pad = 30)
        else:
            ax.set_title(f"Predictions from ensemble ({num_replicas} models) on full dataset", weight="bold", y=0.95, pad = 30)
        ax = plot_spirals(ax, points, labels)
        im = plot_predictions(ax, xx, yy, z)
        fig.colorbar(im, ax=ax, label="mean prediction")
        fig.savefig("predictions_parallel.pdf" if do_bootstrap else "predictions_parallel_no_bootstrap.pdf")

    if show_plot:
        try:
            plt.show()
        except KeyboardInterrupt:
            exit()



