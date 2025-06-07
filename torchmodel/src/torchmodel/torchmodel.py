from typing import Iterable, Mapping, Protocol, Optional, TypeVar, List, Union, Tuple
from collections.abc import Callable, Hashable, Sequence, Iterable
from functools import singledispatchmethod, singledispatch
from typing_extensions import runtime_checkable

import numpy as np
from numpy import typing as npt
from tqdm import tqdm

import torch
from tensordict import TensorDict
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric


from . import callbacks

@runtime_checkable
class _GenericTensorOp(Protocol):
    def __call__(self, *inputs : torch.Tensor) -> torch.Tensor: ...

_Loss = TypeVar("_Loss", torch.nn.Module, _GenericTensorOp)
_Activation = TypeVar("_Activation", torch.nn.Module, Callable[[torch.Tensor], torch.Tensor], _GenericTensorOp)

type _Metric0 = Union[Metric, MetricCollection]
type _Metric = Union[_Metric0, _GenericTensorOp]

type _Metrics0 = Union[Sequence[_Metric], dict[str, _Metric]]
type _Metrics = Union[MetricCollection, Sequence[_Metric], dict[str, _Metric]]

class MetricFunctionWrapper(Metric):
    def __init__(self, fx : _GenericTensorOp, **kwargs):
        self.add_state("agg_metric", default=torch.Tensor(0), dist_reduce_fx="sum")
        self.add_state("n_samples",default=torch.Tensor(0), dist_reduce_fx="sum")
        self.fx : _GenericTensorOp = fx 

    def update(self, preds : torch.Tensor, targets : torch.Tensor) -> None:
        batch_metric = self.fx(preds, targets)
        self.agg_metric += torch.sum(batch_metric)
        self.n_samples += targets.numel()

    def compute(self) -> torch.Tensor:
        return torch.as_tensor(self.agg_metric, dtype=torch.float32).div_(self.n_samples)

def to_torchmetric(maybe_metric : _Metric | _Metrics0, device : Optional[torch.device | str] = None, **kwargs)->MetricCollection:
    res = None
    if isinstance(maybe_metric, (Metric, MetricCollection)):
        res = MetricCollection(maybe_metric, **kwargs)

    elif isinstance(maybe_metric, _GenericTensorOp):
        res = to_torchmetric(MetricFunctionWrapper(maybe_metric, device=device), **kwargs)

    elif isinstance(maybe_metric, Sequence):
        res = MetricCollection([to_torchmetric(metric, device=device) for metric in maybe_metric], **kwargs)

    elif isinstance(maybe_metric, Mapping):
        res = MetricCollection({key : to_torchmetric(val, device=device) for key, val in maybe_metric.items()}, **kwargs)

    if res is not None:
        return res.to(device=device) if device else res

    raise TypeError(f"Can't convert object of type {type(maybe_metric)} into torchmetrics.MetricCollection!")

class TensorDictModel:
    def __init__(self, model: torch.nn.Module, in_keys:List, out_key:str, device: Optional[torch.device] = None, name : str ="", do_debug : bool =False):
        super().__init__()
        self.do_debug = do_debug
        self.model = model
        self.in_keys=in_keys
        self.out_key = out_key
        self.to_device(device)
        self.name = name 
        if self.do_debug:
            print("------Initializing model: ", self.name)

    def print_debug_tensordict(self, tendict : TensorDict, info : str):
        print(f"Full tensordict, {info}:")
        print(tendict)
        for key in tendict.keys(include_nested=True, leaves_only=True):
            print(f"------Key {key}:")
            print(tendict[key])


    def to_device(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device
        self.model.to(self.device)

    def compile(
        self,
        compile_mode: Optional[str] = None,
        criterion: Optional[_Loss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        compile_backend: Optional[str] = None,
    ):
        if isinstance(criterion, torch.nn.Module):
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if compile_mode is not None:
            self.compile_mode = compile_mode
            self.comple_backend = (
                compile_backend if compile_backend is not None else "inductor"
            )
            if self.compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]:
                self.model.compile(mode=self.compile_mode, backend=self.comple_backend)
            else:
                raise ValueError(
                    f"compile_mode must be one of 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs', but got {compile_mode}!"
                )

    def __call__(self):
        return self.model

    def save(self, checkpoint):
        state_dict_ = {}
        state_dict_["model_state_dict"] = self.model.state_dict()
        if self.optimizer is not None:
            state_dict_["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict_["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state_dict_, checkpoint)
        self.last_checkpoint = checkpoint

    def get_last_checkpoint(self):
        return self.last_checkpoint

    def load(self, checkpoint, **torch_load_kwargs):
        checkpoint_dict = torch.load(checkpoint, **torch_load_kwargs)
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.model.to(self.device)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
        self.last_checkpoint = checkpoint
        return checkpoint_dict

    def load_last_checkpoint(self, **torch_load_kwargs):
        return self.load(self.last_checkpoint, **torch_load_kwargs)

    def apply(
        self, batch: TensorDict, verbose: bool = False
    ) -> torch.Tensor:
        inputs = []
        for key in self.in_keys:
            if verbose:
                print(f"Applying key: {key}")
                print(batch.get(key))
            inputs.append(batch.get(key))
        return self.model(*inputs)

    def predict(
        self,
        loader: torch.utils.data.DataLoader,
        out_activation: Optional[_Activation] = None,
    ) -> torch.Tensor:
        outputs = []
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(loader), desc="Predicting...", unit="batch", leave=False) as pbar:
                for ibatch, batch in enumerate(loader):
                    inputs = batch.data.to(self.device)
                    outputs.append(self.apply(inputs))
                    pbar.update()

                    if self.do_debug: 
                        self.print_debug_tensordict(inputs, info=f"{self.name}, prediction")
                        break

        if out_activation is None:
            return torch.cat(outputs)

        return out_activation(torch.cat(outputs)).squeeze_()

    def evaluate(
        self, batch: TensorDict, metrics: Optional[MetricCollection] = None, sample_weights:Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        assert self.criterion is not None
        outputs = self.apply(batch)
        targets = batch[self.out_key]

        if metrics is not None:
            metrics.update(outputs, targets)

        #return torch.dot(sample_weights.squeeze_(), self.criterion(outputs, targets).squeeze_()).div_(sample_weights.sum())
        if sample_weights is not None:
            return torch.dot(sample_weights, self.criterion(outputs, targets).squeeze_()).div_(targets.shape[0])
        else:
            return torch.dot(batch["sample_weights"].squeeze_(), self.criterion(outputs, targets).squeeze_()).div_(targets.shape[0])

    def train_epoch(
        self, loader: torch.utils.data.DataLoader, sample_weights:Optional[npt.ArrayLike]=None, tqdm_desc : str = ""
    ):
        assert self.optimizer is not None
        self.model.train()
        epoch_loss = 0
        nbatches = len(loader)
        # nsamples = 0
        if self.do_debug:
            print("------starting training loop for model,", self.name)

        with tqdm(total = nbatches, desc=tqdm_desc, unit="batch", leave=False) as pbar:
            for ibatch, batch in enumerate(loader): 
                input = batch.data.to(self.device)
                indices = batch.indices.to(self.device)

                if sample_weights is not None:
                    #print(sample_weights, len(sample_weights))
                    #print(indices, len(indices), torch.max(indices))
                    sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)

                #print(input)
                loss = self.evaluate(input, metrics=self.train_metrics, sample_weights=sample_weights[indices])
                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                self.optimizer.step()

                epoch_loss = (epoch_loss*ibatch + loss.item())/(ibatch+1)
                pbar.set_postfix(loss = epoch_loss)
                pbar.update()
                
                if self.do_debug:
                    self.print_debug_tensordict(input, info=f"{self.name}, train")
                    print("Loss:", epoch_loss)
                    break

        epoch_metrics = {}
        if self.train_metrics is not None:
            epoch_metrics = self.train_metrics.compute()
            self.train_metrics.reset()

        return epoch_loss, epoch_metrics

    def validate_epoch(
        self, loader: torch.utils.data.DataLoader, sample_weights:Optional[npt.ArrayLike]=None, tqdm_desc : str = ""
    ):
        self.model.eval()
        epoch_loss = 0
        nbatches = len(loader)
        with torch.no_grad():
            with tqdm(total=nbatches, desc=tqdm_desc, unit="batch", leave=False) as pbar:
                for ibatch, batch in enumerate(loader):
                    input = batch.data.to(self.device)
                    indices = batch.indices.to(self.device)

                    if sample_weights is not None:
                        sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)
                        #sample_weights = torch.as_tensor(sample_weights[indices.cpu().detatch().numpy()], dtype=torch.float32, device=self.device)

                    #print(input)
                    loss = self.evaluate(input, metrics=self.val_metrics, sample_weights=sample_weights[indices])

                    #batch_loss = loss.item()
                    epoch_loss = (epoch_loss*ibatch + loss.item())/(ibatch+1)
                    pbar.set_postfix(loss = epoch_loss)
                    pbar.update()

                    if self.do_debug: 
                        self.print_debug_tensordict(input, info=f"{self.name}, validation")
                        print("Loss:", epoch_loss)
                        break
        
        epoch_metrics = {}
        if self.val_metrics is not None:
            epoch_metrics = self.val_metrics.compute()
            self.val_metrics.reset()

        return epoch_loss, epoch_metrics

    def process_metrics(self, metrics : Optional[_Metric | _Metrics0] = None)->None:
        self.train_metrics = None
        self.val_metrics = None
        if metrics is not None:
            self.train_metrics = to_torchmetric(metrics, prefix="train_", device=self.device)
            self.val_metrics = self.train_metrics.clone(prefix="val_").to(device=self.device)


    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        sample_weights : Optional[npt.ArrayLike] = None, 
        early_stopping: Optional[callbacks.EarlyStopping] = None,
        metrics: Optional[_Metrics] = None,
        tqdm_desc : str = "--->Epoch:"
    ):
        self.process_metrics(metrics)
        history = {}

        history["train_loss"] = []
        if self.train_metrics is not None:
            for key in self.train_metrics.keys():
                history[key] = []

        history["val_loss"] = []
        if self.val_metrics is not None:
            for key in self.val_metrics.keys():
                history[key] = []

        self.last_checkpoint = None
        
        with tqdm(total=epochs, unit="epoch", leave=False, desc=tqdm_desc, dynamic_ncols=True, bar_format="{desc}:[{n_fmt}/{total_fmt}] | {bar} | {rate_fmt}{postfix}") as pbar:
            for epoch in range(epochs):
                train_loss, train_metrics = self.train_epoch(train_loader, sample_weights=sample_weights, tqdm_desc=f"------>Training")
                val_loss, val_metrics = self.validate_epoch(val_loader, sample_weights=sample_weights, tqdm_desc=f"------>Validation")
            
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                for key, val in dict(**train_metrics, **val_metrics).items():
                    history[key].append(val.item())

                #msg = f"{tqdm_desc} | (train_loss = {train_loss}, val_loss = {val_loss})"
                #for train_key, val_key in zip(train_metrics.keys(), val_metrics.keys()):
                #    msg = f"{msg} | ({train_key} = {history[train_key][-1]}, {val_key} = {history[val_key][-1]})"

                pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
                #pbar.write(msg, end="\r")
            
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                if early_stopping is not None:
                    do_early_stopping = early_stopping(epoch, val_loss, self.model, self.optimizer, self.scheduler)
                    pbar.set_postfix(best_epoch=early_stopping.bestEpoch, best_loss=early_stopping.bestLoss, patience=f"{early_stopping.counter}/{early_stopping.patience}")
                    if do_early_stopping:
                        pbar.update()
                        break
                 
                pbar.update()
        if early_stopping is not None:
            checkpoint_dict = self.load(early_stopping.getLastCheckpoint(), weights_only=True)
            best_epoch = checkpoint_dict['epoch']
            print(f"------>Best model from epoch {best_epoch} with validation loss {checkpoint_dict['val_loss']}, metrics: ")
            print("------>train_loss =", history["train_loss"][best_epoch], ", val loss =", history["val_loss"][best_epoch])
            if self.train_metrics is not None and self.val_metrics is not None:
                for train_key, val_key in zip(self.train_metrics.keys(), self.val_metrics.keys()):
                    print(f"------>{train_key} =", history[train_key][best_epoch], ",", val_key, "=", history[val_key][best_epoch])


        return history
