import torch
import torch.distributed as dist
import torch_optimizer as optim
import torch.nn as nn
import numpy as np
from .transformer import RMSNorm


class MovingBuffer:
    def __init__(self, initValue=None, maxLen=None):
        from collections import deque

        self.values = deque(maxlen=maxLen)
        if initValue is not None:
            self.step(initValue)

    def step(self, value):
        self.values.append(value)

    def getQuantile(self, quantile):
        return float(np.quantile(self.values, q=quantile))


def checkNoneGradient(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print("Warning: detected parameter with no gradient that requires gradient:")
            print(param)
            print(param.shape)
            print(name)


def average_gradients(model, c=None, parallel=True):
    if parallel:
        size = float(dist.get_world_size())
        if c is None:
            c = size

        # size = float(dist.get_world_size())
        checkNoneGradient(model)
        for param in model.parameters():
            if param.requires_grad:
                # print(param)
                # print(param.shape)
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # param.grad.data /= c
    else:
        checkNoneGradient(model)
        if c is None:
            c = 1
        for param in model.parameters():
            if param.requires_grad:
                param.grad.data /= c


def load_state_dict_tolerant(model, state_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def save_checkpoint(filename, epoch, nIter, model, lossTracker, best_state_dict, optimizer, lrScheduler):
    checkpoint = {
        #'conf':model.conf.__dict__,
        "state_dict": model.state_dict(),
        "best_state_dict": best_state_dict,
        "epoch": epoch,
        "nIter": nIter,
        "loss_tracker": lossTracker,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lrScheduler.state_dict(),
    }
    torch.save(checkpoint, filename)


def getOptimizerGroup(model):
    param_optimizer = list(model.named_parameters())
    # exclude GroupNorm and PositionEmbedding from weight decay
    # no_decay = ['bias', 'LayerNorm', 'GroupNorm, PositionEmbedding']  # Specify the names of parameters to exclude from weight decay
    # optimizerConfig = [
    # # Parameters with weight decay
    # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    # # Parameters without weight decay
    # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # print([n for n, p in param_optimizer if any(nd in n for nd in no_decay)])

    noDecay = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm) or isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            noDecay.extend(list(module.parameters()))
        else:
            noDecay.extend([p for n, p in module.named_parameters() if "bias" in n])

    otherParams = set(model.parameters()) - set(noDecay)
    otherParams = [param for param in model.parameters() if param in otherParams]
    noDecay = set(noDecay)
    noDecay = [param for param in model.parameters() if param in noDecay]

    optimizerConfig = [
        {"params": otherParams},
        {"params": noDecay, "weight_decay": 0e-7},
    ]

    return optimizerConfig


def initializeCheckpoint(Model, device, max_lr, weight_decay, nIter, conf):
    model = Model(conf).to(device)

    optimizerGroup = getOptimizerGroup(model)

    optimizer = optim.AdaBelief(
        optimizerGroup,
        max_lr,
        weight_decouple=True,
        eps=1e-8,
        weight_decay=weight_decay,
        rectify=True,
    )

    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr,
        nIter,
        pct_start=0.05,
        cycle_momentum=False,
        final_div_factor=2,
        div_factor=20,
    )

    lossTracker = {"train": [], "val": []}
    best_state_dict = None
    startEpoch = 0
    startIter = 0

    return (
        startEpoch,
        startIter,
        model,
        lossTracker,
        best_state_dict,
        optimizer,
        lrScheduler,
    )


def load_checkpoint(Model, conf, filename, device, strict=False):
    checkpoint = torch.load(filename, map_location=device)

    startEpoch = checkpoint.get("epoch", 0)
    startIter = checkpoint.get("nIter", 0)

    model = Model(conf=conf).to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    optimizerGroup = getOptimizerGroup(model)

    optimizer = optim.AdaBelief(
        optimizerGroup,
        1e-5,
        weight_decouple=True,
        eps=1e-8,
        weight_decay=1e-2,
        rectify=True,
    )

    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        4e-4,
        500000,
        pct_start=0.05,
        cycle_momentum=False,
        final_div_factor=2,
        div_factor=20,
    )

    # debugging flag
    restartFromTheBest = False

    if restartFromTheBest:
        if not strict:
            load_state_dict_tolerant(model, checkpoint["best_state_dict"])
        else:
            model.load_state_dict(checkpoint["best_state_dict"])
    else:
        if not strict:
            load_state_dict_tolerant(model, checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "lr_scheduler_state_dict" in checkpoint:
            lrScheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if "best_state_dict" in checkpoint:
        best_state_dict = checkpoint["best_state_dict"]
    else:
        best_state_dict = None

    if "loss_tracker" in checkpoint:
        lossTracker = checkpoint["loss_tracker"]
    else:
        lossTracker = {"train": [], "val": []}

    return (
        startEpoch,
        startIter,
        model,
        lossTracker,
        best_state_dict,
        optimizer,
        lrScheduler,
    )