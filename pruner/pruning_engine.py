import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np


def apply_pruning(model, method, sparsity, layer_type="all",
                  train_loader=None, device="cpu"):

    if method in ["l1", "l2", "global", "random"]:
        return magnitude_pruning(model, method, sparsity, layer_type)

    elif method == "taylor":
        return taylor_pruning(model, sparsity, train_loader, device)

    elif method == "gm":
        return gm_pruning(model, sparsity)

    else:
        raise ValueError("Unknown pruning method")


# ---------------- BASIC METHODS ---------------- #

def magnitude_pruning(model, method, sparsity, layer_type):

    for name, module in model.named_modules():

        if layer_type == "conv" and not isinstance(module, nn.Conv2d):
            continue
        if layer_type == "fc" and not isinstance(module, nn.Linear):
            continue

        if hasattr(module, "weight"):

            if method == "l1":
                prune.ln_structured(module, "weight", sparsity, n=1, dim=0)

            elif method == "l2":
                prune.ln_structured(module, "weight", sparsity, n=2, dim=0)

            elif method == "global":
                prune.l1_unstructured(module, "weight", sparsity)

            elif method == "random":
                prune.random_unstructured(module, "weight", sparsity)

    remove_pruning(model)
    return model


# ---------------- TAYLOR PRUNING ---------------- #

def taylor_pruning(model, sparsity, train_loader, device):

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    importance = {}

    # register hooks
    def hook_fn(module, grad_input, grad_output):
        importance[module] = torch.abs(module.weight.grad * module.weight).mean(dim=(1,2,3))

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_backward_hook(hook_fn))

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    for h in hooks:
        h.remove()

    # prune lowest importance filters
    for module, scores in importance.items():
        num_prune = int(len(scores) * sparsity)
        idx = torch.argsort(scores)[:num_prune]

        mask = torch.ones_like(scores)
        mask[idx] = 0
        prune.custom_from_mask(module, "weight",
                                mask[:, None, None, None])

    remove_pruning(model)
    return model


# ---------------- GM PRUNING ---------------- #

def gm_pruning(model, sparsity):

    for module in model.modules():
        if isinstance(module, nn.Conv2d):

            W = module.weight.data.cpu().numpy()
            filters = W.reshape(W.shape[0], -1)

            center = np.mean(filters, axis=0)
            distances = np.linalg.norm(filters - center, axis=1)

            num_prune = int(len(distances) * sparsity)
            idx = np.argsort(distances)[:num_prune]

            mask = torch.ones(module.weight.shape[0])
            mask[idx] = 0

            prune.custom_from_mask(module, "weight",
                mask[:, None, None, None])

    remove_pruning(model)
    return model


# ---------------- UTIL ---------------- #

def remove_pruning(model):
    for module in model.modules():
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
