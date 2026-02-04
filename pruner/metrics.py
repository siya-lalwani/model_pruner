import torch
import pandas as pd
import os, time
from torchprofile import profile_macs


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)


def compute_sparsity(model):
    zero = 0
    total = 0
    for p in model.parameters():
        zero += torch.sum(p == 0).item()
        total += p.numel()
    return (zero / total) * 100


def inference_time(model):
    model.eval()
    dummy = torch.randn(1,3,224,224)

    start = time.time()
    with torch.no_grad():
        model(dummy)
    end = time.time()

    return (end - start) * 1000


def compute_flops(model):
    dummy = torch.randn(1,3,224,224)
    macs = profile_macs(model, dummy)
    return macs / 1e9


def generate_csv(orig_model, pruned_model, orig_acc, pruned_acc, save_path):

    os.makedirs(save_path, exist_ok=True)

    orig_path = os.path.join(save_path, "orig_tmp.pth")
    pruned_path = os.path.join(save_path, "pruned_tmp.pth")

    torch.save(orig_model.state_dict(), orig_path)
    torch.save(pruned_model.state_dict(), pruned_path)

    orig_params = count_params(orig_model)
    pruned_params = count_params(pruned_model)

    orig_size = model_size_mb(orig_path)
    pruned_size = model_size_mb(pruned_path)

    orig_flops = compute_flops(orig_model)
    pruned_flops = compute_flops(pruned_model)

    orig_time = inference_time(orig_model)
    pruned_time = inference_time(pruned_model)

    sparsity = compute_sparsity(pruned_model)

    compression_ratio = orig_params / pruned_params

    data = {
        "Metric": [
            "Accuracy (%)",
            "Parameters",
            "Model Size (MB)",
            "FLOPs (GFLOPs)",
            "Inference Time (ms)",
            "Sparsity (%)",
            "Compression Ratio"
        ],
        "Original": [
            round(orig_acc,2),
            orig_params,
            round(orig_size,2),
            round(orig_flops,2),
            round(orig_time,2),
            0,
            1
        ],
        "Pruned": [
            round(pruned_acc,2),
            pruned_params,
            round(pruned_size,2),
            round(pruned_flops,2),
            round(pruned_time,2),
            round(sparsity,2),
            round(compression_ratio,2)
        ]
    }

    df = pd.DataFrame(data)
    csv_path = os.path.join(save_path, "metrics.csv")
    df.to_csv(csv_path, index=False)

    return csv_path
