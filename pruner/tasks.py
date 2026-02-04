import torch
import copy
import os
import time

from .models import PruningJob
from .pruning_engine import apply_pruning
from .trainer import train_model, evaluate_model
from .metrics import generate_csv
from .dataset_loader import get_dataloaders


def run_pruning_job(job_id, model, dataset, pruning_type,
                    sparsity, layer_type, epochs, batch_size):

    print("✅ Job started:", job_id)

    job = PruningJob.objects.get(id=job_id)
    job.status = "RUNNING"
    job.progress = 5
    job.save()

    try:
        # ---------------- Dataset ----------------
        print("📦 Loading dataset:", dataset)
        train_loader, test_loader = get_dataloaders(dataset, batch_size)

        job.progress = 15
        job.save()

        # ---------------- Train original ----------------
        print("🏋️ Training original model")
        orig_model = copy.deepcopy(model)
        orig_model = train_model(orig_model, train_loader, epochs)
        orig_acc = evaluate_model(orig_model, test_loader)

        job.progress = 40
        job.save()

        # ---------------- Pruning ----------------
        print("✂️ Applying pruning:", pruning_type)

        pruned_model = apply_pruning(
            model,
            pruning_type,
            sparsity,
            layer_type,
            train_loader
        )

        job.progress = 60
        job.save()

        # ---------------- Train pruned ----------------
        print("🏋️ Training pruned model")
        pruned_model = train_model(pruned_model, train_loader, epochs)
        pruned_acc = evaluate_model(pruned_model, test_loader)

        job.progress = 80
        job.save()

        # ---------------- Save outputs ----------------
        os.makedirs("pruner/media/outputs", exist_ok=True)

        pruned_path = "pruner/media/outputs/pruned_model.pth"
        torch.save(pruned_model.state_dict(), pruned_path)

        print("💾 Saved pruned model")

        generate_csv(
            orig_model,
            pruned_model,
            orig_acc,
            pruned_acc,
            "pruner/media/outputs"
        )

        print("📊 Metrics CSV generated")

        job.progress = 100
        job.status = "DONE"
        job.save()

        print("✅ Job completed:", job_id)

    except Exception as e:
        print("❌ Job failed:", e)
        job.status = "FAILED"
        job.progress = 0
        job.save()
