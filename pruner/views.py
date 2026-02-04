import threading
import os
import torch

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

from .forms import UploadModelForm, ModelSpecForm
from .model_factory import load_predefined_model, CustomCNN
from .models import PruningJob
from .tasks import run_pruning_job


def home(request):
    return render(request, "home.html", {
        "upload_form": UploadModelForm(),
        "spec_form": ModelSpecForm()
    })


def prune_model_view(request):
    if request.method != "POST":
        return HttpResponse("Invalid request method")

    form_type = request.POST.get("form_type")

    # -------- Upload model --------
    if form_type == "upload":
        form = UploadModelForm(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponse("Upload form invalid")

        model_file = request.FILES["model_file"]
        arch = form.cleaned_data["architecture"]
        num_classes = form.cleaned_data["num_classes"]
        dataset = form.cleaned_data["dataset"]
        pruning_type = form.cleaned_data["pruning_type"]
        layer_type = form.cleaned_data["layer_type"]
        sparsity = form.cleaned_data["sparsity"]
        epochs = form.cleaned_data["epochs"]
        batch_size = form.cleaned_data["batch_size"]

        model = load_predefined_model(arch, num_classes)

        os.makedirs("pruner/media/uploads", exist_ok=True)
        path = "pruner/media/uploads/model.pth"

        with open(path, "wb+") as f:
            for chunk in model_file.chunks():
                f.write(chunk)

        model.load_state_dict(torch.load(path, map_location="cpu"))

    # -------- Model spec --------
    elif form_type == "spec":
        form = ModelSpecForm(request.POST)
        if not form.is_valid():
            return HttpResponse("Spec form invalid")

        filters = list(map(int, form.cleaned_data["filters"].split(",")))
        num_classes = form.cleaned_data["num_classes"]
        dataset = form.cleaned_data["dataset"]
        pruning_type = form.cleaned_data["pruning_type"]
        layer_type = form.cleaned_data["layer_type"]
        sparsity = form.cleaned_data["sparsity"]
        epochs = form.cleaned_data["epochs"]
        batch_size = form.cleaned_data["batch_size"]

        model = CustomCNN(filters, num_classes)

    else:
        return HttpResponse("Invalid form type")

    # -------- Create job --------
    job = PruningJob.objects.create(
        pruning_type=pruning_type,
        sparsity=sparsity,
        status="PENDING",
        progress=0
    )

    # -------- Run in background --------
    thread = threading.Thread(
        target=run_pruning_job,
        args=(job.id, model, dataset, pruning_type,
              sparsity, layer_type, epochs, batch_size),
        daemon=True
    )
    thread.start()

    return render(request, "progress.html", {"job_id": job.id})


def job_status(request, job_id):
    job = PruningJob.objects.get(id=job_id)
    return JsonResponse({
        "status": job.status,
        "progress": job.progress
    })


def result_view(request):
    return render(request, "result.html", {
        "model_path": "/media/outputs/pruned_model.pth",
        "csv_path": "/media/outputs/metrics.csv"
    })
