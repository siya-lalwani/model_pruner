from django.db import models

class PruningJob(models.Model):
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("RUNNING", "Running"),
        ("DONE", "Done"),
        ("FAILED", "Failed"),
    ]

    pruning_type = models.CharField(max_length=50)
    sparsity = models.FloatField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    progress = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
