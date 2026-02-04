from django import forms

PRUNING_CHOICES = [
    ("l1", "L1 Structured"),
    ("l2", "L2 Structured"),
    ("global", "Global Unstructured"),
    ("random", "Random Pruning"),
    ("taylor", "Taylor Pruning"),
    ("gm", "GM Pruning"),
]

LAYER_CHOICES = [
    ("all", "All Layers"),
    ("conv", "Conv Layers Only"),
    ("fc", "FC Layers Only"),
]

DATASET_CHOICES = [
    ("mnist", "MNIST"),
    ("cifar10", "CIFAR-10"),
]


class UploadModelForm(forms.Form):
    model_file = forms.FileField()
    architecture = forms.ChoiceField(choices=[
        ("vgg16", "VGG16"),
        ("resnet18", "ResNet18"),
        ("custom", "Custom CNN"),
    ])
    num_classes = forms.IntegerField()
    dataset = forms.ChoiceField(choices=DATASET_CHOICES)
    pruning_type = forms.ChoiceField(choices=PRUNING_CHOICES)
    layer_type = forms.ChoiceField(choices=LAYER_CHOICES)
    sparsity = forms.FloatField()
    epochs = forms.IntegerField(initial=1)
    batch_size = forms.IntegerField(initial=32)


class ModelSpecForm(forms.Form):
    filters = forms.CharField(help_text="e.g. 32,64,128")
    num_classes = forms.IntegerField()
    dataset = forms.ChoiceField(choices=DATASET_CHOICES)
    pruning_type = forms.ChoiceField(choices=PRUNING_CHOICES)
    layer_type = forms.ChoiceField(choices=LAYER_CHOICES)
    sparsity = forms.FloatField()
    epochs = forms.IntegerField(initial=1)
    batch_size = forms.IntegerField(initial=32)
