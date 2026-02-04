import torch.nn as nn
from torchvision import models


def load_predefined_model(name, num_classes):
    if name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)

    elif name == "custom":
        model = CustomCNN([32, 64, 128], num_classes)

    else:
        raise ValueError("Unknown model architecture")

    return model


import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, filters, num_classes):
        super().__init__()
        layers = []
        in_c = 3

        for f in filters:
            layers.append(nn.Conv2d(in_c, f, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_c = f

        self.conv = nn.Sequential(*layers)

        # ✅ adaptive pooling fixes shape problem
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[-1], num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

