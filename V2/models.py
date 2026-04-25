import torch
import torch.nn as nn
import torchvision

def load_model(model_path, device):
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model