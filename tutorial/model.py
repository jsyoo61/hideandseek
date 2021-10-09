import logging

import torch
import torch.nn as nn
import torchvision

log = logging.getLogger(__name__)

# %%
def Resnet(input_shape=None, pretrained=False, progress=True, version=18, in_channels=3, n_classes=10):
    log.info(f'input_shape: {input_shape}')
    allowed_version = ['18', '34', '50', '101', '152']
    assert str(version) in allowed_version, f'version must be one of: {allowed_version}'
    if version == 18:
        model = torchvision.models.resnet18(pretrained=pretrained, progress=progress)
    elif version == 34:
        model = torchvision.models.resnet34(pretrained=pretrained, progress=progress)
    elif version == 50:
        model = torchvision.models.resnet50(pretrained=pretrained, progress=progress)
    elif version == 101:
        model = torchvision.models.resnet101(pretrained=pretrained, progress=progress)
    elif version == 152:
        model = torchvision.models.resnet152(pretrained=pretrained, progress=progress)
    # model = torch.hub.load('pytorch/vision:v0.9.0', f'resnet{version}', pretrained=pretrained)
    # model = torchvision.models.resnet101(pretrained=pretrained, progress=progress)
    if in_channels is not 3:
        model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias is not None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
