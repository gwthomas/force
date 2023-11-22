import torch.nn as nn
import torchvision.models as tvm

from force import defaults
from force.nn.module import ConfigurableModule


class TorchvisionModel(ConfigurableModule):
    """
    Lightweight wrapper around torchvision models, to integrate with config
    """

    class Config(ConfigurableModule.Config):
        arch = 'resnet18'
        pretrained = False

    def __init__(self, cfg, output_dim=None):
        super().__init__(cfg)
        assert hasattr(tvm, self.arch), f'{self.arch} is not a recognized torchvision model'
        self.model = getattr(tvm, self.arch)(pretrained=self.pretrained)

        if output_dim is not None:
            prev_fc = self.model.fc
            self.model.fc = nn.Linear(prev_fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)