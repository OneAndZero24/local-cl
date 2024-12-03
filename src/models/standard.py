import torch.nn as nn
import torch.nn.functional as F
from src.models.dynamic import MultiTaskModule
from torch import Tensor


class MLP(MultiTaskModule):
    """Multi-layer perceptron."""

    def __init__(self, input_size: int, hidden_dim: int, output_classes: int, heads: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes

        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last = nn.ModuleList(nn.Linear(self.hidden_dim, self.output_classes) for _ in range(heads))

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last[task_id](x)

        return x

    @property
    def device(self):
        return self.fc1.weight.device


class VGG(MultiTaskModule):
    CFG = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self,
                 variant: str,
                 in_channels: int,
                 output_classes: int,
                 heads: int,
                 batch_norm: bool):
        super().__init__()
        self.features = self.make_layers(self.CFG[variant], in_channels, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.last = nn.ModuleList(nn.Linear(4096, output_classes) for _ in range(heads))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.last[task_id](x)
        return x

    @staticmethod
    def make_layers(cfg, in_channels, batch_norm):
        layers = []
        input_channel = in_channels
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(l)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = l
        return nn.Sequential(*layers)


class MobileNet(MultiTaskModule):
    class Block(nn.Module):
        '''Depthwise conv + Pointwise conv'''

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            conv_layers = []
            conv_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                          bias=False))
            conv_layers.append(nn.BatchNorm2d(in_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())

            self.layers = nn.Sequential(*conv_layers)

        def forward(self, x):
            fwd = self.layers(x)
            return fwd

    def __init__(self,
                 in_channels: int,
                 output_classes: int,
                 heads: int,
                 batch_norm: bool):
        super().__init__()
        self.cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        if batch_norm is False:
            raise NotImplementedError('MobileNet has no no-BN variant!')
        self.in_channels = in_channels
        self.initial_channels = 32
        init_conv = []
        init_conv.append(
            nn.Conv2d(self.in_channels, self.initial_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.initial_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)
        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.initial_channels))
        end_layers = []
        end_layers.append(nn.AvgPool2d(2))
        end_layers.append(nn.Flatten())
        self.end_layers = nn.Sequential(*end_layers)
        self.last = nn.ModuleList(nn.Linear(1024, output_classes) for _ in range(heads))

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(self.Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return layers

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = self.init_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.end_layers(x)
        x = self.last[task_id](x)
        return x


class AlexNet(MultiTaskModule):

    def __init__(self, in_channels, output_classes, heads):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.last = nn.ModuleList(nn.Linear(4096, output_classes) for _ in range(heads))

    def forward_single_task(self, x: Tensor, task_id: int):  # type: ignore
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        x = self.last[task_id](x)
        return x
