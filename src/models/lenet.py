import torch
import torch.nn as nn

from models import instantiate, instantiate2D
from models.local import LocalLayer, LocalConv2DLayerOld, LocalConv2DLayer
from models.incremental import IncrementalClassifier


class LeNet(nn.Module):
    def __init__(self,
                 size: int,
                 stride: int,
                 sizes: list[int],
                 head_size: int,
                 conv_type: str = "Conv2d",
                 head_type: str = "Linear",
                 add_avg_pool: bool = True,
                 initial_out_features: int = 2,
                 **kwargs
            ):
        
        super().__init__()

        self.size = size
        self.stride = stride
        self.sizes = sizes
        self.head_size = head_size
        self.conv_type = conv_type
        self.initial_out_features = initial_out_features

        self.train_domain = kwargs["train_domain"] if "train_domain" in kwargs else True
        self.toggle_linear = kwargs["toggle_linear"] if "toggle_linear" in kwargs else False

        layers = []
        for i in range(len(sizes)-1):
            layers.append(instantiate2D(sizes[i], sizes[i+1], size, stride, conv_type, self.train_domain))
            layers.append(nn.Tanh())
            if add_avg_pool and (i < len(sizes)-2):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                layers.append(nn.Tanh())

        self.conv_seq = nn.Sequential(*layers)
        self.fc_seq = nn.Sequential(
            nn.Flatten(),
            instantiate(sizes[-1], head_size, head_type, self.train_domain, self.toggle_linear),
            nn.Tanh(),
        )

        self.actiavtions = None
        self.head = IncrementalClassifier(
            head_size, 
            initial_out_features,
            head_type,
            train_domain=self.train_domain,
            toggle_linear=self.toggle_linear
        )

    def forward(self, x):
        activations = []
        for layer in self.conv_seq:
            x = layer(x)
            
            if isinstance(layer, LocalConv2DLayer) or isinstance(layer, LocalConv2DLayerOld):
                activations.append(x)
        
        for layer in self.fc_seq:
            x = layer(x)

            if isinstance(layer, LocalLayer):
                activations.append(x)

        self.activations = activations
        x = self.head(x)
        return x
    
    def get_args(self):
        """
        Return the arguments required to reinitialize this model.
        """
        return {
            "size": self.size,
            "stride": self.stride,
            "sizes": self.sizes,
            "head_size": self.head_size,
            "conv_type": self.conv_type,
            "initial_out_features": self.initial_out_features,
            "train_domain": self.train_domain,
            "toggle_linear": self.toggle_linear
        }
