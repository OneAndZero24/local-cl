from torch import nn

from local import *


def instantiate(
        in_features: int, 
        out_features: int, 
        layer_type: str = "Linear", 
        train_domain: bool = True,
        toggle_linear: bool = False
    ):
    layer_map = {
        "Local": LocalLayer,
        "Linear": nn.Linear,
    }
    layer = layer_map[layer_type]
    if layer_type == "Local":
        return layer(in_features, out_features, train_domain, toggle_linear)
    return layer(in_features, out_features)


def instantiate2D(
        in_ch: int, 
        out_ch: int, 
        size: int,
        stride: int,
        conv_type: str= "Conv2d", 
        train_domain: bool=True
    ):
    conv_map = {
        "LocalConv": LocalConv2DLayer,
        "LocalConvOld": LocalConv2DLayerOld,
        "Conv2d": nn.Conv2d
    }
    conv = conv_map[conv_type]
    if conv_type != "Conv2d":
        if conv_type == "LocalConvOld":
            return conv(in_ch, out_ch, train_domain)
        conv(in_ch, out_ch, size, stride, train_domain)
    return conv(in_ch, out_ch, size, stride)