from enum import Enum
from functools import partial

from torch import nn

from .local import LocalLayer
from .local_conv2d import LocalConv2DLayer


class LayerType(Enum):
    """
    enum = (LOCAL, NORMAL)
    """

    LOCAL = "Local"
    NORMAL = "Normal"


def _instantiate(
    map: dict,
    layer_type: str=LayerType.NORMAL,
    *args, 
    **kwargs
):
    layer = map[layer_type]
    if layer_type == LayerType.LOCAL:
        return layer(*args, **kwargs)
    return layer(*args)


instantiate = partial(_instantiate, {
    LayerType.LOCAL: LocalLayer,
    LayerType.NORMAL: nn.Linear
})


instantiate2D = partial(_instantiate, {
    LayerType.LOCAL: LocalConv2DLayer,
    LayerType.NORMAL: nn.Conv2d
})