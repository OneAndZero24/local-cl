from enum import Enum
from functools import partial

from torch import nn

from .rbf import RBFLayer
from .rbf_head import SingleRBFHeadLayer
from .local import LocalLayer
from .local_conv2d import LocalConv2DLayer
from .local_module import LocalModule
from .local_head import LocalHead

class LayerType(Enum):
    """
    enum = (LOCAL, NORMAL, RBF)
    """

    LOCAL = "Local"
    NORMAL = "Normal"
    RBF = "RBF"
    SingleRBFHead = "SingleRBFHead"
    MultiRBFHead = "MultiRBFHead"


def _instantiate(
    map: dict,
    layer_type: str=LayerType.NORMAL,
    *args, 
    **kwargs
):
    layer = map[layer_type]
    if layer_type in [LayerType.NORMAL,LayerType.SingleRBFHead]:
        return layer(*args)
    return layer(*args, **kwargs)


instantiate = partial(_instantiate, {
    LayerType.LOCAL: LocalLayer,
    LayerType.RBF: RBFLayer,
    LayerType.NORMAL: nn.Linear,
    LayerType.SingleRBFHead: SingleRBFHeadLayer,
    LayerType.MultiRBFHead: RBFLayer
})


instantiate2D = partial(_instantiate, {
    LayerType.LOCAL: LocalConv2DLayer,
    LayerType.NORMAL: nn.Conv2d
})