import warnings
import logging

from torchvision import transforms

from .hydra import *
from .wandb import *
from .fabric import *
from .tensor import *

resize_transform = lambda : [
    transforms.Resize(32), 
    transforms.ToTensor()
]

def deprecation_warning(message: str):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.warning(message)
    warnings.warn(message, DeprecationWarning)
