from torchvision import transforms

from .hydra import *
from .wandb import *
from .fabric import *

resize_transform = lambda : [
    transforms.Resize(32), 
    transforms.ToTensor()
]