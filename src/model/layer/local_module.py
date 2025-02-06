import torch
import abc


class LocalModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class that acts as a wrapper around nn.Module.
    
    This class does not introduce additional functionality but allows
    instance checking against its child classes.
    """
    
    pass
