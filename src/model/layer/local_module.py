import torch
import abc


class LocalModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class that acts as a wrapper around nn.Module.
    
    This class does not introduce additional functionality but allows
    instance checking against its child classes.
    """
    
    def incrementable_params(self):
        """
        Returns the incrementable parameters of the module. 
        Important for IncrementalClassifier.
        
        Returns:
            list: A list of incrementable parameters names.
        """

        return [name for name, _ in self.named_parameters()]


    def get_slice(self, old_nclasses):
        """
        Returns a slice object to select the parameters that will be updated.
        
        Args:
            old_nclasses (int): The number of classes before the increment.
        
        Returns:
            slice: A slice object to select the parameters that will be updated.
        """

        return slice(None, old_nclasses)