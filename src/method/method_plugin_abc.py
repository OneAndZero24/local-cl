from abc import ABCMeta, abstractmethod


class MethodPluginABC(metaclass=ABCMeta):
    """
    Interface for continual learning methods as plugins for composer.

    Methods:
        _setup_task(task_id: int):
            Abstract method for setting up a task. Must be implemented by subclasses.
        _forward(x, y, loss, preds):
            Abstract method for the forward pass. Must be implemented by subclasses.
    """

    # TODO module

    @abstractmethod
    def _setup_task(self, task_id: int):
        """
        Internal setup task.
        
        Args:
            task_id (int): The unique identifier of the task to be set up.
        """

        pass


    @abstractmethod
    def _forward(self, x, y, loss, preds):
        """
        Internal forward pass.
        """

        pass