import logging
from typing import Optional

from torch import nn
from torch import optim

import wandb

from model.cl_module_abc import CLModuleABC
from method.regularization import regularization
from method.method_plugin_abc import MethodPluginABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Composer:
    """
    Composer class for managing the training process of a module with optional plugins and regularization.

    Attributes:
        module (CLModuleABC): The module to be trained.
        criterion (nn.Module): The loss function.
        optimizer (Optional[optim.Optimizer]): The optimizer for training.
        first_lr (float): The learning rate for the first task.
        lr (float): The learning rate for subsequent tasks.
        reg_type (Optional[str]): The type of regularization to apply.
        gamma (Optional[float]): The regularization strength.
        clipgrad (Optional[float]): The gradient clipping value.
        plugins (Optional[list[MethodPluginABC]]): List of plugins to be used during training.

    Methods:
        __init__(module, criterion, first_lr, lr, reg_type=None, gamma=None, clipgrad=None, plugins=[]):
            Initializes the Composer with the given parameters.
        _setup_optim(task_id):
            Sets up the optimizer for the given task.
        _add_reg(loss):
            Adds regularization to the loss if applicable.
        setup_task(task_id):
            Sets up the task by initializing the optimizer and plugins.
        forward(x, y):
            Performs a forward pass through the module and plugins, returning the loss and predictions.
        backward(loss):
            Performs a backward pass, applying gradient clipping if specified, and updates the model parameters.
    """

    def __init__(self, 
        module: CLModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        reg_type: Optional[str]=None,
        gamma: Optional[float]=None,
        clipgrad: Optional[float]=None,
        retaingraph: Optional[bool]=False,
        log_reg: Optional[bool]=False,
        plugins: Optional[list[MethodPluginABC]]=[]
    ):
        """
        Initialize the Composer class.

        Args:
            module (CLModuleABC): The module to be used.
            criterion (nn.Module): The criterion (loss function) to be used.
            first_lr (float): The initial learning rate.
            lr (float): The learning rate.
            reg_type (Optional[str], optional): The type of regularization to be used. Defaults to None.
            gamma (Optional[float], optional): The gamma value for learning rate decay. Defaults to None.
            clipgrad (Optional[float], optional): The value to clip gradients. Defaults to None.
            retaingraph (Optional[bool], optional): Whether to retain the computation graph. Defaults to False.
            log_reg (Optional[bool], optional): Whether to log the regularization loss. Defaults to False.
            plugins (Optional[list[MethodPluginABC]], optional): A list of plugins to be used. Defaults to an empty list.
        """

        self.module = module
        self.criterion = criterion
        self.optimizer = None
        self.first_lr = first_lr
        self.lr = lr
        self.reg_type = reg_type
        self.gamma = gamma
        self.clipgrad = clipgrad
        self.retaingraph = retaingraph
        self.plugins = plugins
        self.log_reg = log_reg
        
        for plugin in self.plugins:
            plugin.set_module(self.module)
            log.info(f'Plugin {plugin.__class__.__name__} added to composer')


    def _setup_optim(self, task_id: int):
        """
        Sets up the optimizer for the model.
        This method initializes the optimizer with the model parameters that require
        gradients. It uses the Adam optimizer with a learning rate that depends on
        the task ID. If the task ID is 0, it uses `first_lr`, otherwise it uses `lr`.

        Args:
            task_id (int): The ID of the current task. Determines the learning rate to use.
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.first_lr if task_id == 0 else self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    def _add_reg(self, loss):
        """
        Adds regularization to the given loss if gamma and reg_type are set.

        Args:
            loss (float): The original loss value.

        Returns:
            float: The loss value with regularization added if applicable.
        """

        if self.gamma is not None and self.reg_type is not None:
            loss = (1-self.gamma)*loss+self.gamma*regularization(self.module.activations, self.reg_type)
        return loss


    def setup_task(self, task_id: int):
        """
        Set up the task with the given task ID.
        This method initializes the optimizer for the specified task and
        calls the setup_task method on each plugin associated with this instance.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """

        self._setup_optim(task_id)
        for plugin in self.plugins:
            plugin.setup_task(task_id)


    def forward(self, x, y, task_id):
        """
        Perform a forward pass through the model and apply plugins.

        Args:
            x (torch.Tensor): Input tensor to the model.
            y (torch.Tensor): Target tensor for computing the loss.
            task_id (int): The ID of the current task.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss after applying regularization and plugins.
                - preds (torch.Tensor): The model predictions after applying plugins.
        """

        preds = self.module(x)
        loss = self.criterion(preds, y)
        loss = self._add_reg(loss)

        old_loss = loss
        for plugin in self.plugins:
            loss, preds = plugin.forward(x, y, loss, preds)
        if self.log_reg:
            wandb.log({f'Loss/train/{task_id}/reg': loss-old_loss})
        return loss, preds


    def backward(self, loss):  
        """
        Performs a backward pass and updates the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
            
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Optionally clips the gradients to prevent exploding gradients.
        4. Updates the model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        loss.backward(retain_graph=self.retaingraph)
        if self.clipgrad is not None:
            nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()