from typing import Optional

from torch import nn
from torch import optim

from model.cl_module_abc import CLModuleABC
from method.regularization import regularization
from method.method_plugin_abc import MethodPluginABC


class Composer:
    def __init__(self, 
        module: CLModuleABC,
        criterion: nn.Module, 
        first_lr: float, 
        lr: float,
        reg_type: Optional[str]=None,
        gamma: Optional[float]=None,
        clipgrad: Optional[float]=None,
        plugins: Optional[list[MethodPluginABC]]=[]
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = None
        self.first_lr = first_lr
        self.lr = lr
        self.reg_type = reg_type
        self.gamma = gamma
        self.clipgrad = clipgrad
        self.plugins = plugins
        # TODO init plugins with module

    def _setup_optim(self, task_id: int):
        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.first_lr if task_id == 0 else self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    def _add_reg(self, loss):
        if self.gamma is not None and self.reg_type is not None:
            loss = (1-self.gamma)*loss+self.gamma*regularization(self.module.activations, self.reg_type)
        return loss

    def setup_task(self, task_id: int):
        self._setup_optim(task_id)
        for plugin in self.plugins:
            plugin._setup_task(task_id)


    def forward(self, x, y):
        preds = self.module(x)
        loss = self.criterion(preds, y)
        loss = self._add_reg(loss)

        for plugin in self.plugins:
            loss, preds = plugin._forward(x, y, loss, preds)
        return loss, preds

    def backward(self, loss):  
        self.optimizer.zero_grad()
        loss.backward()
        if self.clipgrad is not None:
            nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()