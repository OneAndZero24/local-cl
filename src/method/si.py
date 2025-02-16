from copy import deepcopy
import torch

from method.regularization import param_change_loss
from src.method.method_plugin_abc import MethodPluginABC


class SI(MethodPluginABC):
    def __init__(self,
        alpha: float,
        eps: float = 1e-6
    ):
        super().__init__()
        self.task_id = None
        self.alpha = alpha
        self.eps = eps

        self.prev_param = {}
        self.omega = {}
        self.importance = {}


    def setup_task(self, task_id: int):
        self.task_id = task_id
        if task_id == 0:
            for name, p in self.module.named_parameters():
                if p.requires_grad:
                    self.prev_param[name] = p.data.clone()
                    self.omega[name] = torch.zeros_like(p)
                    self.importance[name] = torch.zeros_like(p)
        elif task_id > 0:
            self._compute_importance()


    def forward(self, x, y, loss, preds):
        params_buffer = {}
        for name, p in self.module.named_parameters():
            params_buffer[name] = torch.zeros_like(p)
            s = self.prev_param[name].shape[0]
            params_buffer[name][:s] = self.prev_param[name].clone()
            if p.requires_grad and p.grad is not None:
                delta_param = p.data - params_buffer[name]
                self.omega[name] += p.grad * delta_param / (delta_param ** 2 + 1e-6)
                self.prev_param[name] = p.data.clone()

        loss *= self.alpha
        loss += (1-self.alpha)*param_change_loss(self.module, self.importance, params_buffer)
        return loss, preds
    

    def _compute_importance(self):
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                s = self.omega[name].shape[0]
                tmp_omega = torch.zeros_like(p)
                tmp_prev = torch.zeros_like(p)
                tmp_omega[:s] = self.omega[name]
                tmp_prev[:s] = self.prev_param[name]
                tmp_importance = self.importance[name]
                self.importance[name] = tmp_omega / (((p.data - tmp_prev)**2)+self.eps)
                self.importance[name][:s] += tmp_importance
                self.omega[name] = torch.zeros_like(p)