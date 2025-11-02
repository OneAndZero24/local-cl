import logging
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class BigModelIntervalPenalization(MethodPluginABC):
    """
    Continual learning regularizer that protects representations learned inside 
    `IntervalActivation` hypercubes across tasks for the architectures implemented
    in `big_model.py` file.

    This plugin adds multiple penalties to the task loss:
    
    - **Variance loss (`var_scale`)**  
      Minimizes activation variance inside each interval, encouraging stable 
      and compact representations.
    
    - **Output preservation loss (`lambda_int_drift`)**  
      Constrains parameters above an `IntervalActivation` to keep producing 
      similar outputs for previously learned intervals.
    
    - **Interval drift loss (`lambda_feat`)**  
      Penalizes deviations of new activations from old-task activations 
      inside the same hypercube, with a stronger penalty near the cube center.

    - **Hypercube distance loss (`hypercube_dist_loss`)**  
      Penalizes distance between representations learned within hypercubes.

    Together, these terms reduce representation drift inside protected regions 
    while allowing free adaptation outside them.

    Args:
        var_scale (float): Weight of the variance regularizer.
        lambda_int_drift (float): Weight of the output preservation term.
        lambda_feat (float): Weight of the interval drift regularizer.
        use_hypercube_dist_loss (bool, optional): Whether to use the hypercube distance loss. Defaults to True.
        dil_mode (bool, optional): If True, also regularizes the classifier head. Defaults to False.
        regularize_classifier (bool, optional): If True, the classifier head is regularized. Defaults to False.

    Attributes:
        task_id (int): Identifier for the current task.
        params_buffer (dict): Snapshot of frozen parameters from the previous task.
        old_module (nn.Module): Deep copy of the previous model used for activation comparison.
        data_buffer (list): Buffer to store representative input samples.
        var_scale (float): Weight of the variance penalty term.
        lambda_int_drift (float): Weight of the output preservation term.
        lambda_feat (float): Weight of the drift penalty term.
        use_hypercube_dist_loss (bool): Flag indicating whether to include the hypercube distance loss.
        dil_mode (bool): Whether to apply regularization to the classifier head.
        regularize_classifier (bool): If True, includes classifier head in regularization.
    """

    def __init__(self,
            var_scale: float = 0.01,
            lambda_int_drift: float = 1.0,
            lambda_feat: float = 1.0,
            use_hypercube_dist_loss: bool = True,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Default: 0.01.
            lambda_int_drift (float, optional): Weight of the output preservation penalty. Default: 1.0.
            lambda_feat (float, optional): Weight of the interval drift penalty. Default: 1.0.
            use_hypercube_dist_loss (bool, optional): If True, hypercube distance loss is used to keep the learned
                                                      representations close to each other.
            dil_mode (bool, optional): If True, the classifier head is also regularized. If False (TIL/CIL scenarios)
                                        past class neurons should be simply masked without the regularization.
            regularize_classifier (bool, optional): If True, the classifier head is regularized. Default: False.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with var_scale={var_scale}, "
                 f"lambda_int_drift={lambda_int_drift}, "
                 f"lambda_feat={lambda_feat}")

        self.var_scale = var_scale
        self.lambda_int_drift = lambda_int_drift
        self.lambda_feat = lambda_feat
        self.use_hypercube_dist_loss = use_hypercube_dist_loss

        self.input_shape = None
        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier
        self.params_buffer = {}
        self.data_buffer = set()

    @torch.no_grad()
    def snapshot_state(self) -> dict:
        """
        Take a full snapshot of the current model state.  
        Stores both parameters and buffers (detached & cloned).  

        Returns:
            dict: {"params": OrderedDict, "buffers": OrderedDict}
        """
        return {
            "params": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_parameters()),
            "buffers": OrderedDict((k, v.detach().clone()) for k, v in self.module.named_buffers()),
        }


    def setup_task(self, task_id: int) -> None:
        """
        Prepare the plugin for a new task.  

        - Task 0: only sets `task_id`.  
        - Task >0: freezes trainable parameters, saves a snapshot in `old_state`,
        collects activations from all IntervalActivation layers over `self.data_buffer`,
        and resets their intervals.  

        Args:
            task_id (int): Identifier for the current task.
        """

        self.task_id = task_id
        if task_id > 0:
            self.params_buffer = {}

            for module in self.module.modules():
                if type(module).__name__ == "IntervalActivation":
                    del module.curr_task_last_batch

            for name, p in deepcopy(list(self.module.named_parameters())):
                if p.requires_grad:
                    p.requires_grad = False
                    self.params_buffer[name] = p.detach().clone()

            self.old_module = deepcopy(self.module)
            for p in self.old_module.parameters():
                p.requires_grad = False
            self.old_module.eval()

            activation_buffers = {}
            hook_handles = []

            for idx, layer in enumerate(self.module.layers):
                if type(layer).__name__ == "IntervalActivation":
                    activation_buffers[idx] = []

                    def hook(module, input, output, idx=idx):
                        activation_buffers[idx].append(output.detach())
                    
                    handle = layer.register_forward_hook(hook)
                    hook_handles.append(handle)

            self.module.eval()
            with torch.no_grad():
                for x in self.data_buffer:
                    x = x.to(next(self.module.parameters()).device)
                    _ = self.module(x)

            for idx, layer in enumerate(self.module.layers):
                if type(layer).__name__ == "IntervalActivation":
                    layer.reset_range(activation_buffers[idx])

            for handle in hook_handles:
                handle.remove()

        self.module.train()
        self.data_buffer = set()
                    
    def forward(self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, 
                preds: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Add interval regularization penalties to the current loss.  

        Penalties:
            - Variance loss: discourages variance within interval activations.  
            - Drift loss: penalizes change of activations inside the old-task hypercube.  
            - Output reg: discourages parameter updates that break interval consistency.  

        Args:
            x (torch.Tensor): Input tensor.  
            y (torch.Tensor): Target labels (unused here, passed through).  
            loss (torch.Tensor): Current task loss.  
            preds (torch.Tensor): Model predictions.  

        Returns:
            (loss, preds): Updated loss with added penalties, predictions unchanged.
        """

        self.data_buffer.add(x)

        layers = self.module.layers
        interval_act_layers = [module for _, module in self.module.c_head.named_modules() if type(module).__name__ == "IntervalActivation"]
        var_loss = torch.tensor(0.0, device=x.device)
        interval_drift_loss = torch.tensor(0.0, device=x.device)
        output_reg_loss = torch.tensor(0.0, device=x.device)
        hypercube_dist_loss = torch.tensor(0.0, device=x.device)

        # Drift only at the FIRST IntervalActivation
        if self.task_id > 0:
            y_old = self.old_module(x, return_first_interval_activation=True)
            layer = self.module.interval_act_layer_after_backbone
            acts = self.module.interval_act_layer_after_backbone.curr_task_last_batch
            lb = layer.min.to(x.device)
            ub = layer.max.to(x.device)
            mask = ((acts >= lb) & (acts <= ub)).float()
            interval_drift_loss += (
                (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
            )

            interval_act_layers.insert(0, layer)            

        for idx, layer in enumerate(interval_act_layers):

            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layer.min.to(x.device)
                ub = layer.max.to(x.device)
                
                # Regularize all layers above
                next_layer = layers[2*idx+1]

                if isinstance(next_layer, torch.nn.Linear):
                    target_module = next_layer
                elif (self.regularize_classifier or self.dil_mode) and hasattr(next_layer, "classifier"):
                    target_module = next_layer.classifier
                else:
                    target_module = None

                if target_module is not None:
                    lower_bound_reg = 0.0
                    upper_bound_reg = 0.0
                    for name, p in target_module.named_parameters():
                        for mod_name, mod_param in self.module.named_parameters():
                            if mod_param is p and mod_name in self.params_buffer:
                                prev_param = self.params_buffer[mod_name]
                                if "weight" in name:
                                    weight_diff = p - prev_param

                                    weight_diff_pos = torch.relu(weight_diff)
                                    weight_diff_neg = torch.relu(-weight_diff)

                                    lower_bound_reg += weight_diff_pos @ lb - weight_diff_neg @ ub
                                    upper_bound_reg += weight_diff_pos @ ub - weight_diff_neg @ lb

                                elif "bias" in name:
                                    bias_diff = p - prev_param

                                    lower_bound_reg += bias_diff
                                    upper_bound_reg += bias_diff

                    output_reg_loss += lower_bound_reg.sum().pow(2) + upper_bound_reg.sum().pow(2)

                if self.use_hypercube_dist_loss:
                    prev_center = (ub + lb) / 2.0
                    prev_radii  = (ub - lb) / 2.0
                    
                    lb_prev_hypercube = prev_center - prev_radii
                    ub_prev_hypercube = prev_center + prev_radii

                    new_lb, _ = acts_flat.min(dim=0)
                    new_ub, _ = acts_flat.max(dim=0)

                    non_overlap_mask = (new_lb > ub_prev_hypercube) | (new_ub < lb_prev_hypercube)
                    new_center = (new_ub + new_lb) / 2.0

                    center_loss = torch.norm(new_center[non_overlap_mask] - prev_center[non_overlap_mask], p=2)

                    hypercube_dist_loss += center_loss / (prev_radii.mean() + 1e-8)


        loss = (
            loss
            + self.var_scale * var_loss
            + self.lambda_int_drift * output_reg_loss
            + self.lambda_feat * interval_drift_loss
            + hypercube_dist_loss
        )
        return loss, preds