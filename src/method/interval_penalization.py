import logging
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict

import torch

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class IntervalPenalization(MethodPluginABC):
    """
    Continual learning regularizer that protects representations learned inside 
    `IntervalActivation` hypercubes across tasks.

    This plugin adds multiple penalties to the task loss:
    
    - **Variance loss (`var_scale`)**  
      Minimizes activation variance inside each interval, encouraging stable 
      and compact representations.
    
    - **Output preservation loss (`output_reg_scale`)**  
      Constrains parameters above an `IntervalActivation` to keep producing 
      similar outputs for previously learned intervals.
    
    - **Interval drift loss (`interval_drift_reg_scale`)**  
      Penalizes deviations of new activations from old-task activations 
      inside the same hypercube, with a stronger penalty near the cube center.

    - **Hypercube distance loss (`hypercube_dist_loss`)**
      Penalizes distance between representations learned within hypercubes.

    Together, these terms reduce representation drift inside protected regions, 
    while still allowing free adaptation outside.

    Attributes:
        var_scale (float): Weight of the variance regularizer.
        output_reg_scale (float): Weight of the output preservation term.
        interval_drift_reg_scale (float): Weight of the drift regularizer.
        task_id (int): Identifier of the current task.
        params_buffer (dict): Snapshot of frozen parameters from the previous task.
        old_state (dict): Full parameter/buffer snapshot used for drift comparison.
        use_hypercube_dist_loss (bool, optional): If True, hypercube distance loss is used to keep the learned
                                                      representations close to each other. 

    Methods:
        setup_task(task_id):
            Prepares state before starting a new task (snapshots old params/buffers).
        forward_with_snapshot(x, stop_at="IntervalActivation"):
            Runs a forward pass with frozen params up to the first IntervalActivation.
        snapshot_state():
            Creates a snapshot of all parameters and buffers.
        forward(x, y, loss, preds):
            Adds interval regularization terms to the given loss.
    """

    def __init__(self,
            var_scale: float = 0.01,
            output_reg_scale: float = 1.0,
            interval_drift_reg_scale: float = 1.0,
            use_hypercube_dist_loss: bool = True
        ) -> None:
        """
        Initialize the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Default: 0.01.
            output_reg_scale (float, optional): Weight of the output preservation penalty. Default: 1.0.
            interval_drift_reg_scale (float, optional): Weight of the interval drift penalty. Default: 1.0.
            use_hypercube_dist_loss (bool, optional): If True, hypercube distance loss is used to keep the learned
                                                      representations close to each other.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with var_scale={var_scale}, "
                 f"output_reg_scale={output_reg_scale}, "
                 f"interval_drift_reg_scale={interval_drift_reg_scale}")

        self.var_scale = var_scale
        self.output_reg_scale = output_reg_scale
        self.interval_drift_reg_scale = interval_drift_reg_scale
        self.use_hypercube_dist_loss = use_hypercube_dist_loss

        self.input_shape = None
        self.params_buffer = {}

    def forward_with_snapshot(self, x: torch.Tensor, stop_at: str="IntervalActivation") -> torch.Tensor:
        """
        Runs the model forward using parameters and buffers from the previous task snapshot.  
        Used to compare new activations with old-task activations.

        Args:
            x (torch.Tensor): Input tensor.
            stop_at (str, optional): Layer type name at which to stop the forward pass.
                                     Default is "IntervalActivation".

        Returns:
            torch.Tensor: Activations at the stopping point with old parameters/buffers.
        """
        saved_param_datas = {name: param.data for name, param in self.module.named_parameters()}
        saved_buffers = {name: buf for name, buf in self.module.named_buffers()}

        for name, param in self.module.named_parameters():
            param.data = self.old_state["params"][name].clone()
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = self.old_state["buffers"][name].clone()

        out = x
        for layer in self.module.layers:
            out = layer(out)
            if type(layer).__name__ == stop_at:
                break

        for name, param in self.module.named_parameters():
            param.data = saved_param_datas[name]
        
        for name, buf in self.module.named_buffers():
            self.module._buffers[name] = saved_buffers[name]

        return out.detach()

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

        - On task 0: only sets task id.  
        - On later tasks: freezes parameters, saves previous params to `params_buffer`,
          and snapshots full state into `old_state`.

        Args:
            task_id (int): Identifier for the current task.
        """

        self.task_id = task_id
        if task_id > 0:
            self.params_buffer = {}
            for name, p in deepcopy(list(self.module.named_parameters())):
                if p.requires_grad:
                    p.requires_grad = False
                    self.params_buffer[name] = p.detach().clone()
            self.old_state = self.snapshot_state()
                    
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

        x = x.flatten(start_dim=1)
        self.input_shape = x.shape

        layers = self.module.layers + [self.module.head]
        interval_act_layers = [layer for layer in layers if type(layer).__name__ == "IntervalActivation"]

        var_loss = 0.0
        output_reg_loss = 0.0
        interval_drift_loss = 0.0
        hypercube_dist_loss = 0.0

        for idx, layer in enumerate(interval_act_layers):

            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layer.min.to(x.device)
                ub = layer.max.to(x.device)

                # Drift only at the FIRST IntervalActivation
                if idx == 0:
                    y_old = self.forward_with_snapshot(x)
                    mask = ((acts >= lb) & (acts <= ub)).float()
                    interval_drift_loss += (
                        (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                    )

                # Output reg at this interval (first and all above)
                # In pattern [Linear, Interval, Linear, Interval, ...],
                # the *next* Linear belongs to this Interval
                next_layer = layers[2*idx+2]

                if isinstance(next_layer, torch.nn.Linear):
                    target_module = next_layer
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
                    prev_hypercube_center = (ub + lb) / 2.0
                    prev_hypercube_radii = (ub - lb) / 2.0
                    
                    lb_prev_hypercube = prev_hypercube_center - prev_hypercube_radii
                    ub_prev_hypercube = prev_hypercube_center + prev_hypercube_radii

                    acts_center = acts_flat.mean(dim=0)
                    hypercube_dist_loss = torch.relu(lb_prev_hypercube - acts_center) + torch.relu(acts_center - ub_prev_hypercube)
                    hypercube_dist_loss = hypercube_dist_loss.mean()

        loss = (
            loss
            + self.var_scale * var_loss
            + self.output_reg_scale * output_reg_loss
            + self.interval_drift_reg_scale * interval_drift_loss
            + hypercube_dist_loss
        )
        return loss, preds