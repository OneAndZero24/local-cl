import logging
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        task_id (int): Identifier for the current task.
        params_buffer (dict): Snapshot of frozen parameters from the previous task.
        old_state (dict): Full parameter/buffer snapshot used for drift comparison.
        use_hypercube_dist_loss (bool, optional): If True, hypercube distance loss is used to keep the learned
                                                      representations close to each other.
        data_buffer (list): A buffer to store data samples.
        regularize_classifier (bool): If True, the classifier head is regularized. Default: False.

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
            use_hypercube_dist_loss: bool = True,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Default: 0.01.
            output_reg_scale (float, optional): Weight of the output preservation penalty. Default: 1.0.
            interval_drift_reg_scale (float, optional): Weight of the interval drift penalty. Default: 1.0.
            use_hypercube_dist_loss (bool, optional): If True, hypercube distance loss is used to keep the learned
                                                      representations close to each other.
            dil_mode (bool, optional): If True, the classifier head is also regularized. If False (TIL/CIL scenarios)
                                        past class neurons should be simply masked without the regularization.
            regularize_classifier (bool, optional): If True, the classifier head is regularized. Default: False.
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
        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier
        self.params_buffer = {}
        self.data_buffer = []
        self.old_module = None

    def forward_with_snapshot(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the model forward using parameters and buffers from the previous task snapshot.  
        Used to compare new activations with old-task activations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activations at the stopping point with old parameters/buffers.
        """
        if hasattr(self.old_module, "fe"):
            with torch.no_grad():
                _, out = self.old_module.forward(x, return_first_interval_activation=True)
        else:
            out = x.flatten(start_dim=1)
            for layer in self.old_module.mlp:
                out = layer(out)
                if type(layer).__name__ == "IntervalActivation":
                    break
        
        return out.detach()

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

            self.old_module = deepcopy(self.module)
            self.old_module.eval()
            for name, p in self.old_module.named_parameters():
                p.requires_grad = False
                self.params_buffer[name] = p.detach().clone()

            self.old_interval_to_param = self._map_interval_to_nearest_param(self.old_module)
            self.curr_interval_to_param = self._map_interval_to_nearest_param(self.module)
          
            # for (name_1, p_1), (name_2, p_2) in zip(self.old_interval_to_param.items(), self.curr_interval_to_param.items()):
            #     print(f"name_1: {name_1}, p1: {p_1}")
            #     print(f"name_2: {name_2}, p2: {p_2}")
            #     print("\n")
            # exit(0)
            self.old_interval_act_layers = [module for _, module in self.old_module.named_modules() if type(module).__name__ == "IntervalActivation"]

            interval_act_layers = [module for _, module in self.module.named_modules() if type(module).__name__ == "IntervalActivation"]

            activation_buffers = {}
            hook_handles = []
            for idx, layer in enumerate(interval_act_layers):
                activation_buffers[idx] = []

                def hook(module, input, output, idx=idx):
                    activation_buffers[idx].append(output.detach())
                
                handle = layer.register_forward_hook(hook)
                hook_handles.append(handle)

            with torch.no_grad():
                for x in self.data_buffer:
                    x = x.to(next(self.module.parameters()).device)
                    if x.size(0) == 1:
                        x = x.repeat(2, 1, 1, 1)
                    _ = self.module(x)

            for idx, layer in enumerate(interval_act_layers):
                layer.reset_range(activation_buffers[idx])

            for handle in hook_handles:
                handle.remove()

        self.data_buffer = []
                    
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

        self.data_buffer.append(x.detach())

        interval_act_layers = [module for _, module in self.module.named_modules() if type(module).__name__ == "IntervalActivation"]

        var_loss = torch.tensor(0.0, device=x.device)
        output_reg_loss = torch.tensor(0.0, device=x.device)
        interval_drift_loss = torch.tensor(0.0, device=x.device)
        hypercube_dist_loss = torch.tensor(0.0, device=x.device)


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
                    if hasattr(self.module, "fe"):
                        y_old = self.forward_with_snapshot(x)
                    else:
                        y_old = self.forward_with_snapshot(x.flatten(start_dim=1))
                    mask = ((acts >= lb) & (acts <= ub)).float()
                    interval_drift_loss += (
                        (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                    )
                   
                # Output reg for the nearest upper layer
                curr_target = self.curr_interval_to_param.get(layer, None)
                old_target = self.old_interval_to_param.get(self.old_interval_act_layers[idx], None)

                if curr_target is not None and old_target is not None:

                    # Handle classifier if applicable
                    if (self.regularize_classifier or self.dil_mode) and hasattr(curr_target, "classifier"):
                        curr_target = curr_target.classifier
                        old_target = old_target.classifier

                    lower_bound_reg = 0.0
                    upper_bound_reg = 0.0
                    for (param_name, p_curr), (_, p_old) in zip(curr_target.named_parameters(), old_target.named_parameters()):
                        weight_diff = p_curr - p_old
                        weight_diff_pos = torch.relu(weight_diff)
                        weight_diff_neg = torch.relu(-weight_diff)

                        if "weight" in param_name:
                            if isinstance(curr_target, nn.Linear):
                                lb = lb.view(-1)
                                ub = ub.view(-1)
                              
                                lower_bound_reg += (weight_diff_pos @ lb - weight_diff_neg @ ub).sum()
                                upper_bound_reg += (weight_diff_pos @ ub - weight_diff_neg @ lb).sum()
                            elif isinstance(curr_target, nn.Conv2d):
                                lb_view = lb.view(1, -1, 1, 1) if lb.dim() == 1 else lb
                                ub_view = ub.view(1, -1, 1, 1) if ub.dim() == 1 else ub

                                conv_kwargs = {
                                            "stride": curr_target.stride,
                                            "padding": curr_target.padding,
                                            "dilation": curr_target.dilation,
                                            "groups": curr_target.groups,
                                        }

                                lower = F.conv2d(lb_view, weight_diff_pos, None, **conv_kwargs).sum()
                                lower -= F.conv2d(ub_view, weight_diff_neg, None, **conv_kwargs).sum()
                                upper = F.conv2d(ub_view, weight_diff_pos, None, **conv_kwargs).sum()
                                upper -= F.conv2d(lb_view, weight_diff_neg, None, **conv_kwargs).sum()

                                lower_bound_reg += lower
                                upper_bound_reg += upper
                            elif isinstance(curr_target, nn.BatchNorm2d):
                                n_lb = (lb - old_target.running_mean) / torch.sqrt(old_target.running_var + old_target.eps)
                                n_ub = (ub - old_target.running_mean) / torch.sqrt(old_target.running_var + old_target.eps)
                                pos = weight_diff_pos.squeeze()
                                neg = weight_diff_neg.squeeze()
                                lower_bound_reg += (pos * n_lb - neg * n_ub).sum()
                                upper_bound_reg += (pos * n_ub - neg * n_lb).sum()
                        elif "bias" in param_name:
                            bias_diff = p_curr - p_old
                            lower_bound_reg += bias_diff.sum()
                            upper_bound_reg += bias_diff.sum()

                    output_reg_loss += lower_bound_reg.pow(2) + upper_bound_reg.pow(2)

                # if self.use_hypercube_dist_loss:
                #     prev_center = (ub + lb) / 2.0
                #     prev_radii  = (ub - lb) / 2.0
                    
                #     lb_prev_hypercube = prev_center - prev_radii
                #     ub_prev_hypercube = prev_center + prev_radii

                #     new_lb, _ = acts_flat.min(dim=0)
                #     new_ub, _ = acts_flat.max(dim=0)

                #     non_overlap_mask = (new_lb > ub_prev_hypercube) | (new_ub < lb_prev_hypercube)
                #     new_center = (new_ub + new_lb) / 2.0

                #     center_loss = torch.norm(new_center[non_overlap_mask] - prev_center[non_overlap_mask], p=2)

                #     hypercube_dist_loss += center_loss / (prev_radii.mean() + 1e-8)


        loss = (
            loss
            + self.var_scale * var_loss
            + self.output_reg_scale * output_reg_loss
            + self.interval_drift_reg_scale * interval_drift_loss
            + hypercube_dist_loss
        )
        return loss, preds
    
    def _map_interval_to_nearest_param(self, module: nn.Module) -> dict:
        """
        Map each IntervalActivation layer to the nearest learnable layer directly above it.

        A "learnable layer" is one of:
            - nn.Conv2d
            - nn.BatchNorm1d / nn.BatchNorm2d
            - nn.Linear

        Returns:
            dict: mapping {IntervalActivation: nearest parameterized layer or None}
        """
        if hasattr(module, 'fe'):
            # ResNet architecture
            m = module
            mapping = {
                m.interval_l4_0_conv1: m.fe.layer4_0_bn1,
                m.interval_l4_0_bn1: m.fe.layer4_0_conv2,
                m.interval_l4_0_conv2: m.fe.layer4_0_bn2,
                m.interval_l4_0_bn2: m.fe.layer4_1_conv1,
                m.interval_l4_1_conv1: m.fe.layer4_1_bn1,
                m.interval_l4_1_bn1: m.fe.layer4_1_conv2,
                m.interval_l4_1_conv2: m.fe.layer4_1_bn2,
                m.interval_l4_1_bn2: m.mlp[0],
                m.mlp[0]: m.mlp[1],
                m.mlp[1]: m.head if self.regularize_classifier else None
            }
            if hasattr(m.fe, 'layer4_0_downsample'):
                mapping[m.interval_l4_0_downsample_0] = m.fe.layer4_0_downsample[1]
        else:
            # MLP architecture
            m = module
            mapping = {
                m.mlp[0]: m.mlp[1],
                m.mlp[2]: m.head if self.regularize_classifier else None
            }
        return mapping