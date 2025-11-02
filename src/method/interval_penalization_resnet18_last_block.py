import logging
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ResNet18IntervalPenalizationLastBlock(MethodPluginABC):
    """
    Continual learning regularizer that protects representations learned inside 
    `IntervalActivation` hypercubes across tasks for the ResNet-18 architecture.
    The regularizer is applied only to the last block of the ResNet-18.

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
        Initializes the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Defaults to 0.01.
            lambda_int_drift (float, optional): Weight of the output preservation penalty. Defaults to 1.0.
            lambda_feat (float, optional): Weight of the interval drift penalty. Defaults to 1.0.
            use_hypercube_dist_loss (bool, optional): Whether to include the hypercube distance loss. Defaults to True.
            dil_mode (bool, optional): If True, applies regularization to the classifier head. Defaults to False.
            regularize_classifier (bool, optional): If True, includes the classifier in regularization. Defaults to False.
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
        self.data_buffer = []
        self.old_module = None

    def forward_with_snapshot(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the model forward using parameters and buffers from the previous task snapshot.

        This is used to obtain activations from the frozen (old) model for drift comparison
        against the current model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]]: Activations from the old model
            corresponding to the first `IntervalActivation` layer.
        """
        with torch.no_grad():
            identity, out = self.old_module.forward(x, return_first_interval_activation=True)

            return identity.detach(), out.detach()

    def setup_task(self, task_id: int) -> None:
        """
        Prepares the plugin for a new task.

        - For the first task (task_id == 0), only initializes internal variables.
        - For subsequent tasks, creates a frozen snapshot of model parameters and buffers,
          collects activations for all `IntervalActivation` layers from stored samples,
          and resets their activation intervals.

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
        Adds interval regularization penalties to the current task loss.

        The method computes and applies the following penalties:
            - Variance loss: discourages high variance within interval activations.
            - Drift loss: penalizes deviation from old-task activations inside hypercubes.
            - Output regularization: constrains parameter changes above intervals.
            - (Optional) Hypercube distance loss: keeps new representations close to previous ones.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target labels (not directly used here).
            loss (torch.Tensor): Current task loss.
            preds (torch.Tensor): Model predictions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the updated loss and unmodified predictions.
        """

        self.data_buffer.append(x)

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
                    identity_bounds = self.module.interval_l4_0_downsample_0

                    identity_bounds_min = identity_bounds.min.to(x.device)
                    identity_bounds_max = identity_bounds.max.to(x.device)

                    out_bounds = self.module.interval_l4_0_conv1
                    out_bounds_min = out_bounds.min.to(x.device)
                    out_bounds_max = out_bounds.max.to(x.device)

                    old_identity, old_out = self.forward_with_snapshot(x)
                    curr_identity, curr_out = self.module.forward(x, return_first_interval_activation=True)

                    mask_identity = ((old_identity >= identity_bounds_min) & (old_identity <= identity_bounds_max)).float()
                    mask_out = ((old_out >= out_bounds_min) & (old_out <= out_bounds_max)).float()

                    interval_drift_loss += (
                        (mask_identity * (old_identity - curr_identity).pow(2)).sum() / (mask_identity.sum() + 1e-8)
                    )
                    interval_drift_loss += (
                        (mask_out * (old_out - curr_out).pow(2)).sum() / (mask_out.sum() + 1e-8)
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

                if self.use_hypercube_dist_loss:
                    prev_center = (ub + lb) / 2.0
                    prev_radii  = (ub - lb) / 2.0
                    
                    lb_prev_hypercube = prev_center - prev_radii
                    ub_prev_hypercube = prev_center + prev_radii

                    lb_prev_hypercube = lb_prev_hypercube.view(-1)
                    ub_prev_hypercube = ub_prev_hypercube.view(-1)

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
    
    def _map_interval_to_nearest_param(self, module: nn.Module) -> dict:
        """
        Maps each `IntervalActivation` layer to the nearest learnable layer above it.

        Learnable layers include:
            - nn.Conv2d
            - nn.BatchNorm1d / nn.BatchNorm2d
            - nn.Linear

        This mapping is used to associate activation intervals with the layers
        most responsible for generating them, enabling localized regularization.

        Args:
            module (nn.Module): The model instance to analyze.

        Returns:
            dict: Mapping from `IntervalActivation` layers to their nearest parameterized layers.
        """
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
        
        return mapping