import logging
from typing import Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class UnlearnIntervalProtection:
    """
    Unlearning with interval protection for non-continual learning scenarios.
    
    This method protects the knowledge of classes we want to RETAIN by:
    1. Computing activation intervals on retain classes
    2. Penalizing weight changes that would affect activations in the protected intervals
    
    The goal is to unlearn specific classes while preserving the rest.
    
    Args:
        lambda_interval (float): Weight for the interval protection loss
        compute_intervals_from_data (bool): If True, computes intervals from data before unlearning
        infinity_scale (float): Scale factor for "infinity" bounds (default: 10.0)
    """
    
    def __init__(
        self,
        lambda_interval: float = 1.0,
        compute_intervals_from_data: bool = True,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.8,
        infinity_scale: float = 10.0
    ):
        self.lambda_interval = lambda_interval
        self.compute_intervals_from_data = compute_intervals_from_data
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.infinity_scale = infinity_scale
        
        self.params_snapshot = {}
        self.protected_intervals = []  # List of (min, max) tuples for each interval layer
        self.interval_layers = []
        
        log.info(f"UnlearnIntervalProtection initialized with lambda_interval={lambda_interval}, "
                 f"percentiles=[{lower_percentile}, {upper_percentile}], infinity_scale={infinity_scale}")
    
    
    def setup_protection(self, model: nn.Module, retain_dataloader, unlearn_dataloader, device):
        """
        Set up interval protection before unlearning starts.
        
        Steps:
        1. Collect activations from retain classes (what we want to keep)
        2. Compute protected intervals based on retain class activations
        3. Save parameter snapshot for computing weight changes
        
        Args:
            model: The model to protect
            retain_dataloader: DataLoader with samples from classes to retain
            unlearn_dataloader: DataLoader with samples from classes to unlearn (not used)
            device: Device to run computations on
        """
        
        log.info("Setting up interval protection...")
        
        # Find all IntervalActivation layers
        self.interval_layers = []
        for name, module in model.named_modules():
            if type(module).__name__ == "IntervalActivation":
                self.interval_layers.append((name, module))
                log.info(f"Found IntervalActivation layer: {name}")
        
        if len(self.interval_layers) == 0:
            log.warning("No IntervalActivation layers found! Protection will be disabled.")
            return
        
        # Collect activations from retain classes only
        retain_activations = self._collect_activations(model, retain_dataloader, device)
        
        # Compute protected intervals based on retain data
        self.protected_intervals = []
        for idx, (layer_name, layer) in enumerate(self.interval_layers):
            retain_acts = retain_activations[idx]  # Shape: (N_retain, features)
            
            if len(retain_acts) == 0:
                log.warning(f"No activations collected for layer {layer_name}")
                continue
            
            # Compute bounds for retain classes using percentiles
            sorted_acts, _ = torch.sort(retain_acts, dim=0)
            n_samples = sorted_acts.size(0)
            
            lower_idx = int(n_samples * self.lower_percentile)
            upper_idx = int(n_samples * self.upper_percentile)
            
            retain_min = sorted_acts[lower_idx]
            retain_max = sorted_acts[upper_idx]
            
            protected_interval = {
                'retain_min': retain_min,
                'retain_max': retain_max,
                'layer_name': layer_name
            }
            
            self.protected_intervals.append(protected_interval)
            
            log.info(f"Layer {layer_name}: Protected interval computed")
            log.info(f"  Retain range: [{retain_min.mean().item():.4f}, {retain_max.mean().item():.4f}]")
        
        # Save parameter snapshot
        self.params_snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_snapshot[name] = param.detach().clone()
        
        log.info(f"Interval protection setup complete. Tracking {len(self.params_snapshot)} parameters.")
    
    
    def _collect_activations(self, model, dataloader, device):
        """
        Collect activations from all IntervalActivation layers.
        
        Returns:
            List of activation tensors, one per interval layer
        """
        model.eval()
        
        activation_buffers = {idx: [] for idx in range(len(self.interval_layers))}
        hook_handles = []
        
        # Register hooks to collect activations
        for idx, (layer_name, layer) in enumerate(self.interval_layers):
            def hook(module, input, output, idx=idx):
                activation_buffers[idx].append(output.detach())
            
            handle = layer.register_forward_hook(hook)
            hook_handles.append(handle)
        
        # Run forward passes
        with torch.no_grad():
            for batch in dataloader:
                # Handle both 2-value and 3-value unpacking
                if len(batch) == 3:
                    X, y, _ = batch
                else:
                    X, y = batch
                X = X.to(device)
                _ = model(X)
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        
        # Concatenate all activations
        result = []
        for idx in range(len(self.interval_layers)):
            if len(activation_buffers[idx]) > 0:
                acts = torch.cat(activation_buffers[idx], dim=0)  # (N, features)
                result.append(acts)
            else:
                result.append(torch.tensor([]))
        
        model.train()
        return result
    
    
    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        """
        Compute the interval protection loss.
        
        This loss penalizes weight changes that would cause the output to change
        within the protected intervals: [-inf, forget_min] and [forget_max, +inf].
        
        Returns:
            Tensor: Protection loss value
        """
        
        if len(self.protected_intervals) == 0 or len(self.params_snapshot) == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        
        # For each interval layer, compute protection loss for two intervals
        for idx, interval_info in enumerate(self.protected_intervals):
            layer_name = interval_info['layer_name']
            forget_min = interval_info['retain_min'].to(device)
            forget_max = interval_info['retain_max'].to(device)
            
            # Find the next linear layer after this interval activation
            next_linear = self._find_next_linear(model, layer_name)
            
            if next_linear is None:
                continue
            
            # Compute "infinity" bounds
            neg_inf = forget_min - self.infinity_scale * torch.abs(forget_min)
            pos_inf = forget_max + self.infinity_scale * torch.abs(forget_max)
            
            # Apply loss to interval 1: [-inf, forget_min]
            lower_bound_reg_1 = None
            upper_bound_reg_1 = None
            
            # Apply loss to interval 2: [forget_max, +inf]
            lower_bound_reg_2 = None
            upper_bound_reg_2 = None
            
            for name, param in next_linear.named_parameters():
                # Find corresponding snapshot parameter
                param_full_name = None
                for mod_name, mod_param in model.named_parameters():
                    if mod_param is param:
                        param_full_name = mod_name
                        break
                
                if param_full_name is None or param_full_name not in self.params_snapshot:
                    continue
                
                prev_param = self.params_snapshot[param_full_name]
                
                if "weight" in name:
                    weight_diff = param - prev_param
                    weight_diff_pos = torch.relu(weight_diff)
                    weight_diff_neg = torch.relu(-weight_diff)
                    
                    # Interval 1: [-inf, forget_min]
                    lower_contrib_1 = weight_diff_pos @ neg_inf - weight_diff_neg @ forget_min
                    upper_contrib_1 = weight_diff_pos @ forget_min - weight_diff_neg @ neg_inf
                    
                    # Interval 2: [forget_max, +inf]
                    lower_contrib_2 = weight_diff_pos @ forget_max - weight_diff_neg @ pos_inf
                    upper_contrib_2 = weight_diff_pos @ pos_inf - weight_diff_neg @ forget_max
                    
                    if lower_bound_reg_1 is None:
                        lower_bound_reg_1 = lower_contrib_1
                        upper_bound_reg_1 = upper_contrib_1
                        lower_bound_reg_2 = lower_contrib_2
                        upper_bound_reg_2 = upper_contrib_2
                    else:
                        lower_bound_reg_1 = lower_bound_reg_1 + lower_contrib_1
                        upper_bound_reg_1 = upper_bound_reg_1 + upper_contrib_1
                        lower_bound_reg_2 = lower_bound_reg_2 + lower_contrib_2
                        upper_bound_reg_2 = upper_bound_reg_2 + upper_contrib_2
                
                elif "bias" in name:
                    bias_diff = param - prev_param
                    # Bias affects all outputs equally
                    if lower_bound_reg_1 is not None:
                        lower_bound_reg_1 = lower_bound_reg_1 + bias_diff
                        upper_bound_reg_1 = upper_bound_reg_1 + bias_diff
                        lower_bound_reg_2 = lower_bound_reg_2 + bias_diff
                        upper_bound_reg_2 = upper_bound_reg_2 + bias_diff
            
            if lower_bound_reg_1 is not None:
                # Add loss from both intervals
                total_loss += lower_bound_reg_1.sum().pow(2) + upper_bound_reg_1.sum().pow(2)
                total_loss += lower_bound_reg_2.sum().pow(2) + upper_bound_reg_2.sum().pow(2)
        
        return self.lambda_interval * total_loss
    
    
    def _find_next_linear(self, model: nn.Module, interval_layer_name: str) -> Optional[nn.Module]:
        """
        Find the next Linear layer after the specified IntervalActivation layer.
        """
        # Get ordered list of modules
        module_list = list(model.named_modules())
        
        # Find index of interval layer
        interval_idx = None
        for i, (name, module) in enumerate(module_list):
            if name == interval_layer_name:
                interval_idx = i
                break
        
        if interval_idx is None:
            return None
        
        # Find next Linear layer
        for i in range(interval_idx + 1, len(module_list)):
            name, module = module_list[i]
            if isinstance(module, nn.Linear):
                return module
            # Also check if it's an IncrementalClassifier with a Linear classifier
            if hasattr(module, 'classifier') and isinstance(module.classifier, nn.Linear):
                return module.classifier
        
        return None
