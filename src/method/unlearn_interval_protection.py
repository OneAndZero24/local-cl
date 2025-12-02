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
    2. Computing the "negative space" (inverse intervals) by running unlearn classes
    3. Penalizing weight changes that would affect activations in the protected intervals
    
    The goal is to unlearn specific classes while preserving the rest.
    
    Args:
        lambda_interval (float): Weight for the interval protection loss
        compute_intervals_from_data (bool): If True, computes intervals from data before unlearning
    """
    
    def __init__(
        self,
        lambda_interval: float = 1.0,
        compute_intervals_from_data: bool = True,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ):
        self.lambda_interval = lambda_interval
        self.compute_intervals_from_data = compute_intervals_from_data
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        self.params_snapshot = {}
        self.protected_intervals = []  # List of (min, max) tuples for each interval layer
        self.interval_layers = []
        
        log.info(f"UnlearnIntervalProtection initialized with lambda_interval={lambda_interval}, "
                 f"percentiles=[{lower_percentile}, {upper_percentile}]")
    
    
    def setup_protection(self, model: nn.Module, retain_dataloader, unlearn_dataloader, device):
        """
        Set up interval protection before unlearning starts.
        
        Steps:
        1. Collect activations from retain classes (what we want to keep)
        2. Collect activations from unlearn classes (what we want to forget)
        3. Compute protected intervals as the "negative space" - regions where retain 
           classes are active but unlearn classes are not
        4. Save parameter snapshot for computing weight changes
        
        Args:
            model: The model to protect
            retain_dataloader: DataLoader with samples from classes to retain
            unlearn_dataloader: DataLoader with samples from classes to unlearn
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
        
        # Collect activations
        retain_activations = self._collect_activations(model, retain_dataloader, device)
        unlearn_activations = self._collect_activations(model, unlearn_dataloader, device)
        
        # Compute protected intervals (negative space of unlearn intervals)
        self.protected_intervals = []
        for idx, (layer_name, layer) in enumerate(self.interval_layers):
            retain_acts = retain_activations[idx]  # Shape: (N_retain, features)
            unlearn_acts = unlearn_activations[idx]  # Shape: (N_unlearn, features)
            
            # Compute bounds for retain and unlearn classes
            retain_min = retain_acts.min(dim=0)[0]
            retain_max = retain_acts.max(dim=0)[0]
            
            unlearn_min = unlearn_acts.min(dim=0)[0]
            unlearn_max = unlearn_acts.max(dim=0)[0]
            
            # Protected intervals: where retain is active but unlearn is not
            # We protect the regions [retain_min, unlearn_min) and (unlearn_max, retain_max]
            # This creates two "safe zones" on either side of the unlearn region
            
            protected_interval = {
                'retain_min': retain_min,
                'retain_max': retain_max,
                'unlearn_min': unlearn_min,
                'unlearn_max': unlearn_max,
                'layer_name': layer_name
            }
            
            self.protected_intervals.append(protected_interval)
            
            log.info(f"Layer {layer_name}: Protected intervals computed")
            log.info(f"  Retain range: [{retain_min.mean().item():.4f}, {retain_max.mean().item():.4f}]")
            log.info(f"  Unlearn range: [{unlearn_min.mean().item():.4f}, {unlearn_max.mean().item():.4f}]")
        
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
            for X, y, _ in dataloader:
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
        within the protected intervals (where retain classes are but unlearn classes aren't).
        
        Based on the output_reg_loss from interval_penalization_resnet18_cls.py
        
        Returns:
            Tensor: Protection loss value
        """
        
        if len(self.protected_intervals) == 0 or len(self.params_snapshot) == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        
        # For each interval layer, compute protection loss
        for idx, interval_info in enumerate(self.protected_intervals):
            layer_name = interval_info['layer_name']
            retain_min = interval_info['retain_min'].to(device)
            retain_max = interval_info['retain_max'].to(device)
            unlearn_min = interval_info['unlearn_min'].to(device)
            unlearn_max = interval_info['unlearn_max'].to(device)
            
            # Find the next linear layer after this interval activation
            # Assuming structure: IntervalActivation -> Linear -> ...
            next_linear = self._find_next_linear(model, layer_name)
            
            if next_linear is None:
                continue
            
            # Compute output change bounds for PROTECTED regions
            # We protect two regions: [retain_min, unlearn_min] and [unlearn_max, retain_max]
            
            lower_bound_reg = torch.tensor(0.0, device=device)
            upper_bound_reg = torch.tensor(0.0, device=device)
            
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
                    
                    # Protect lower interval [retain_min, unlearn_min]
                    lower_bound_reg += weight_diff_pos @ retain_min - weight_diff_neg @ unlearn_min
                    upper_bound_reg += weight_diff_pos @ unlearn_min - weight_diff_neg @ retain_min
                    
                    # Protect upper interval [unlearn_max, retain_max]
                    lower_bound_reg += weight_diff_pos @ unlearn_max - weight_diff_neg @ retain_max
                    upper_bound_reg += weight_diff_pos @ retain_max - weight_diff_neg @ unlearn_max
                
                elif "bias" in name:
                    bias_diff = param - prev_param
                    lower_bound_reg += bias_diff
                    upper_bound_reg += bias_diff
            
            total_loss += lower_bound_reg.sum().pow(2) + upper_bound_reg.sum().pow(2)
        
        return total_loss
    
    
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
