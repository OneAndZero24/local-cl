"""
Visualization script for regression tasks in continual learning.

This script creates publication-quality plots showing:
- Regression targets (ground truth)
- Model predictions
- Interval activation ranges (if using IntervalActivation)
- Task boundaries

Usage:
    python src/visualization.py --config-name interval_penalization_gauss_regression_mlp
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

from util.fabric import setup_fabric
from model.layer.interval_activation import IntervalActivation

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_scenarios(config: DictConfig):
    """Get train and test scenarios from config."""
    dataset_partial = instantiate(config.dataset)
    train_dataset = dataset_partial(train=True)
    test_dataset = dataset_partial(train=False)
    
    scenario_partial = instantiate(config.scenario)
    train_scenario = scenario_partial(train_dataset)
    test_scenario = scenario_partial(test_dataset)

    return train_scenario, test_scenario


def collect_activations(model, dataloader, fabric):
    """Run forward pass on dataloader and collect interval activations."""
    model.eval()
    
    with torch.no_grad():
        for X, y, _ in dataloader:
            _ = model(X)
            break  # Just need one pass to initialize
    
    # Collect activations from all IntervalActivation layers
    interval_layers = []
    for module in model.modules():
        if isinstance(module, IntervalActivation):
            interval_layers.append(module)
    
    return interval_layers


def visualize_regression(config: DictConfig):
    """
    Create visualization for regression tasks.
    
    For each task, plots:
    - Ground truth regression target
    - Model predictions
    - Interval activation ranges (shaded regions)
    """
    
    log.info('Initializing scenarios')
    train_scenario, test_scenario = get_scenarios(config)
    
    log.info('Launching Fabric')
    fabric = setup_fabric(config)
    
    log.info('Building model')
    model = fabric.setup(instantiate(config.model))
    
    log.info('Setting up method')
    method = instantiate(config.method)(model)
    
    # Prepare data
    log.info('Setting up dataloaders')
    train_tasks = []
    test_tasks = []
    for train_task, test_task in zip(train_scenario, test_scenario):
        train_tasks.append(fabric.setup_dataloaders(DataLoader(
            train_task, 
            batch_size=config.exp.batch_size, 
            shuffle=True,
            generator=torch.Generator(device=fabric.device)
        )))
        test_tasks.append(fabric.setup_dataloaders(DataLoader(
            test_task,
            batch_size=1,
            shuffle=False,
            generator=torch.Generator(device=fabric.device)
        )))
    
    N = len(train_scenario)
    
    # Set up plotting
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    
    # Create output directory
    output_dir = Path(config.exp.log_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare storage for activation ranges and trained models for each task
    task_data = [
        {"task_id": i, "old_intervals": None, "current_intervals": None, "model_state": None, "data_buffer": None}
        for i in range(N)
    ]
    
    # Find IntervalPenalization plugin
    interval_plugin = None
    if hasattr(method, 'plugins'):
        for plugin in method.plugins:
            if plugin.__class__.__name__ == 'IntervalPenalization':
                interval_plugin = plugin
                break
    
    # Train all tasks and save their models and data
    for task_id, train_task in enumerate(tqdm(train_tasks, desc="Tasks")):
        log.info(f'Task {task_id + 1}/{N}')
        
        # Setup task
        method.setup_task(task_id)
        
        # Train
        log.info('Training...')
        method.module.train()
        pbar = tqdm(range(config.exp.epochs), desc=f"Task {task_id+1} Training", leave=False)
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0
            for X, y, _ in train_task:
                # Ensure y has correct shape for regression
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)
                
                loss, preds = method.forward(X, y, task_id)
                loss = loss.mean()
                method.backward(loss)
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Log final statistics
        log.info(f'Task {task_id}: Final training loss = {avg_loss:.6f}')
        
        # Evaluate on test set to check actual fit
        method.module.eval()
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for X, y, _ in test_tasks[task_id]:
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)
                preds = method.module(X)
                if preds.dim() > 1 and preds.shape[-1] == 1:
                    preds = preds.squeeze(-1)
                loss = torch.nn.functional.mse_loss(preds, y)
                test_loss += loss.item()
                test_batches += 1
        avg_test_loss = test_loss / test_batches
        log.info(f'Task {task_id}: Test MSE loss = {avg_test_loss:.6f}')
        
        # Save model state and data buffer after training this task
        import io
        # Use state_dict to avoid deepcopy issues with non-leaf tensors
        task_data[task_id]['model_state'] = io.BytesIO()
        torch.save(method.module.state_dict(), task_data[task_id]['model_state'])
        task_data[task_id]['model_state'].seek(0)
        
        # Save data buffer if available
        if interval_plugin is not None and interval_plugin.data_buffer:
            task_data[task_id]['data_buffer'] = [x.clone() for x in interval_plugin.data_buffer]
    
    # Now compute intervals for all tasks
    log.info("Computing activation intervals for all tasks...")
    
    def compute_intervals(model, data_buffer, percentiles=(0.05, 0.95)):
        """Helper function to compute intervals from a model and data buffer.
        Only collects intervals from the regression head."""
        if not data_buffer:
            return None
        
        activation_buffers = {}
        # Only get IntervalActivation layers from the regression head
        interval_layers = []
        for name, module in model.named_modules():
            if isinstance(module, IntervalActivation) and 'head' in name:
                interval_layers.append(module)
        
        if not interval_layers:
            return None
        
        for idx, layer in enumerate(interval_layers):
            activation_buffers[idx] = []
        
        # Hook into interval layers to collect activations
        hook_handles = []
        for idx, layer in enumerate(interval_layers):
            def hook(module, input, output, idx=idx):
                activation_buffers[idx].append(output.detach())
            handle = layer.register_forward_hook(hook)
            hook_handles.append(handle)
        
        # Run model on data buffer
        model.eval()
        with torch.no_grad():
            for x_batch in data_buffer:
                x_batch = x_batch.to(next(model.parameters()).device)
                _ = model(x_batch)
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        
        # Calculate intervals from activations
        intervals = []
        for idx, layer in enumerate(interval_layers):
            if len(activation_buffers[idx]) > 0:
                activations = torch.cat(activation_buffers[idx], dim=0)
                activations_flat = activations.view(activations.size(0), -1)
                
                sorted_buf, _ = torch.sort(activations_flat, dim=0)
                n = sorted_buf.size(0)
                
                l_idx = int(np.clip(int(n * percentiles[0]), 0, n - 1))
                u_idx = int(np.clip(int(n * percentiles[1]), 0, n - 1))
                
                min_vals = sorted_buf[l_idx]
                max_vals = sorted_buf[u_idx]
                
                intervals.append({
                    'min': min_vals.cpu().numpy().copy(),
                    'max': max_vals.cpu().numpy().copy(),
                    'name': layer.log_name or f'head_layer_{idx}'
                })
        
        return intervals if intervals else None
    
    # For each task, compute current_intervals (using its own model on its own data)
    for task_id in range(N):
        if task_data[task_id]['model_state'] and task_data[task_id]['data_buffer']:
            # Load model from saved state
            temp_model = instantiate(config.model)
            task_data[task_id]['model_state'].seek(0)
            temp_model.load_state_dict(torch.load(task_data[task_id]['model_state']))
            temp_model = fabric.setup(temp_model)
            temp_model.eval()
            
            task_data[task_id]['current_intervals'] = compute_intervals(
                temp_model,
                task_data[task_id]['data_buffer']
            )
            log.info(f"Task {task_id}: Computed current_intervals")
    
    # For tasks 1+, compute old_intervals (using previous task's model on previous task's data)
    for task_id in range(1, N):
        if task_data[task_id - 1]['model_state'] and task_data[task_id - 1]['data_buffer']:
            # Load model from saved state
            temp_model = instantiate(config.model)
            task_data[task_id - 1]['model_state'].seek(0)
            temp_model.load_state_dict(torch.load(task_data[task_id - 1]['model_state']))
            temp_model = fabric.setup(temp_model)
            temp_model.eval()
            
            task_data[task_id]['old_intervals'] = compute_intervals(
                temp_model,
                task_data[task_id - 1]['data_buffer']
            )
            log.info(f"Task {task_id}: Computed old_intervals from task {task_id - 1}")
    
    log.info(f"Task data intervals summary:")
    for i, td in enumerate(task_data):
        log.info(f"  Task {i}: old_intervals={td['old_intervals'] is not None}, current_intervals={td['current_intervals'] is not None}")

    # Now create all visualizations with proper activation range alignment
    log.info('Creating visualizations...')
    
    # Use distinct, publication-quality colors for tasks
    task_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for task_id in range(N):
        fig, ax_main = plt.subplots(figsize=(12, 6))
        
        # Plot all previous and current tasks
        method.module.eval()
        with torch.no_grad():
            for tid in range(task_id + 1):
                task_x = []
                task_y_pred = []
                
                for X, y, _ in test_tasks[tid]:
                    preds = method.module(X)
                    
                    # Flatten to 1D for proper concatenation
                    task_x.append(X.cpu().numpy().flatten())
                    task_y_pred.append(preds.cpu().numpy().flatten())
                
                task_x = np.concatenate(task_x)
                task_y_pred = np.concatenate(task_y_pred)
                
                # Log prediction ranges for debugging
                log.info(f"Task {task_id}, plotting task {tid}: predictions in range [{task_y_pred.min():.4f}, {task_y_pred.max():.4f}]")
                
                # Sort by x for plotting
                sort_idx = np.argsort(task_x)
                task_x = task_x[sort_idx]
                task_y_pred = task_y_pred[sort_idx]
                
                # Plot predictions with distinct colors, no transparency
                ax_main.plot(task_x, task_y_pred, '-', color=task_colors[tid % len(task_colors)], 
                           linewidth=2.5, label=f'Task {tid+1}', zorder=4)
        
        # Plot the true function as reference (with transparency)
        # Dynamically determine x range from test data
        x_min_plot = test_tasks[0].dataset.x_range[0]
        x_max_plot = test_tasks[-1].dataset.x_range[1]
        x_full = np.linspace(x_min_plot, x_max_plot, 500)
        # Get the function from config
        func = instantiate(config.dataset.func)
        y_full = func(x_full)
        ax_main.plot(x_full, y_full, 'k--', linewidth=1.5, alpha=0.4, 
                    label='True Function', zorder=2)
        
        # Add vertical dashed lines for ALL task boundaries (not just trained ones)
        for tid in range(N):
            if hasattr(test_tasks[tid].dataset, 'x_range'):
                x_start, x_end = test_tasks[tid].dataset.x_range
                if tid > 0:  # Don't draw line before first task
                    ax_main.axvline(x_start, color='gray', linestyle='--', 
                                  linewidth=1.5, alpha=0.6, zorder=1)
        
        # Add activation range visualization
        # Use distinct colors for old and current intervals
        color_old = '#d62728'  # Red for old intervals
        color_current = '#1f77b4'  # Blue for current intervals
        
        # Get current task data
        current_task_data = task_data[task_id]
        
        log.info(f"Task {task_id}: Plotting intervals - old={current_task_data['old_intervals'] is not None}, current={current_task_data['current_intervals'] is not None}")
        
        # Show old_intervals (if available) - from previous task
        if current_task_data['old_intervals']:
            log.info(f"  Plotting {len(current_task_data['old_intervals'])} old interval layers")
            for layer_idx, layer_ranges in enumerate(current_task_data['old_intervals']):
                min_vals = layer_ranges['min']
                max_vals = layer_ranges['max']
                
                # Handle both scalar and array cases
                if min_vals.size == 1:
                    min_val = min_vals.item()
                    max_val = max_vals.item()
                else:
                    # For multi-dimensional activations, take the mean
                    min_val = min_vals.mean()
                    max_val = max_vals.mean()
                
                log.info(f"    Layer {layer_idx}: old interval [{min_val:.4f}, {max_val:.4f}]")
                
                # Draw old intervals with dashed lines (no fill)
                ax_main.axhline(min_val, color=color_old, linestyle='--', 
                              linewidth=2, zorder=3, 
                              label='Old Interval' if layer_idx == 0 else '')
                ax_main.axhline(max_val, color=color_old, linestyle='--', 
                              linewidth=2, zorder=3)
        
        # Show current_intervals (if available) - computed for this task using old_module
        if current_task_data['current_intervals']:
            log.info(f"  Plotting {len(current_task_data['current_intervals'])} current interval layers")
            for layer_idx, layer_ranges in enumerate(current_task_data['current_intervals']):
                min_vals = layer_ranges['min']
                max_vals = layer_ranges['max']
                
                # Handle both scalar and array cases
                if min_vals.size == 1:
                    min_val = min_vals.item()
                    max_val = max_vals.item()
                else:
                    # For multi-dimensional activations, take the mean
                    min_val = min_vals.mean()
                    max_val = max_vals.mean()
                
                log.info(f"    Layer {layer_idx} ({layer_ranges['name']}): current interval [{min_val:.4f}, {max_val:.4f}]")
                
                # Draw current intervals with dashed lines (no fill)
                ax_main.axhline(min_val, color=color_current, linestyle='--', 
                              linewidth=2, zorder=3,
                              label='Current Interval' if layer_idx == 0 else '')
                ax_main.axhline(max_val, color=color_current, linestyle='--', 
                              linewidth=2, zorder=3)
        
        ax_main.set_xlabel('Input (x)', fontsize=12)
        ax_main.set_ylabel('Output (y)', fontsize=12)
        ax_main.set_title(f'After Task {task_id + 1}', fontsize=14)
        ax_main.legend(loc='best', fontsize=10, framealpha=0.95)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'task_{task_id + 1:02d}_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f'Saved visualization to {output_path}')
        plt.close()
    
    log.info(f'All visualizations saved to {output_dir}')
