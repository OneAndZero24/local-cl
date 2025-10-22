"""
Visualization script for regression tasks in continual learning.

This script creates publication-quality plots showing:
- Regression targets (ground truth)
- Model predictions
- Interval activation ranges (if using IntervalActivation)
- Task boundaries

Usage:
    python src/visualization.py --config-name naive_sin_regression_mlp
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
    
    # Colors for tasks
    colors = plt.cm.Set2(np.linspace(0, 1, N))
    
    # Prepare storage for activation ranges for each task
    # We'll fill these when setup_task(...) runs: the intervals computed in setup_task(k)
    # are the "current" intervals for task k-1 and the "reference" intervals for task k.
    task_data = [
        {"task_id": i, "reference_intervals": None, "current_intervals": None}
        for i in range(N)
    ]
    
    # Train and visualize each task
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
        
        # After calling setup_task(task_id), the plugin may have just computed
        # intervals (this happens for task_id > 0). The intervals computed in
        # setup_task(k) are based on data from task k-1. We capture them here
        # and assign them as:
        #   - task_data[k-1]['current_intervals'] (the intervals learned on task k-1)
        #   - task_data[k]['reference_intervals'] (the reference intervals used when training task k)
        if hasattr(method, 'plugins'):
            for plugin in method.plugins:
                if plugin.__class__.__name__ == 'IntervalPenalization':
                    interval_plugin = plugin
                    break
            else:
                interval_plugin = None

        # If setup_task set new ranges (only for task_id > 0), read them from model
        if task_id > 0:
            new_ranges = []
            for module in method.module.modules():
                if isinstance(module, IntervalActivation):
                    if module.min is not None and module.max is not None:
                        new_ranges.append({
                            'min': module.min.cpu().numpy().copy(),
                            'max': module.max.cpu().numpy().copy(),
                            'name': module.log_name or 'unnamed'
                        })

            # assign to previous task as its "current" intervals
            task_data[task_id - 1]['current_intervals'] = new_ranges if new_ranges else None
            # and also as reference intervals for current task
            task_data[task_id]['reference_intervals'] = new_ranges if new_ranges else None
        
    # After training all tasks, run one final setup_task(N) to compute the intervals
    # for the last task (these are based on task N-1 data). This ensures the
    # last task has its "current_intervals" populated as well.
    try:
        method.setup_task(N)
        # collect ranges set by the final setup
        final_ranges = []
        for module in method.module.modules():
            if isinstance(module, IntervalActivation):
                if module.min is not None and module.max is not None:
                    final_ranges.append({
                        'min': module.min.cpu().numpy().copy(),
                        'max': module.max.cpu().numpy().copy(),
                        'name': module.log_name or 'unnamed'
                    })
        if final_ranges:
            task_data[N - 1]['current_intervals'] = final_ranges
    except Exception:
        # If setup_task(N) is not supported for some method, ignore silently
        pass

    # Now create all visualizations with proper activation range alignment
    log.info('Creating visualizations...')
    for task_id in range(N):
        fig, ax_main = plt.subplots(figsize=(14, 8))
        
        # Collect all data for plotting
        all_x = []
        all_y_true = []
        all_y_pred = []
        
        # Plot all previous and current tasks
        method.module.eval()
        with torch.no_grad():
            for tid in range(task_id + 1):
                task_x = []
                task_y_true = []
                task_y_pred = []
                
                for X, y, _ in test_tasks[tid]:
                    preds = method.module(X)
                    
                    # Flatten to 1D for proper concatenation
                    task_x.append(X.cpu().numpy().flatten())
                    task_y_true.append(y.cpu().numpy().flatten())
                    task_y_pred.append(preds.cpu().numpy().flatten())
                
                task_x = np.concatenate(task_x)
                task_y_true = np.concatenate(task_y_true)
                task_y_pred = np.concatenate(task_y_pred)
                
                # Sort by x for plotting
                sort_idx = np.argsort(task_x)
                task_x = task_x[sort_idx]
                task_y_true = task_y_true[sort_idx]
                task_y_pred = task_y_pred[sort_idx]
                
                # Plot predictions only (no scatter points)
                ax_main.plot(task_x, task_y_pred, '-', color=colors[tid], 
                           linewidth=2.5, label=f'Task {tid+1} Pred', alpha=0.9, zorder=4)
        
        # Plot the true sine function as reference
        x_full = np.linspace(0, 15.707963267948966, 500)
        y_full = np.sin(x_full)
        ax_main.plot(x_full, y_full, 'k--', linewidth=1.5, alpha=0.5, 
                    label='True Function', zorder=2)
        
        # Add vertical dashed lines for task boundaries
        for tid in range(task_id + 1):
            if hasattr(test_tasks[tid].dataset, 'x_range'):
                x_start, x_end = test_tasks[tid].dataset.x_range
                if tid > 0:  # Don't draw line before first task
                    ax_main.axvline(x_start, color='gray', linestyle='--', 
                                  linewidth=1.5, alpha=0.6, zorder=1)
        
        ax_main.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Output (y)', fontsize=12, fontweight='bold')
        ax_main.set_title(f'Regression Performance after Task {task_id + 1}', 
                         fontsize=14, fontweight='bold', pad=20)
        ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Add activation range visualization
        # Use consistent colors: blue for current, red for reference/previous
        color_current = 'blue'
        color_reference = 'red'
        
        x_min, x_max = ax_main.get_xlim()
        
        # Get current task data
        current_task_data = task_data[task_id]
        
        # For task 0: show only current intervals
        # For task N>0: show both reference (previous) and current intervals
        if task_id == 0:
            # Task 1: Show only current intervals
            if current_task_data['current_intervals']:
                for layer_idx, layer_ranges in enumerate(current_task_data['current_intervals']):
                    min_vals = layer_ranges['min']
                    max_vals = layer_ranges['max']
                    layer_name = layer_ranges['name']
                    
                    if min_vals.size == 1:
                        min_val = min_vals.item()
                        max_val = max_vals.item()
                        
                        # Draw current intervals
                        ax_main.axhline(min_val, color=color_current, linestyle='--', 
                                      linewidth=2, alpha=0.7, zorder=3)
                        ax_main.axhline(max_val, color=color_current, linestyle='--', 
                                      linewidth=2, alpha=0.7, zorder=3)
                        
                        ax_main.axhspan(min_val, max_val, alpha=0.15, 
                                      color=color_current, zorder=0)
                        
                        mid_val = (min_val + max_val) / 2
                        label = 'Current Range' if layer_name == 'unnamed' else f'Current {layer_name}'
                        ax_main.text(x_max * 0.02, mid_val, label, 
                                   fontsize=8, ha='left', va='center',
                                   color=color_current, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', alpha=0.8, 
                                           edgecolor=color_current))
        else:
            # Task 2+: Show both reference (previous) and current intervals
            # First, show reference intervals (from previous tasks)
            if current_task_data['reference_intervals']:
                for layer_idx, layer_ranges in enumerate(current_task_data['reference_intervals']):
                    min_vals = layer_ranges['min']
                    max_vals = layer_ranges['max']
                    layer_name = layer_ranges['name']
                    
                    if min_vals.size == 1:
                        min_val = min_vals.item()
                        max_val = max_vals.item()
                        
                        # Draw reference intervals with dotted lines
                        ax_main.axhline(min_val, color=color_reference, linestyle=':', 
                                      linewidth=2, alpha=0.7, zorder=3)
                        ax_main.axhline(max_val, color=color_reference, linestyle=':', 
                                      linewidth=2, alpha=0.7, zorder=3)
                        
                        ax_main.axhspan(min_val, max_val, alpha=0.15, 
                                      color=color_reference, zorder=0)
                        
                        mid_val = (min_val + max_val) / 2
                        label = 'Previous Range' if layer_name == 'unnamed' else f'Prev {layer_name}'
                        ax_main.text(x_max * 0.98, mid_val, label, 
                                   fontsize=8, ha='right', va='center',
                                   color=color_reference, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', alpha=0.8, 
                                           edgecolor=color_reference))
            
            # Then, show current intervals
            if current_task_data['current_intervals']:
                for layer_idx, layer_ranges in enumerate(current_task_data['current_intervals']):
                    min_vals = layer_ranges['min']
                    max_vals = layer_ranges['max']
                    layer_name = layer_ranges['name']
                    
                    if min_vals.size == 1:
                        min_val = min_vals.item()
                        max_val = max_vals.item()
                        
                        # Draw current intervals
                        ax_main.axhline(min_val, color=color_current, linestyle='--', 
                                      linewidth=2, alpha=0.7, zorder=3)
                        ax_main.axhline(max_val, color=color_current, linestyle='--', 
                                      linewidth=2, alpha=0.7, zorder=3)
                        
                        ax_main.axhspan(min_val, max_val, alpha=0.15, 
                                      color=color_current, zorder=0)
                        
                        mid_val = (min_val + max_val) / 2
                        label = 'Current Range' if layer_name == 'unnamed' else f'Current {layer_name}'
                        ax_main.text(x_max * 0.02, mid_val, label, 
                                   fontsize=8, ha='left', va='center',
                                   color=color_current, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', alpha=0.8, 
                                           edgecolor=color_current))
        
        # Add task info text
        task_info = f'Trained on {task_id + 1} task(s)'
        fig.text(0.99, 0.01, task_info, ha='right', va='bottom', 
                fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'task_{task_id + 1:02d}_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f'Saved visualization to {output_path}')
        plt.close()
    
    # Create a final comparison plot showing all tasks
    fig, ax = plt.subplots(figsize=(14, 8))
    
    method.module.eval()
    with torch.no_grad():
        for tid in range(N):
            task_x = []
            task_y_true = []
            task_y_pred = []
            
            for X, y, _ in test_tasks[tid]:
                preds = method.module(X)
                # Flatten to 1D for proper concatenation
                task_x.append(X.cpu().numpy().flatten())
                task_y_true.append(y.cpu().numpy().flatten())
                task_y_pred.append(preds.cpu().numpy().flatten())
            
            task_x = np.concatenate(task_x)
            task_y_true = np.concatenate(task_y_true)
            task_y_pred = np.concatenate(task_y_pred)
            
            # Sort by x
            sort_idx = np.argsort(task_x)
            task_x = task_x[sort_idx]
            task_y_true = task_y_true[sort_idx]
            task_y_pred = task_y_pred[sort_idx]
            
            # Plot predictions only (no scatter points)
            ax.plot(task_x, task_y_pred, '-', color=colors[tid], 
                   linewidth=2.5, label=f'Task {tid+1}', alpha=0.9, zorder=4)
    
    # Plot the true sine function as reference
    x_full = np.linspace(0, 15.707963267948966, 500)
    y_full = np.sin(x_full)
    ax.plot(x_full, y_full, 'k--', linewidth=1.5, alpha=0.5, 
           label='True Function', zorder=2)
    
    # Add vertical dashed lines for task boundaries
    for tid in range(1, N):  # Start from 1 to skip line before first task
        if hasattr(test_tasks[tid].dataset, 'x_range'):
            x_start, _ = test_tasks[tid].dataset.x_range
            ax.axvline(x_start, color='gray', linestyle='--', 
                      linewidth=1.5, alpha=0.6, zorder=1)
    
    ax.set_xlabel('Input (x)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Output (y)', fontsize=14, fontweight='bold')
    ax.set_title('Final Regression Performance on All Tasks', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'final_all_tasks.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f'Saved final visualization to {output_path}')
    plt.close()
    
    log.info(f'All visualizations saved to {output_dir}')
