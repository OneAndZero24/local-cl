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
    
    # Prepare storage for predictions and intervals for each task
    task_data = [
        {
            "task_id": i, 
            "predictions_per_task": {},  # Predictions on ALL tasks (key: task_id, value: {x, y})
            "old_interval": None,  # Previous task's interval (computed at setup)
            "current_interval": None  # This task's interval (computed after training)
        }
        for i in range(N)
    ]
    
    # Find IntervalPenalization plugin
    interval_plugin = None
    if hasattr(method, 'plugins'):
        for plugin in method.plugins:
            if plugin.__class__.__name__ == 'IntervalPenalization':
                interval_plugin = plugin
                break
    
    def compute_interval_from_data(model, data_buffer, percentiles=(0.05, 0.95)):
        """Compute interval from model predictions on data buffer."""
        if not data_buffer or len(data_buffer) == 0:
            return None
        
        # Collect all predictions
        model.eval()
        all_predictions = []
        with torch.no_grad():
            for x_batch in data_buffer:
                x_batch = x_batch.to(next(model.parameters()).device)
                preds = model(x_batch)
                all_predictions.append(preds.detach())
        
        # Concatenate and compute percentiles
        all_predictions = torch.cat(all_predictions, dim=0)
        all_predictions = all_predictions.view(-1)  # Flatten
        
        sorted_preds, _ = torch.sort(all_predictions)
        n = sorted_preds.size(0)
        
        l_idx = int(np.clip(int(n * percentiles[0]), 0, n - 1))
        u_idx = int(np.clip(int(n * percentiles[1]), 0, n - 1))
        
        min_val = sorted_preds[l_idx].item()
        max_val = sorted_preds[u_idx].item()
        
        return {'min': min_val, 'max': max_val}
    
    # Train all tasks
    for task_id, train_task in enumerate(tqdm(train_tasks, desc="Tasks")):
        log.info(f'Task {task_id + 1}/{N}')
        
        # BEFORE training: If task > 0, compute old_interval as union of all previous current_intervals
        if task_id > 0:
            log.info(f'Computing old_interval for task {task_id} as union of all previous intervals...')
            # Collect all previous current_intervals
            previous_intervals = [task_data[i]['current_interval'] for i in range(task_id) if task_data[i]['current_interval'] is not None]
            
            if len(previous_intervals) > 0:
                # Take min of all lower bounds and max of all upper bounds
                all_mins = [interval['min'] for interval in previous_intervals]
                all_maxs = [interval['max'] for interval in previous_intervals]
                task_data[task_id]['old_interval'] = {
                    'min': min(all_mins),
                    'max': max(all_maxs)
                }
                log.info(f"  Old interval (union of {len(previous_intervals)} previous): [{task_data[task_id]['old_interval']['min']:.4f}, {task_data[task_id]['old_interval']['max']:.4f}]")
        
        # Setup task (clears data_buffer for new task)
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
        
        # AFTER training: Save predictions on ALL tasks (including future ones)
        log.info(f'Saving predictions for all tasks 0 to {N-1}...')
        method.module.eval()
        with torch.no_grad():
            for tid in range(N):  # Changed to include all tasks, not just seen ones
                task_x = []
                task_y_pred = []
                for X, y, _ in test_tasks[tid]:
                    preds = method.module(X)
                    task_x.append(X.cpu().numpy().flatten())
                    task_y_pred.append(preds.cpu().numpy().flatten())
                
                task_data[task_id]['predictions_per_task'][tid] = {
                    'x': np.concatenate(task_x),
                    'y': np.concatenate(task_y_pred)
                }
                log.info(f"  Task {task_id} model on Task {tid} data: predictions range [{task_data[task_id]['predictions_per_task'][tid]['y'].min():.4f}, {task_data[task_id]['predictions_per_task'][tid]['y'].max():.4f}]")
        
        # AFTER training: Compute current_interval from this task's data
        if interval_plugin is not None and len(interval_plugin.data_buffer) > 0:
            log.info(f'Computing current_interval for task {task_id}...')
            task_data[task_id]['current_interval'] = compute_interval_from_data(
                method.module,
                interval_plugin.data_buffer
            )
            if task_data[task_id]['current_interval']:
                log.info(f"  Current interval: [{task_data[task_id]['current_interval']['min']:.4f}, {task_data[task_id]['current_interval']['max']:.4f}]")
    
    # Log summary
    log.info("Task data summary:")
    for i, td in enumerate(task_data):
        log.info(f"  Task {i}: predictions_per_task={len(td['predictions_per_task'])} tasks, old_interval={td['old_interval'] is not None}, current_interval={td['current_interval'] is not None}")

    # Now create all visualizations
    log.info('Creating visualizations...')
    
    # Determine global x and y ranges for consistent axes across all plots
    x_min_global = test_tasks[0].dataset.x_range[0]
    x_max_global = test_tasks[-1].dataset.x_range[1]
    
    # Compute y range from true function
    x_full = np.linspace(x_min_global, x_max_global, 500)
    func = instantiate(config.dataset.func)
    y_full = func(x_full)
    y_min_global = y_full.min() - 0.1 * (y_full.max() - y_full.min())
    y_max_global = y_full.max() + 0.1 * (y_full.max() - y_full.min())
    
    for task_id in range(N):
        fig, ax_main = plt.subplots(figsize=(12, 6))
        
        # Plot saved predictions for ALL tasks (trained and future)
        # This shows how training on new tasks affects old knowledge and what future tasks look like
        for tid in sorted(task_data[task_id]['predictions_per_task'].keys()):
            pred_data = task_data[task_id]['predictions_per_task'][tid]
            task_x = pred_data['x']
            task_y_pred = pred_data['y']
            
            # Sort by x for plotting
            sort_idx = np.argsort(task_x)
            task_x = task_x[sort_idx]
            task_y_pred = task_y_pred[sort_idx]
            
            # Determine style based on task status
            if tid == task_id:
                # Current task: red solid line
                ax_main.plot(task_x, task_y_pred, '-', color='red', 
                           linewidth=2.5, zorder=4)
            elif tid < task_id:
                # Previous tasks: black solid line
                ax_main.plot(task_x, task_y_pred, '-', color='black', 
                           linewidth=2.5, zorder=4)
            else:
                # Future tasks (not yet trained): black dashed line
                ax_main.plot(task_x, task_y_pred, '--', color='black', 
                           linewidth=2.5, zorder=4)
            
            log.info(f"Visualization {task_id+1}: Plotted Task {tid+1} predictions in range [{task_y_pred.min():.4f}, {task_y_pred.max():.4f}]")
        
        # Plot the true function as reference (with transparency)
        ax_main.plot(x_full, y_full, 'k--', linewidth=1.5, alpha=0.4, 
                    label='Ground Truth', zorder=2)
        
        # Add vertical dashed lines for ALL task boundaries (not just trained ones)
        for tid in range(N):
            if hasattr(test_tasks[tid].dataset, 'x_range'):
                x_start, x_end = test_tasks[tid].dataset.x_range
                if tid > 0:  # Don't draw line before first task
                    ax_main.axvline(x_start, color='gray', linestyle='--', 
                                  linewidth=1.5, alpha=0.6, zorder=1)
        
        # Add activation range visualization
        # Previous interval limits: black
        # Current interval limits: red
        color_old = 'black'
        color_current = 'red'
        
        # Get current task data and x boundaries
        current_task_data = task_data[task_id]
        current_x_start, current_x_end = test_tasks[task_id].dataset.x_range
        
        log.info(f"Task {task_id}: Plotting intervals - old={current_task_data['old_interval'] is not None}, current={current_task_data['current_interval'] is not None}")
        
        # Show old_interval (if available) - from x_min to start of current task
        if current_task_data['old_interval']:
            min_val = current_task_data['old_interval']['min']
            max_val = current_task_data['old_interval']['max']
            
            log.info(f"  Old interval: [{min_val:.4f}, {max_val:.4f}]")
            
            # Draw old interval lines only up to current task boundary
            ax_main.hlines(min_val, x_min_global, current_x_start, 
                          color=color_old, linestyle='--', linewidth=2, zorder=3, label='Previous Interval')
            ax_main.hlines(max_val, x_min_global, current_x_start, 
                          color=color_old, linestyle='--', linewidth=2, zorder=3)
        
        # Show current_interval (if available) - from start of current task onwards
        if current_task_data['current_interval']:
            min_val = current_task_data['current_interval']['min']
            max_val = current_task_data['current_interval']['max']
            
            log.info(f"  Current interval: [{min_val:.4f}, {max_val:.4f}]")
            
            # Draw current interval lines from current task start to current task end
            ax_main.hlines(min_val, current_x_start, current_x_end, 
                          color=color_current, linestyle='--', linewidth=2, zorder=3, label='Current Interval')
            ax_main.hlines(max_val, current_x_start, current_x_end, 
                          color=color_current, linestyle='--', linewidth=2, zorder=3)
        
        # Set consistent axis ranges for all plots
        ax_main.set_xlim(x_min_global, x_max_global)
        ax_main.set_ylim(y_min_global, y_max_global)
        
        # Remove title and axis labels
        ax_main.legend(loc='best', fontsize=10, framealpha=0.95)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'task_{task_id + 1:02d}_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f'Saved visualization to {output_path}')
        plt.close()
    
    log.info(f'All visualizations saved to {output_dir}')
