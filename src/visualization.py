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
from matplotlib.gridspec import GridSpec

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
    
    # Train and visualize each task
    for task_id, train_task in enumerate(train_tasks):
        log.info(f'Task {task_id + 1}/{N}')
        
        # Setup task
        method.setup_task(task_id)
        
        # Train
        log.info('Training...')
        method.module.train()
        for epoch in range(config.exp.epochs):
            for X, y, _ in train_task:
                loss, preds = method.forward(X, y, task_id)
                loss = loss.mean()
                method.backward(loss)
        
        # Collect interval activation ranges
        interval_layers = []
        for module in method.module.modules():
            if isinstance(module, IntervalActivation):
                if module.min is not None and module.max is not None:
                    interval_layers.append({
                        'min': module.min.cpu().numpy(),
                        'max': module.max.cpu().numpy(),
                    })
        
        # Create visualization for this task
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # Main plot: predictions vs ground truth
        ax_main = fig.add_subplot(gs[0])
        
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
                    task_x.append(X.cpu().numpy())
                    task_y_true.append(y.cpu().numpy())
                    task_y_pred.append(preds.cpu().numpy())
                
                task_x = np.concatenate(task_x).flatten()
                task_y_true = np.concatenate(task_y_true).flatten()
                task_y_pred = np.concatenate(task_y_pred).flatten()
                
                # Sort by x for plotting
                sort_idx = np.argsort(task_x)
                task_x = task_x[sort_idx]
                task_y_true = task_y_true[sort_idx]
                task_y_pred = task_y_pred[sort_idx]
                
                all_x.extend(task_x)
                all_y_true.extend(task_y_true)
                all_y_pred.extend(task_y_pred)
                
                # Plot ground truth
                ax_main.plot(task_x, task_y_true, 'o', color=colors[tid], 
                           alpha=0.3, markersize=3, label=f'Task {tid+1} (True)' if tid == task_id else None)
                
                # Plot predictions
                ax_main.plot(task_x, task_y_pred, '-', color=colors[tid], 
                           linewidth=2, label=f'Task {tid+1} (Pred)', alpha=0.8)
        
        # Highlight task boundaries
        for tid in range(task_id + 1):
            if hasattr(test_tasks[tid].dataset, 'x_range'):
                x_start, x_end = test_tasks[tid].dataset.x_range
                ax_main.axvspan(x_start, x_end, alpha=0.1, color=colors[tid])
        
        ax_main.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Output (y)', fontsize=12, fontweight='bold')
        ax_main.set_title(f'Regression Performance after Task {task_id + 1}', 
                         fontsize=14, fontweight='bold', pad=20)
        ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Interval activation visualization
        if interval_layers:
            ax_interval = fig.add_subplot(gs[1])
            
            # Plot interval ranges for each neuron in the first layer
            layer_intervals = interval_layers[0]  # First IntervalActivation layer
            n_neurons = len(layer_intervals['min'])
            
            neuron_indices = np.arange(n_neurons)
            mins = layer_intervals['min']
            maxs = layer_intervals['max']
            centers = (mins + maxs) / 2
            ranges = maxs - mins
            
            # Plot as horizontal bars
            ax_interval.barh(neuron_indices, ranges, left=mins, 
                           height=0.8, color=colors[task_id], alpha=0.6, 
                           edgecolor='black', linewidth=0.5)
            
            ax_interval.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
            ax_interval.set_ylabel('Neuron Index', fontsize=12, fontweight='bold')
            ax_interval.set_title(f'Interval Activation Ranges (Layer 1)', 
                                fontsize=12, fontweight='bold')
            ax_interval.grid(True, alpha=0.3, linestyle='--', axis='x')
            
            # Limit y-axis to show at most 20 neurons
            if n_neurons > 20:
                ax_interval.set_ylim(-0.5, 19.5)
                ax_interval.set_yticks(range(20))
        
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
                task_x.append(X.cpu().numpy())
                task_y_true.append(y.cpu().numpy())
                task_y_pred.append(preds.cpu().numpy())
            
            task_x = np.concatenate(task_x).flatten()
            task_y_true = np.concatenate(task_y_true).flatten()
            task_y_pred = np.concatenate(task_y_pred).flatten()
            
            # Sort by x
            sort_idx = np.argsort(task_x)
            task_x = task_x[sort_idx]
            task_y_true = task_y_true[sort_idx]
            task_y_pred = task_y_pred[sort_idx]
            
            # Plot
            ax.plot(task_x, task_y_true, 'o', color=colors[tid], 
                   alpha=0.3, markersize=3)
            ax.plot(task_x, task_y_pred, '-', color=colors[tid], 
                   linewidth=2.5, label=f'Task {tid+1}', alpha=0.8)
            
            # Task boundary
            if hasattr(test_tasks[tid].dataset, 'x_range'):
                x_start, x_end = test_tasks[tid].dataset.x_range
                ax.axvspan(x_start, x_end, alpha=0.1, color=colors[tid])
    
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
