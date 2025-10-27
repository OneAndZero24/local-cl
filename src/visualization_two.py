"""
Visualization script for tracking activation changes across continual learning tasks.

This script creates a plot showing how layer activations for specific images change
as the network learns new tasks. For each task, it:
- Selects one representative image
- Records average activations when that task is current
- Records activations for all images after each subsequent task
- Plots the difference in activations across tasks

Usage:
    python src/visualization_two.py --config-name naive_split_mnist_mlp
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util.fabric import setup_fabric
from util import preprocess_config

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


def get_layer_activations(model, x):
    """
    Get average activations for each layer.
    
    Args:
        model: The model to extract activations from
        x: Input tensor
        
    Returns:
        List of average activation values (one per layer)
    """
    model.eval()
    activations = []
    
    with torch.no_grad():
        # Manually extract activations from MLP layers
        current = torch.flatten(x, start_dim=1)
        
        # Get the actual module (unwrap from Fabric wrapper if needed)
        actual_model = model._forward_module if hasattr(model, '_forward_module') else model
        
        # MLP has self.mlp which is a ModuleList of layers + activations
        if hasattr(actual_model, 'mlp'):
            for layer in actual_model.mlp:
                current = layer(current)
                # Only record from non-activation layers (Linear layers)
                # We want to record after activation functions
                if isinstance(layer, (nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU)) or \
                   (hasattr(layer, '__class__') and 'Activation' in layer.__class__.__name__):
                    avg_act = current.mean().item()
                    activations.append(avg_act)
        else:
            # Fallback: try to use the model's activation recording
            _ = actual_model(x)
            if hasattr(actual_model, 'activations') and actual_model.activations:
                for layer_activation in actual_model.activations:
                    avg_act = layer_activation.mean().item()
                    activations.append(avg_act)
    
    return activations


def visualize_activation_drift(config: DictConfig):
    """
    Create visualization for activation drift across tasks.
    
    For each task:
    - Select one representative image
    - Track how its activations change as new tasks are learned
    - Plot the differences with images displayed
    """
    
    log.info('Initializing scenarios')
    train_scenario, test_scenario = get_scenarios(config)
    
    log.info('Launching Fabric')
    fabric = setup_fabric(config)
    
    log.info('Building model')
    model = fabric.setup(instantiate(config.model))
    
    log.info('Setting up optimizer')
    # Use a simple optimizer without the complex method/plugin system
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
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
    
    # Select one image from each task
    log.info('Selecting representative images from each task')
    selected_images = []
    selected_labels = []
    
    for task_id, test_task in enumerate(test_tasks):
        # Get the first image from this task
        for X, y, _ in test_task:
            selected_images.append(X)
            selected_labels.append(y.item())
            break
    
    log.info(f'Selected {len(selected_images)} images (one per task)')
    
    # Storage for activations:
    # activation_history[task_id][image_id] = list of avg activations per layer
    activation_history = [
        [None for _ in range(N)] for _ in range(N)
    ]
    
    # Train all tasks and record activations
    for task_id, train_task in enumerate(tqdm(train_tasks, desc="Training tasks")):
        log.info(f'Task {task_id + 1}/{N}')
        
        # Check if model has IncrementalClassifier head
        if hasattr(model, 'head') and hasattr(model.head, 'increment') \
            and not config.exp.dil:
            log.info(f'Incrementing model head')
            model.head.increment(train_task.dataset.get_classes())
        
        # Train
        log.info('Training...')
        model.train()
        pbar = tqdm(range(config.exp.epochs), desc=f"Task {task_id+1} Training", leave=False)
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0
            for X, y, _ in train_task:
                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        log.info(f'Task {task_id}: Final training loss = {avg_loss:.6f}')
        
        # After training, record activations for ALL selected images
        log.info(f'Recording activations for all images after task {task_id}')
        model.eval()
        
        for img_id in range(N):
            img = selected_images[img_id]
            avg_activations = get_layer_activations(model, img)
            activation_history[task_id][img_id] = avg_activations
            log.info(f'  Task {task_id}, Image {img_id}: {len(avg_activations)} layers recorded')
    
    # Now create the visualization
    log.info('Creating activation drift visualization...')
    
    # Create output directory
    output_dir = Path(config.exp.log_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate differences from baseline (when image's task was current)
    # For image i, baseline is activation_history[i][i] (task i on image i)
    activation_differences = []
    
    for img_id in range(N):
        baseline_activations = activation_history[img_id][img_id]  # When task was current
        differences = []
        
        for task_id in range(img_id, N):  # Start from when image was introduced
            current_activations = activation_history[task_id][img_id]
            
            # Calculate average absolute difference across all layers
            diff = np.mean([abs(curr - base) for curr, base in zip(current_activations, baseline_activations)])
            differences.append(diff)
        
        activation_differences.append(differences)
    
    # Create the plot
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for each image/task
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    
    # Plot lines
    for img_id in range(N):
        x_values = list(range(img_id, N))  # Start from when image was introduced
        y_values = activation_differences[img_id]
        
        ax.plot(x_values, y_values, 
               marker='o', 
               linewidth=2.5, 
               markersize=8,
               color=colors[img_id],
               label=f'Task {img_id+1} (class {selected_labels[img_id]})')
    
    # Add images to the plot
    # Position images on the right side of the plot with colored frames
    max_diff = max([max(diffs) if len(diffs) > 0 else 0 for diffs in activation_differences])
    if max_diff == 0:
        max_diff = 1.0  # Default if no differences
    y_positions = np.linspace(0, max_diff, N)
    
    for img_id in range(N):
        # Get the image as numpy array
        img_np = selected_images[img_id].cpu().squeeze().numpy()
        
        # Handle different image formats
        if img_np.ndim == 3:
            if img_np.shape[0] == 1:  # Grayscale (1, H, W)
                img_np = img_np[0]
            elif img_np.shape[0] == 3:  # RGB (3, H, W)
                img_np = np.transpose(img_np, (1, 2, 0))
        
        # Normalize to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create image box with colored border
        imagebox = OffsetImage(img_np, zoom=2.0, cmap='gray' if img_np.ndim == 2 else None)
        
        # Position on the right side
        x_pos = N - 0.5
        y_pos = y_positions[img_id]
        
        ab = AnnotationBbox(imagebox, (x_pos, y_pos),
                          frameon=True,
                          pad=0.3,
                          bboxprops=dict(
                              edgecolor=colors[img_id],
                              linewidth=3,
                              facecolor='white'
                          ))
        ax.add_artist(ab)
    
    # Formatting
    ax.set_xlabel('Task ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Activation Difference from Baseline', fontsize=14, fontweight='bold')
    ax.set_title('Activation Drift Across Continual Learning Tasks', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Set x-axis to show task IDs
    ax.set_xticks(range(N))
    ax.set_xticklabels([f'{i}' for i in range(N)])
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'activation_drift_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f'Saved visualization to {output_path}')
    plt.close()
    
    # Also create a detailed plot showing per-layer activations
    log.info('Creating detailed per-layer activation plot...')
    
    num_layers = len(activation_history[0][0]) if activation_history[0][0] else 0
    
    if num_layers == 0:
        log.warning('No layer activations recorded. Skipping detailed per-layer plot.')
        log.info(f'Main visualization saved to {output_dir}')
        return
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), sharex=True)
    
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        for img_id in range(N):
            x_values = list(range(img_id, N))
            y_values = [activation_history[task_id][img_id][layer_idx] 
                       for task_id in range(img_id, N)]
            
            ax.plot(x_values, y_values,
                   marker='o',
                   linewidth=2.5,
                   markersize=8,
                   color=colors[img_id],
                   label=f'Task {img_id+1} (class {selected_labels[img_id]})')
        
        ax.set_ylabel(f'Layer {layer_idx+1}\nAvg Activation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_xticks(range(N))
        ax.set_xticklabels([f'{i}' for i in range(N)])
    
    axes[-1].set_xlabel('Task ID', fontsize=14, fontweight='bold')
    fig.suptitle('Per-Layer Activation Tracking Across Tasks', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_path_detailed = output_dir / 'activation_per_layer_detailed.png'
    plt.savefig(output_path_detailed, dpi=300, bbox_inches='tight')
    log.info(f'Saved detailed visualization to {output_path_detailed}')
    plt.close()
    
    log.info(f'All visualizations saved to {output_dir}')


@hydra.main(version_base=None, config_path="../config", config_name="interval_penalization_mnist_mlp_interval")
def main(config: DictConfig):
    """Main entry point for visualization script."""
    preprocess_config(config)
    visualize_activation_drift(config)


if __name__ == "__main__":
    import pyrootutils
    pyrootutils.setup_root(
        search_from=__file__,
        indicator="requirements.txt",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )
    main()
