#!/usr/bin/env python3
"""
Activation Interval Visualization for ResNet18 on CIFAR-10

This script visualizes activation landscapes for different classes on a pretrained
ResNet18 model with IntervalActivation layers. It computes and plots the intervals
learned by the model, showing how different classes activate within these intervals.

Output: Saves visualization images to the outputs/ directory.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import random
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from model.resnet18_interval_last_block import ResNet18IntervalLastBlock


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(device, num_classes=10):
    """Load ResNet18 with IntervalActivation layers."""
    model = ResNet18IntervalLastBlock(
        initial_out_features=num_classes,
        dim_hidden=512,
        interval_layer_kwargs={"lower_percentile": 0.05, "upper_percentile": 0.95},
        head_type="Normal",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def get_cifar10_loader(batch_size=32, num_samples=1000):
    """Load CIFAR-10 test dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    subset = Subset(dataset, range(min(num_samples, len(dataset))))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def collect_activations(model, dataloader, device, num_features=30):
    """
    Collect activations from IntervalActivation layers for all classes.
    
    Returns:
        dict: {layer_name: {class_id: [activations_array]}}
    """
    activation_dict = {}
    random_indices = {}
    
    # Get all interval layers
    interval_layers = []
    for name, module in model.named_modules():
        if isinstance(module, type(model.interval_l4_0_conv1)):
            interval_layers.append((name, module))
    
    print(f"Found {len(interval_layers)} interval activation layers")
    
    # Initialize storage
    for layer_name, _ in interval_layers:
        activation_dict[layer_name] = {i: [] for i in range(10)}
    
    # Collect activations
    for images, targets in tqdm(dataloader, desc="Collecting activations"):
        images = images.to(device)
        
        with torch.no_grad():
            # Forward pass
            _ = model(images)
            
            # Extract activations from each interval layer
            for layer_name, layer_module in interval_layers:
                if layer_module.curr_task_last_batch is not None:
                    activations = layer_module.curr_task_last_batch.detach().cpu()
                    
                    # Handle different tensor shapes
                    if len(activations.shape) == 4:  # Conv layers: (B, C, H, W)
                        activations = torch.mean(activations, dim=[2, 3])  # Global average pooling
                    elif len(activations.shape) == 2:  # Linear layers: (B, C)
                        pass
                    else:
                        continue
                    
                    # Sample features if needed
                    if layer_name not in random_indices:
                        num_channels = activations.shape[1]
                        k = min(num_features, num_channels)
                        random_indices[layer_name] = np.random.choice(num_channels, size=k, replace=False)
                    
                    indices = random_indices[layer_name]
                    activations = activations[:, indices].numpy()
                    
                    # Store by class
                    for i, target in enumerate(targets):
                        activation_dict[layer_name][target.item()].append(activations[i])
    
    # Convert lists to arrays
    for layer_name in activation_dict:
        for class_id in activation_dict[layer_name]:
            if len(activation_dict[layer_name][class_id]) > 0:
                activation_dict[layer_name][class_id] = np.array(activation_dict[layer_name][class_id])
    
    return activation_dict, interval_layers


def compute_intervals(activation_dict, interval_layers, lower_percentile=0.05, upper_percentile=0.95):
    """
    Compute interval bounds for each feature based on collected activations.
    
    Returns:
        dict: {layer_name: {'min': array, 'max': array}}
    """
    intervals = {}
    
    for layer_name, _ in interval_layers:
        all_activations = []
        for class_id in activation_dict[layer_name]:
            if len(activation_dict[layer_name][class_id]) > 0:
                all_activations.append(activation_dict[layer_name][class_id])
        
        if len(all_activations) > 0:
            all_activations = np.concatenate(all_activations, axis=0)  # Shape: (N, num_features)
            
            # Compute percentiles for each feature
            min_vals = np.percentile(all_activations, lower_percentile * 100, axis=0)
            max_vals = np.percentile(all_activations, upper_percentile * 100, axis=0)
            
            intervals[layer_name] = {'min': min_vals, 'max': max_vals}
    
    return intervals


def visualize_intervals_2d(activation_dict, intervals, layer_name, output_dir="outputs/interval_visualizations"):
    """
    Visualize activations as 2D heatmap with interval bounds overlaid.
    
    Shows feature index vs activation value, with interval bounds marked.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_name not in activation_dict or layer_name not in intervals:
        return
    
    palette = sns.color_palette("tab10", 10)
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Prepare data
    data_list = []
    for class_id in range(10):
        if len(activation_dict[layer_name][class_id]) > 0:
            acts = activation_dict[layer_name][class_id]  # Shape: (N, num_features)
            for feat_idx in range(acts.shape[1]):
                for val in acts[:, feat_idx]:
                    data_list.append({
                        'class': class_id,
                        'feature': feat_idx,
                        'activation': val
                    })
    
    if len(data_list) == 0:
        return
    
    df = pd.DataFrame(data_list)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot 2D histogram for each class
    num_features = df['feature'].max() + 1
    legend_elements = []
    
    for class_id in range(10):
        class_df = df[df['class'] == class_id]
        if len(class_df) > 0:
            ax.hist2d(
                class_df['feature'],
                class_df['activation'],
                bins=[num_features, 50],
                cmap=sns.light_palette(palette[class_id], as_cmap=True),
                alpha=0.3
            )
            legend_elements.append(Patch(facecolor=palette[class_id], label=cifar10_classes[class_id]))
    
    # Overlay interval bounds
    min_bounds = intervals[layer_name]['min']
    max_bounds = intervals[layer_name]['max']
    
    feature_indices = np.arange(len(min_bounds))
    ax.plot(feature_indices, min_bounds, 'r--', linewidth=2, label='Lower Bound (5th percentile)')
    ax.plot(feature_indices, max_bounds, 'b--', linewidth=2, label='Upper Bound (95th percentile)')
    
    # Fill interval region
    ax.fill_between(feature_indices, min_bounds, max_bounds, alpha=0.1, color='gray', label='Interval Region')
    
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Activation Value', fontsize=12)
    ax.set_title(f'CIFAR-10 Activation Landscape with Intervals\n{layer_name}', fontsize=14)
    
    # Add legends
    legend1 = ax.legend(handles=legend_elements, title='Class', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.add_artist(legend1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    
    # Save figure
    safe_layer_name = layer_name.replace('.', '_')
    output_path = os.path.join(output_dir, f"intervals_2d_{safe_layer_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_intervals_kde(activation_dict, intervals, layer_name, output_dir="outputs/interval_visualizations"):
    """
    Visualize activations using KDE plots with interval bounds for selected features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_name not in activation_dict or layer_name not in intervals:
        return
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get number of features
    sample_data = None
    for class_id in range(10):
        if len(activation_dict[layer_name][class_id]) > 0:
            sample_data = activation_dict[layer_name][class_id]
            break
    
    if sample_data is None:
        return
    
    num_features = sample_data.shape[1]
    features_to_plot = min(6, num_features)  # Plot up to 6 features
    selected_features = np.linspace(0, num_features - 1, features_to_plot, dtype=int)
    
    min_bounds = intervals[layer_name]['min']
    max_bounds = intervals[layer_name]['max']
    
    for feat_idx in selected_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot KDE for each class
        for class_id in range(10):
            if len(activation_dict[layer_name][class_id]) > 0:
                acts = activation_dict[layer_name][class_id][:, feat_idx]
                ax = sns.kdeplot(data=acts, label=cifar10_classes[class_id], fill=True, alpha=0.4, ax=ax)
        
        # Add interval bounds
        ax.axvline(min_bounds[feat_idx], color='red', linestyle='--', linewidth=2, label='Lower Bound')
        ax.axvline(max_bounds[feat_idx], color='blue', linestyle='--', linewidth=2, label='Upper Bound')
        
        # Shade interval region
        ax.axvspan(min_bounds[feat_idx], max_bounds[feat_idx], alpha=0.1, color='gray', label='Interval')
        
        ax.set_xlabel('Activation Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'CIFAR-10 Activation Distribution with Intervals\n{layer_name} - Feature {feat_idx}', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        safe_layer_name = layer_name.replace('.', '_')
        output_path = os.path.join(output_dir, f"intervals_kde_{safe_layer_name}_feat{feat_idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def visualize_interval_coverage(activation_dict, intervals, layer_name, output_dir="outputs/interval_visualizations"):
    """
    Visualize what percentage of activations fall within the learned intervals per class.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if layer_name not in activation_dict or layer_name not in intervals:
        return
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    min_bounds = intervals[layer_name]['min']
    max_bounds = intervals[layer_name]['max']
    
    coverage_data = []
    
    for class_id in range(10):
        if len(activation_dict[layer_name][class_id]) > 0:
            acts = activation_dict[layer_name][class_id]  # Shape: (N, num_features)
            
            # Check if activations are within bounds
            within_lower = acts >= min_bounds
            within_upper = acts <= max_bounds
            within_interval = within_lower & within_upper
            
            # Calculate percentage within interval
            coverage_pct = (within_interval.sum() / within_interval.size) * 100
            coverage_data.append({
                'class': cifar10_classes[class_id],
                'coverage': coverage_pct
            })
    
    if len(coverage_data) == 0:
        return
    
    df = pd.DataFrame(coverage_data)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['class'], df['coverage'], color=sns.color_palette("tab10", 10))
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title(f'Percentage of Activations Within Learned Intervals\n{layer_name}', fontsize=14)
    ax.set_ylim(0, 100)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    safe_layer_name = layer_name.replace('.', '_')
    output_path = os.path.join(output_dir, f"interval_coverage_{safe_layer_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    seed_everything(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading ResNet18 with IntervalActivation layers...")
    model = load_model(device)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_loader(batch_size=32, num_samples=1000)
    
    # Collect activations
    print("Collecting activations from interval layers...")
    activation_dict, interval_layers = collect_activations(model, dataloader, device, num_features=30)
    
    # Compute intervals
    print("Computing interval bounds...")
    intervals = compute_intervals(activation_dict, interval_layers)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = "outputs/interval_visualizations"
    
    for layer_name, _ in interval_layers:
        print(f"\nProcessing layer: {layer_name}")
        
        # 2D heatmap with intervals
        visualize_intervals_2d(activation_dict, intervals, layer_name, output_dir)
        
        # KDE plots with intervals (for selected features)
        visualize_intervals_kde(activation_dict, intervals, layer_name, output_dir)
        
        # Coverage analysis
        visualize_interval_coverage(activation_dict, intervals, layer_name, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
