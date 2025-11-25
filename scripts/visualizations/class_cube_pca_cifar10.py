#!/usr/bin/env python3
"""
Class-wise Bounding Cube Visualization for ResNet18 on CIFAR-10

This script extracts features from a pretrained ResNet18 (without the classification head),
performs PCA to reduce to 3D, and fits a bounding cube for each class. The visualization
shows how different classes occupy different regions in the feature space.

Output: Saves 3D visualization images to the outputs/ directory.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
import os


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_resnet18_feature_extractor(device):
    """Load pretrained ResNet18 without classification head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Remove the final classification layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    return feature_extractor.to(device)


def get_cifar10_loader(batch_size=64, num_samples_per_class=100):
    """Load CIFAR-10 test dataset with balanced samples per class."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    
    # Sample equally from each class
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    selected_indices = []
    for class_id in range(10):
        indices = class_indices[class_id][:num_samples_per_class]
        selected_indices.extend(indices)
    
    subset = Subset(dataset, selected_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def extract_features(model, dataloader, device):
    """
    Extract features from ResNet18 for all images.
    
    Returns:
        features: np.array of shape (N, 512)
        labels: np.array of shape (N,)
    """
    all_features = []
    all_labels = []
    
    for images, targets in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        
        with torch.no_grad():
            features = model(images)
            features = features.squeeze()  # Remove spatial dimensions
            
        all_features.append(features.cpu().numpy())
        all_labels.append(targets.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels


def apply_pca(features, n_components=3):
    """Apply PCA to reduce features to 3D."""
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_variance}")
    print(f"Total variance explained: {explained_variance.sum():.4f}")
    
    return features_pca, pca


def compute_bounding_cube(points, lower_percentile=1, upper_percentile=99):
    """
    Compute axis-aligned bounding cube for a set of points using percentiles.
    
    Args:
        points: np.array of shape (N, 3)
        lower_percentile: Lower percentile for bounds (default: 1)
        upper_percentile: Upper percentile for bounds (default: 99)
    
    Returns:
        min_corner: np.array of shape (3,) - lower percentile coordinates
        max_corner: np.array of shape (3,) - upper percentile coordinates
    """
    min_corner = np.percentile(points, lower_percentile, axis=0)
    max_corner = np.percentile(points, upper_percentile, axis=0)
    
    return min_corner, max_corner


def cube_corners_to_vertices(min_corner, max_corner):
    """
    Convert min/max corners to 8 vertices in the format expected by pytorch3d.
    
    Args:
        min_corner: np.array of shape (3,)
        max_corner: np.array of shape (3,)
    
    Returns:
        vertices: np.array of shape (8, 3)
    """
    r = [min_corner, max_corner]
    vertices = np.array([
        [r[0][0], r[0][1], r[0][2]],
        [r[1][0], r[0][1], r[0][2]],
        [r[1][0], r[1][1], r[0][2]],
        [r[0][0], r[1][1], r[0][2]],
        [r[0][0], r[0][1], r[1][2]],
        [r[1][0], r[0][1], r[1][2]],
        [r[1][0], r[1][1], r[1][2]],
        [r[0][0], r[1][1], r[1][2]]
    ])
    return vertices


def compute_3d_iou(box1_min, box1_max, box2_min, box2_max):
    """
    Compute 3D Intersection over Union (IoU) for two axis-aligned bounding boxes.
    
    Args:
        box1_min: np.array of shape (3,) - minimum corner of box 1
        box1_max: np.array of shape (3,) - maximum corner of box 1
        box2_min: np.array of shape (3,) - minimum corner of box 2
        box2_max: np.array of shape (3,) - maximum corner of box 2
    
    Returns:
        iou: float - 3D IoU value between 0 and 1
    """
    # Compute intersection box
    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    
    # Check if there's an intersection
    if np.any(inter_min >= inter_max):
        return 0.0
    
    # Compute intersection volume
    inter_dims = inter_max - inter_min
    inter_volume = np.prod(inter_dims)
    
    # Compute volumes of both boxes
    box1_dims = box1_max - box1_min
    box2_dims = box2_max - box2_min
    box1_volume = np.prod(box1_dims)
    box2_volume = np.prod(box2_dims)
    
    # Compute union volume
    union_volume = box1_volume + box2_volume - inter_volume
    
    # Compute IoU
    if union_volume == 0:
        return 0.0
    
    iou = inter_volume / union_volume
    return iou


def draw_cube(ax, min_corner, max_corner, color='blue', alpha=0.1, edge_alpha=0.6):
    """Draw a 3D cube on the given axes."""
    vertices = cube_corners_to_vertices(min_corner, max_corner)
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    # Draw faces
    cube = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidths=1.5)
    cube.set_edgecolor((*sns.color_palette([color])[0], edge_alpha))
    ax.add_collection3d(cube)


def visualize_all_classes_with_cubes(features_pca, labels, output_dir="outputs/pca_cubes"):
    """
    Visualize all classes in 3D PCA space with their bounding cubes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    colors = sns.color_palette("tab10", 10)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each class with its cube
    for class_id in range(10):
        class_mask = labels == class_id
        class_points = features_pca[class_mask]
        
        # Scatter plot
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            class_points[:, 2],
            c=[colors[class_id]],
            label=cifar10_classes[class_id],
            alpha=0.6,
            s=20
        )
        
        # Compute and draw bounding cube
        min_corner, max_corner = compute_bounding_cube(class_points)
        draw_cube(ax, min_corner, max_corner, color=colors[class_id], alpha=0.05, edge_alpha=0.4)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title('CIFAR-10 Classes in 3D PCA Space with Bounding Cubes\n(ResNet18 Features)', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    plt.tight_layout()
    
    # Save from multiple angles
    for angle in [30, 60, 120, 240]:
        ax.view_init(elev=20, azim=angle)
        output_path = os.path.join(output_dir, f"all_classes_cubes_angle{angle}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()


def visualize_single_class_cube(features_pca, labels, class_id, output_dir="outputs/pca_cubes"):
    """
    Visualize a single class with its bounding cube in detail.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    colors = sns.color_palette("tab10", 10)
    
    class_mask = labels == class_id
    class_points = features_pca[class_mask]
    other_points = features_pca[~class_mask]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot other classes in gray
    ax.scatter(
        other_points[:, 0],
        other_points[:, 1],
        other_points[:, 2],
        c='lightgray',
        alpha=0.1,
        s=10,
        label='Other classes'
    )
    
    # Plot target class
    ax.scatter(
        class_points[:, 0],
        class_points[:, 1],
        class_points[:, 2],
        c=[colors[class_id]],
        label=cifar10_classes[class_id],
        alpha=0.8,
        s=50
    )
    
    # Compute and draw bounding cube
    min_corner, max_corner = compute_bounding_cube(class_points)
    draw_cube(ax, min_corner, max_corner, color=colors[class_id], alpha=0.15, edge_alpha=0.8)
    
    # Add cube dimensions to title
    cube_size = max_corner - min_corner
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title(
        f'Bounding Cube for "{cifar10_classes[class_id]}" Class\n'
        f'Cube dimensions: [{cube_size[0]:.2f}, {cube_size[1]:.2f}, {cube_size[2]:.2f}]',
        fontsize=14
    )
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"class_{class_id}_{cifar10_classes[class_id]}_cube.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def visualize_cube_sizes(features_pca, labels, output_dir="outputs/pca_cubes"):
    """
    Visualize the volume and dimensions of bounding cubes for each class.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    colors = sns.color_palette("tab10", 10)
    
    cube_stats = []
    
    for class_id in range(10):
        class_mask = labels == class_id
        class_points = features_pca[class_mask]
        
        min_corner, max_corner = compute_bounding_cube(class_points)
        dimensions = max_corner - min_corner
        volume = np.prod(dimensions)
        
        cube_stats.append({
            'class': cifar10_classes[class_id],
            'class_id': class_id,
            'dim_pc1': dimensions[0],
            'dim_pc2': dimensions[1],
            'dim_pc3': dimensions[2],
            'volume': volume
        })
    
    # Plot cube volumes
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Volume comparison
    ax = axes[0]
    class_names = [s['class'] for s in cube_stats]
    volumes = [s['volume'] for s in cube_stats]
    class_colors = [colors[s['class_id']] for s in cube_stats]
    
    bars = ax.bar(class_names, volumes, color=class_colors, alpha=0.7)
    ax.set_ylabel('Cube Volume', fontsize=12)
    ax.set_title('Bounding Cube Volumes per Class', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Dimension comparison
    ax = axes[1]
    x = np.arange(len(class_names))
    width = 0.25
    
    dim1 = [s['dim_pc1'] for s in cube_stats]
    dim2 = [s['dim_pc2'] for s in cube_stats]
    dim3 = [s['dim_pc3'] for s in cube_stats]
    
    ax.bar(x - width, dim1, width, label='PC1', alpha=0.8)
    ax.bar(x, dim2, width, label='PC2', alpha=0.8)
    ax.bar(x + width, dim3, width, label='PC3', alpha=0.8)
    
    ax.set_ylabel('Dimension Length', fontsize=12)
    ax.set_title('Bounding Cube Dimensions per Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cube_statistics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()
    
    return cube_stats


def visualize_cube_overlaps(features_pca, labels, output_dir="outputs/pca_cubes"):
    """
    Visualize IoU (Intersection over Union) between class bounding cubes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Compute cubes for all classes
    cubes = {}
    for class_id in range(10):
        class_mask = labels == class_id
        class_points = features_pca[class_mask]
        min_corner, max_corner = compute_bounding_cube(class_points)
        cubes[class_id] = (min_corner, max_corner)
    
    # Compute 3D IoU matrix
    iou_matrix = np.zeros((10, 10))
    
    # Compute pairwise 3D IoU using our custom function
    for i in range(10):
        for j in range(10):
            if i == j:
                iou_matrix[i, j] = 1.0
            else:
                min_i, max_i = cubes[i]
                min_j, max_j = cubes[j]
                iou_matrix[i, j] = compute_3d_iou(min_i, max_i, min_j, max_j)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        iou_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        xticklabels=cifar10_classes,
        yticklabels=cifar10_classes,
        cbar_kws={'label': '3D IoU'},
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_title('Bounding Cube 3D IoU Matrix\n(Intersection over Union, 1%-99% percentile bounds)', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cube_iou_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()
    
    return iou_matrix


def main():
    """Main execution function."""
    seed_everything(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading ResNet18 feature extractor...")
    model = load_resnet18_feature_extractor(device)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_loader(batch_size=64, num_samples_per_class=100)
    
    # Extract features
    print("Extracting features...")
    features, labels = extract_features(model, dataloader, device)
    
    # Apply PCA
    print("\nApplying PCA to reduce to 3D...")
    features_pca, pca = apply_pca(features, n_components=3)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = "outputs/pca_cubes"
    
    print("\n1. All classes with cubes (multiple angles)...")
    visualize_all_classes_with_cubes(features_pca, labels, output_dir)
    
    print("\n2. Individual class cubes...")
    for class_id in range(10):
        visualize_single_class_cube(features_pca, labels, class_id, output_dir)
    
    print("\n3. Cube statistics...")
    cube_stats = visualize_cube_sizes(features_pca, labels, output_dir)
    
    print("\n4. Cube 3D IoU analysis...")
    iou_matrix = visualize_cube_overlaps(features_pca, labels, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    
    print("\nAverage IoU between different classes: {:.4f}".format(
        np.sum(iou_matrix - np.eye(10)) / (10 * 9)
    ))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for stats in cube_stats:
        print(f"{stats['class']:10s} - Volume: {stats['volume']:8.2f}, "
              f"Dims: [{stats['dim_pc1']:.2f}, {stats['dim_pc2']:.2f}, {stats['dim_pc3']:.2f}]")


if __name__ == "__main__":
    main()
