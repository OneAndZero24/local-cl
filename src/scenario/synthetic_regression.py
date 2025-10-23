"""
Synthetic Regression Dataset for Continual Learning.

This module provides a synthetic regression dataset that can be split into
multiple tasks, where each task represents regression on a different part
of a continuous function (e.g., sine wave).
"""

import numpy as np
from torch.utils.data import Dataset
from continuum.tasks import TaskSet


class RegressionTaskSet(Dataset):
    """
    Simple wrapper for regression task data that properly handles indexing.
    """
    def __init__(self, x, y, t, x_range):
        self.x_data = x.flatten()  # Ensure 1D
        self.y_data = y.flatten()  # Ensure 1D
        self.t_data = t
        self.x_range = x_range
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # Return properly shaped single samples
        return (
            self.x_data[idx].reshape(1),  # [1] for input dimension
            self.y_data[idx].reshape(1),  # [1] for output dimension
            self.t_data[idx]
        )
    
    def get_classes(self):
        return [0]  # Dummy for compatibility


def sin_function(x):
    """Sine function for regression."""
    return np.sin(x)


def gaussian_function(x):
    """Standard Gaussian (normal) function centered at 0."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class SyntheticRegressionDataset(Dataset):
    """
    Synthetic regression dataset based on a mathematical function.
    
    Args:
        func: Callable that takes x values and returns y values (or a Hydra instantiated object)
        x_range: List/Tuple of [min, max] for the input range
        n_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise to add to targets
        seed: Random seed for reproducibility
        train: Whether this is training data (for compatibility, uses same data)
    """
    
    def __init__(self, func, x_range, n_samples=1000, noise_std=0.1, seed=42, train=True):
        # Handle Hydra instantiation - func might be the actual function or need to be called
        if callable(func):
            self.func = func
        else:
            self.func = func  # Already instantiated
            
        self.x_range = tuple(x_range) if isinstance(x_range, list) else x_range
        self.n_samples = n_samples
        self.noise_std = noise_std
        
        # Use different seed for test to get different noise realizations
        actual_seed = seed if train else seed + 1000
        
        # Generate data
        rng = np.random.RandomState(actual_seed)
        self.x = np.linspace(x_range[0], x_range[1], n_samples, dtype=np.float32)
        self.y = self.func(self.x).astype(np.float32)
        
        # Add noise
        if noise_std > 0:
            self.y += rng.normal(0, noise_std, n_samples).astype(np.float32)

        self.bounding_boxes = None
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Return single sample, not a slice
        return self.x[idx].reshape(1), self.y[idx].reshape(1), 0


class SyntheticRegressionScenario:
    """
    Scenario that splits a synthetic regression dataset into multiple tasks.
    
    Each task covers a continuous portion of the input range.
    
    Args:
        dataset: SyntheticRegressionDataset instance
        n_tasks: Number of tasks to split the data into
        overlap: Fraction of overlap between consecutive tasks (0.0 = no overlap)
    """
    
    def __init__(self, dataset, n_tasks=5, overlap=0.0):
        self.dataset = dataset
        self.n_tasks = n_tasks
        self.overlap = overlap
        
        # Calculate task boundaries
        x_min, x_max = dataset.x_range
        total_range = x_max - x_min
        
        # Calculate task size with overlap
        task_size = total_range / (n_tasks - overlap * (n_tasks - 1))
        step_size = task_size * (1 - overlap)
        
        self.task_boundaries = []
        for i in range(n_tasks):
            start = x_min + i * step_size
            end = start + task_size
            # Ensure last task reaches exactly to x_max
            if i == n_tasks - 1:
                end = x_max
            self.task_boundaries.append((start, end))
        
        self._tasks = None
        
    def _create_tasks(self):
        """Lazy initialization of tasks."""
        if self._tasks is not None:
            return
            
        self._tasks = []
        for task_id, (start, end) in enumerate(self.task_boundaries):
            # Find indices for this task
            mask = (self.dataset.x >= start) & (self.dataset.x <= end)
            indices = np.where(mask)[0]
            
            x_task = self.dataset.x[indices]
            y_task = self.dataset.y[indices]
            t_task = np.full(len(indices), task_id, dtype=np.int64)
            
            # Create custom task wrapper instead of TaskSet
            task = RegressionTaskSet(x_task, y_task, t_task, (start, end))
            
            self._tasks.append(task)
    
    def __len__(self):
        return self.n_tasks
    
    def __getitem__(self, task_index):
        self._create_tasks()
        return self._tasks[task_index]
    
    def __iter__(self):
        self._create_tasks()
        return iter(self._tasks)
