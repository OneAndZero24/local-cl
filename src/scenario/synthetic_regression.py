"""
Synthetic Regression Dataset for Continual Learning.

This module provides a synthetic regression dataset that can be split into
multiple tasks, where each task represents regression on a different part
of a continuous function (e.g., sine wave).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from continuum.tasks import TaskSet


class SyntheticRegressionDataset(Dataset):
    """
    Synthetic regression dataset based on a mathematical function.
    
    Args:
        func: Callable that takes x values and returns y values
        x_range: Tuple of (min, max) for the input range
        n_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise to add to targets
        seed: Random seed for reproducibility
    """
    
    def __init__(self, func, x_range, n_samples=1000, noise_std=0.1, seed=42):
        self.func = func
        self.x_range = x_range
        self.n_samples = n_samples
        self.noise_std = noise_std
        
        # Generate data
        rng = np.random.RandomState(seed)
        self.x = np.linspace(x_range[0], x_range[1], n_samples, dtype=np.float32)
        self.y = func(self.x).astype(np.float32)
        
        # Add noise
        if noise_std > 0:
            self.y += rng.normal(0, noise_std, n_samples).astype(np.float32)
        
        self.data_type = "numpy"
        self.bounding_boxes = None
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.x[idx:idx+1], self.y[idx:idx+1], 0


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
            
            # Create TaskSet
            task = TaskSet(
                x_task.reshape(-1, 1),  # [n_samples, 1]
                y_task.reshape(-1, 1),  # [n_samples, 1]
                t_task,
                trsf=None,
                target_trsf=None,
                data_type="numpy",
            )
            
            # Add custom attributes for visualization
            task.x_range = (start, end)
            task.get_classes = lambda: [0]  # Dummy for compatibility
            
            self._tasks.append(task)
    
    def __len__(self):
        return self.n_tasks
    
    def __getitem__(self, task_index):
        self._create_tasks()
        return self._tasks[task_index]
    
    def __iter__(self):
        self._create_tasks()
        return iter(self._tasks)


def get_sin_regression_scenario(x_max=5*np.pi, n_tasks=5, n_samples=1000, 
                                 noise_std=0.1, overlap=0.0, seed=42):
    """
    Factory function for creating a sine wave regression scenario.
    
    Args:
        x_max: Maximum x value (in multiples of pi)
        n_tasks: Number of tasks to split into
        n_samples: Total number of samples
        noise_std: Standard deviation of noise
        overlap: Fraction of overlap between tasks
        seed: Random seed
        
    Returns:
        Callable that takes train parameter and returns scenario
    """
    
    def scenario_fn(train=True):
        # Sine function
        func = lambda x: np.sin(x)
        
        # Create dataset
        dataset = SyntheticRegressionDataset(
            func=func,
            x_range=(0, x_max),
            n_samples=n_samples,
            noise_std=noise_std,
            seed=seed if train else seed + 1
        )
        
        # Create scenario
        return SyntheticRegressionScenario(dataset, n_tasks=n_tasks, overlap=overlap)
    
    return scenario_fn
