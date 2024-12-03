import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models.dynamic_modules import DynamicModule


class MultiTaskModule(DynamicModule):
    """
    Multi-task modules are `torch.nn.Modules`s for multi-task
    scenarios. The `forward` method accepts task labels, one for
    each sample in the mini-batch.
    By default the `forward` method splits the mini-batch by task
    and calls `forward_single_task`. Subclasses must implement
    `forward_single_task` or override `forward.
    if `task_labels == None`, the output is computed in parallel
    for each task.
    """

    def __init__(self):
        super().__init__()
        self.known_train_tasks_labels = set()
        """ Set of task labels encountered up to now. """

    def train_adaptation(self, dataset: AvalancheDataset = None):
        """Update known task labels."""
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]
        self.known_train_tasks_labels = self.known_train_tasks_labels.union(set(task_labels))

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        """compute the output given the input `x` and task labels.
        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """
        if task_labels is None:
            return self.forward_all_tasks(x)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:], device=out_task.device)
            out[task_mask] = out_task
        return out

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        """compute the output given the input `x` and task label.
        :param x:
        :param task_label: a single task label.
        :return:
        """
        raise NotImplementedError()

    def forward_all_tasks(self, x: torch.Tensor):
        """compute the output given the input `x` and task label.
        By default, it considers only tasks seen at training time.
        :param x:
        :return: all the possible outputs are returned as a dictionary
            with task IDs as keys and the output of the corresponding
            task as output.
        """
        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x, task_id)
        return res
