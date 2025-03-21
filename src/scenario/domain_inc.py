from typing import Union

from continuum.tasks import TaskSet
from continuum import ClassIncremental


class DomainIncremental(ClassIncremental):
    """    
    DomainIncremental is a subclass of ClassIncremental designed to handle 
    domain-incremental learning scenarios. In such scenarios, the tasks 
    share the same set of classes, but the data distribution or domain 
    changes across tasks.
    This class provides functionality to retrieve tasks by their unique 
    index, supporting both single task retrieval and slicing. It also 
    ensures compatibility with transformations applied to the data, 
    raising errors when multiple tasks are selected with differing 
    transformations.
    
    Attributes:
        trsf (Union[Callable, List[Callable]]): Transformation(s) applied 
            to the data. Can be a single transformation or a list of 
            transformations, one per task.
        cl_dataset (object): The continual learning dataset containing 
            data type and bounding box information.

    Methods:
        __getitem__(task_index: Union[int, slice]):
            Retrieves a task or a slice of tasks by their unique index. 
            Returns a TaskSet object containing the selected data, 
            transformations, and metadata.

    Raises:
        ValueError: If a slice of tasks is selected while having different 
            transformations per task.
    """


    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        return TaskSet(
            x,
            y,
            t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            target_trsf=(lambda label: label%(self.increments[task_index])),   # changed here
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes,
            data_indexes=data_indexes,
        )