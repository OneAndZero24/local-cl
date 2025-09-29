import random
import torch

from continuum import ClassIncremental


def get_permuted_class_inc_scenario(nb_classes, increment, seed=None, **kwargs):
    """
    Generates a scenario for class-incremental learning with a permuted class order.
    This function creates a scenario where the classes in the dataset are presented
    incrementally in a permuted order. The permutation is determined by a random seed,
    which can be specified or generated automatically.
    Args:
        nb_classes (int): The total number of classes in the dataset.
        increment (int): The number of classes to include in each incremental step.
        seed (int, optional): The random seed used to generate the class order. If None,
            a random seed is generated automatically.
        **kwargs: Additional keyword arguments passed to the `ClassIncremental` constructor.
    Returns:
        function: A scenario function that takes a dataset as input and returns a
        `ClassIncremental` object configured with the permuted class order.
    Example:
        scenario_fn = get_permuted_class_inc_scenario(increment=5, seed=42)
        scenario = scenario_fn(dataset)
    """

    if seed is None:
        seed = random.randint(0, 10000)

    rng = torch.Generator()
    rng.manual_seed(seed)
    class_order = torch.randperm(nb_classes, generator=rng).tolist()

    def scenario(dataset):
        return ClassIncremental(dataset, increment=increment, class_order=class_order, **kwargs)
    return scenario