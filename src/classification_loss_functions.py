import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class LossCriterionType(Enum):
    CROSS_ENTROPY = "CrossEntropyLoss"
    MAHALANOBIS_DISTANCE = "MahalanobisDistanceLoss"


class LossCriterion(nn.Module):
    """
    Loss function class to perform classification tasks, toggling between different loss types.

    Args:
    - criterion (str): Loss function name, either "CrossEntropyLoss" or "MahalanobisDistanceLoss".
    """

    def __init__(self, criterion: str):
        super().__init__()

        self.criterion = self._map_to_loss_type(criterion)

        self.class_means = None
        self.class_var_inv = None

        self.loss_functions = {
            LossCriterionType.CROSS_ENTROPY: self._cross_entropy_loss,
            LossCriterionType.MAHALANOBIS_DISTANCE: self._mahalanobis_distance_loss,
        }

    def _map_to_loss_type(self, criterion: str) -> LossCriterionType:
        """
        Convert a string criterion to the corresponding LossCriterionType Enum.

        Args:
        - criterion (str): Loss function name, either "CrossEntropyLoss" or "MahalanobisDistanceLoss".

        Returns:
        - LossCriterionType: Corresponding enum value.
        """
        if criterion == "CrossEntropyLoss":
            return LossCriterionType.CROSS_ENTROPY
        elif criterion == "MahalanobisDistanceLoss":
            return LossCriterionType.MAHALANOBIS_DISTANCE
        else:
            raise ValueError("Invalid criterion. Use 'CrossEntropyLoss' or 'MahalanobisDistanceLoss'.")

    def set_class_statistics(self, means: dict, variances_inv: dict):
        """
        Set class means and inverse variances (for Mahalanobis distance computation).

        Args:
        - means (dict): A dictionary with class IDs as keys and means as values.
        - variances_inv (dict): A dictionary with class IDs as keys and inverse variances as values.
        """
        self.class_means = means
        self.class_var_inv = variances_inv

    def compute_mahalanobis_distance(self, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance for each sample in the batch.

        Args:
        - x (torch.Tensor): Input features of shape [batch_size, n_features].
        - class_idx (torch.Tensor): Class indices of shape [batch_size].

        Returns:
        - torch.Tensor: Mahalanobis distances of shape [batch_size].
        """
        means = torch.stack([self.class_means[c.item()] for c in class_idx])
        inv_vars = torch.stack([self.class_var_inv[c.item()] for c in class_idx])

        diff = x - means
        mahalanobis_dist = torch.sqrt(torch.sum(diff**2 * inv_vars, dim=1))

        return mahalanobis_dist

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss based on the chosen criterion (either CrossEntropyLoss or MahalanobisDistanceLoss).

        Args:
        - x (torch.Tensor): Model output logits or input features.
        - target (torch.Tensor): True class labels of shape [batch_size].

        Returns:
        - torch.Tensor: Computed loss.
        """
        # Call the appropriate loss function based on the selected criterion
        loss_fn = self.loss_functions[self.criterion]
        return loss_fn(x, target)

    def _cross_entropy_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Cross Entropy loss.

        Args:
        - x (torch.Tensor): Model output logits of shape [batch_size, num_classes].
        - target (torch.Tensor): True class labels of shape [batch_size].

        Returns:
        - Tensor: CrossEntropy loss.
        """
        return F.cross_entropy(x, target)

    def _mahalanobis_distance_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mahalanobis distance-based loss.

        Args:
        - x (torch.Tensor): Input features (or embeddings) of shape [batch_size, n_features].
        - target (torch.Tensor): Class indices of shape [batch_size].

        Returns:
        - torch.Tensor: Mahalanobis distance loss.
        """
        return self.compute_mahalanobis_distance(x, target).mean()
