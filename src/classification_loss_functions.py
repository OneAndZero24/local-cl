import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class LossCriterionType(Enum):
    CROSS_ENTROPY = "CrossEntropyLoss"
    JUST_MAHA = "JustMahalanobisLoss"
    MAHALANOBIS_DISTANCE = "MahalanobisDistanceLoss"


class LossCriterion(nn.Module):
    """
    Loss function class to perform classification tasks, toggling between different loss types.

    Args:
    - criterion (str): Loss function name, either "CrossEntropyLoss" or "MahalanobisDistanceLoss".
    """

    def __init__(self, criterion: str):
        super().__init__()

        self.criterion_name = self._map_to_loss_type(criterion)

        self.loss_functions = {
            LossCriterionType.CROSS_ENTROPY: self._cross_entropy_loss,
            LossCriterionType.MAHALANOBIS_DISTANCE: self._mahalanobis_distance_loss,
            LossCriterionType.JUST_MAHA: self._just_mahalanobis_loss,
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
        elif criterion == "JustMahalanobisLoss":
            return LossCriterionType.JUST_MAHA
        else:
            raise ValueError("Invalid criterion. Use 'CrossEntropyLoss' or 'MahalanobisDistanceLoss'.")

    def compute_mahalanobis_distance(self, x: torch.Tensor, target: torch.Tensor,
                                     margin: float = 0.5, triplet: bool = True) -> torch.Tensor:
        """
        Contrastive loss using Mahalanobis distance for classification.

        Args:
        - x (torch.Tensor): Model output logits.
        - target (torch.Tensor): True class labels of shape [batch_size].
        - margin (float): Margin for separating correct class and incorrect classes.
        - triplet (bool): If True, uses triplet loss; otherwise, uses a simpler distance-based loss.

        Returns:
        - torch.Tensor: Loss value of shape [batch_size].
        """

        # Convert similarity scores to a distance-like loss
        distances = -torch.log(x)

        batch_size = distances.shape[0]

        # Get Mahalanobis distance for the correct class
        correct_class_distance = distances[torch.arange(batch_size), target]

        if not triplet:
            # If not using triplet loss, return the distance for the correct class
            return correct_class_distance

        # Compute distances for all incorrect classes
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[torch.arange(batch_size), target] = 0
        incorrect_class_distances = distances[mask].view(batch_size, -1)

        # Ensure the correct class has a much smaller Mahalanobis distance
        loss = torch.relu(margin + correct_class_distance - incorrect_class_distances.min(dim=1)[0])
        return loss.mean()

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
        loss_fn = self.loss_functions[self.criterion_name]
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
        return self.compute_mahalanobis_distance(x, target)
    
    def _just_mahalanobis_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mahalanobis distance loss without triplet.

        Args:
        - x (torch.Tensor): Input features (or embeddings) of shape [batch_size, n_features].
        - target (torch.Tensor): Class indices of shape [batch_size].

        Returns:
        - torch.Tensor: Mahalanobis distance loss.
        """

        return self.compute_mahalanobis_distance(x, target, False)
