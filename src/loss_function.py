"""Custom Loss Function based on Wimmer et al. (2019) for the UQ of tropical cyclone loss models."""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class FilteredCrossEntropyLoss(nn.Module):
    """Filtered Cross Entropy Loss Function."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the loss function."""
        super().__init__()

        assert num_classes >= 5, "Number of classes must be at least 5."
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filters = self._init_filters()

        self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')


    def _init_filters(self) -> dict:
        """Initialize the filters."""
        filters = {
            'left': torch.Tensor([0.6, 0.25, 0.15] + [0.0]*(self.num_classes-3)).to(self.device),
            'mid_left': torch.Tensor([0.25, 0.4, 0.25, 0.1] + [0.0]*(self.num_classes-4)).to(self.device),
            'mid_right': torch.Tensor([0.0]*(self.num_classes-4) + [0.1, 0.25, 0.4, 0.25]).to(self.device),
            'right': torch.Tensor([0.0]*(self.num_classes-3) + [0.15, 0.25, 0.6]).to(self.device),
            'else': torch.Tensor([0.1, 0.23, 0.34, 0.23, 0.1]).to(self.device)
        }
  
        return filters

    def apply_filter(self, target: Tensor) -> Tensor:
        """Apply weighted filter."""
        batch_size = target.size(0)
        one_hot_targets = torch.zeros(batch_size, self.num_classes).to(target.device)

        # Create masks for each condition
        mask_left = (target == 0)
        mask_mid_left = (target == 1)
        mask_mid_right = (target == self.num_classes - 2)
        mask_right = (target == self.num_classes - 1)
        mask_else = ~(mask_left | mask_mid_left | mask_mid_right | mask_right)

        # Apply filters
        one_hot_targets[mask_left] = self.filters['left']
        one_hot_targets[mask_mid_left] = self.filters['mid_left']
        one_hot_targets[mask_mid_right] = self.filters['mid_right']
        one_hot_targets[mask_right] = self.filters['right']

        # Apply 'else' filter dynamically for each target in the 'else' case
        for i in range(batch_size):
            if mask_else[i]:
                # Calculate padding for 'else' filter
                padding = (self.num_classes - len(self.filters['else'])) // 2
                # Create 'else' filter with padding
                else_filter = torch.Tensor([0.0]*padding + self.filters['else'].tolist() + [0.0]*(self.num_classes - len(self.filters['else']) - padding)).to(target.device)
                # Shift 'else' filter to center around target
                shift = int(target[i] - len(self.filters['else']) // 2)
                else_filter = torch.roll(else_filter, shifts=shift)
                one_hot_targets[i] = else_filter

        return one_hot_targets

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Forward pass of the loss."""
        # target_probs = self.apply_filter(target)
        # loss = F.cross_entropy(pred, target, reduction='mean')
        # return loss
        target_probs = self.apply_filter(target)
        log_softmax_pred = F.log_softmax(pred, dim=1)
        loss = self.loss_fn(log_softmax_pred, target_probs)
        return loss
    

# num_classes = 10
# loss_fn = FilteredCrossEntropyLoss(num_classes=num_classes)

# # Create dummy input and target tensors
# input = torch.randn(10, num_classes)  # batch size of 10, 5 classes
# # target = torch.randint(num_classes, (10,))  # batch size of 10, each target is a class index
# target = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).long()

# # Compute the loss
# loss = loss_fn(input, target)

# # Print the loss
# print(loss)
