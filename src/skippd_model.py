from typing import Tuple
import torch
from torch import nn

class SkippdModel(nn.Module):
    def __init__(self, 
            in_chans: int = 48,
            num_classes: int=1,
            num_filters: int = 24, 
            kernel_size: int = 3, 
            pool_size: int = 2, 
            strides: int = 2, 
            dense_size: int = 1024, 
            drop_rate: float = 0.4) -> None:
        
        """Initialize a new instance of Skippd Model.
        
        Args:
            in_chans: int, number of input channels
            num_filters: int, number of filters in the first convolutional layer
            kernel_size: int, size of the convolutional kernel
            pool_size: int, size of the pooling kernel
            strides: int, stride of the pooling kernel
            dense_size: int, size of the dense layer
            drop_rate: float, dropout rate
        """
        super(SkippdModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, num_filters, kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, strides)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size, padding='same'),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, strides)
        )

        self.flatten = nn.Flatten()

        # Calculate the output size of the Flatten layer
        conv_out_size: int = num_filters*2 * (64 // (pool_size*strides))**2  # assuming the input size is (64, 64)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, dense_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(dense_size, dense_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(dense_size, num_classes)
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_in: torch.Tensor, shape [batch_size, in_chans, 64, 64]
        """
        x: torch.Tensor = self.conv1(x_in)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.fc(x)

# x = torch.randn(1, 48, 64, 64)
# model = SkippdModel()
# out = model(x)

# import pdb