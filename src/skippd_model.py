import torch
import torch.nn.functional as F

#TODO: what are in_ find n_hidden/output in SKIPPD notebook, check if model translated correctly

class SkippdModel(torch.nn.Module):
    def __init__(self, 
        n_feature, 
        n_hidden, 
        n_output,
        num_filters: int = 24,
        kernel_size = [3,3],
        pool_size = [2,2],
        strides: int = 2,
        dense_size: int = 1024,
        drop_rate: float = 0.4
        ):
        super(SkippdModel, self).__init__()

        self.kernel_size = kernel_size
        self.strides = strides
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.dense_size = dense_size
        self.drop_rate = drop_rate 
        self.n_hidden = n_hidden
        self.n_output = n_output

    
        self.conv = torch.nn.Conv2d(in_channels = self.n_hidden, out_channels=self.n_hidden, kernel_size=self.kernel_size)
        self.relu = torch.nn.ReLU()
        self.sequential = torch.nn.Sequential()
        self.batchnorm = torch.nn.BatchNorm2d()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels =self.n_hidden, out_channels=self.n_output, kernel_size=self.kernel_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = self.kernel_size, stride=self.strides)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = self.n_hidden, out_channels = self.n_output, kernel_size=self.kernel_size),
            torch.nn.BatchNorm2d(),
            torch.nn.MaxPool2d(kernel_size = self.kernel_size, stride=self.strides)
        )

        self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(n_feature, self.dense_size)
                torch.nn.ReLU()
        )
        
        self.dropout = torch.nn.Dropout(p = self.drop_rate)

        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(n_feature, 1)
        

    def forward(self, x):
        """
        x: Tensor, shape [64,64,3*16,8] - pixels, RGB channels*time steps, PV output of 8 time steps (?)
        
        """
        x_2 = x[-1]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = torch.cat([x,x_2])(axis=1)
        x = self.dropout(self.layer3(x))
        x = self.dropout(self.layer3(x))
        x = self.linear(x)

        return x


