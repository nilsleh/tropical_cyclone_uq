import torch
import torch.nn.functional as F

#TODO: what are in_ find n_hidden/output in SKIPPD notebook, check if model translated correctly

#import pdb

#pdb.set_trace()

class SkippdModel(torch.nn.Module):
    def __init__(self, 
        input_dim: int = 24,  
        num_classes: int=1,      
        num_filters: int = 24,
        kernel_size: int = 3,
        pool_size: int=2,
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
        self.input_dim = input_dim
        self.num_classes = num_classes


        #check dimensions out/in_channels

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim, kernel_size=self.kernel_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = self.pool_size, stride=self.strides)
        )

        #check dimensions out/in_channels

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.input_dim, out_channels=2*self.input_dim, kernel_size=self.kernel_size),
            torch.nn.BatchNorm2d(num_features = int(96)),
            torch.nn.MaxPool2d(kernel_size = self.pool_size, stride=self.strides)
        )

        self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(in_features=int(12304), out_features=int(1024)),
                torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
                torch.nn.Linear(in_features=int(1024), out_features=int(1024)),
                torch.nn.ReLU()
        )
        
        self.dropout = torch.nn.Dropout(p = self.drop_rate)

        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=int(1024), out_features=self.num_classes)
        

    def forward(self, x):
        """
        x: Tensor, shape [64,64,3*16,8] - pixels, RGB channels*time steps, PV output of 8 time steps (?)
        
        """
        x_2 = x[-1]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = torch.cat((x,x_2),axis=1)
        x = self.dropout(self.layer3(x))
        x = self.dropout(self.layer4(x))
        x = self.linear(x)

        return x


