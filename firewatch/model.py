import ray
from ray.util.sgd import TorchTrainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import *
from firewatch.src.utils.multicore import RayManager
from firewatch.src.utils.experimental import SmokeClassificationDataset
from firewatch.src.utils import multicore as multi

import time


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x)


class SmokeClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_block64 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=2),
            nn.SELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128)
        )

        self.conv_block128 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256)
        )

        self.convblock256 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512)
        )

        self.conv_block512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.SELU(),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=1),
            nn.SELU(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512)  # New
        )

        self.dense_block = nn.Sequential(  # New
            nn.Flatten(),

            nn.Linear(1024, 1024),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.SELU(),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block64(x)
        x = self.conv_block128(x)
        x = self.convblock256(x)
        x = self.conv_block512(x)

        x = self.dense_block(x)

        return x


class PaddedConv2d(nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class Block(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, activation_function: Callable=nn.ReLU,
                 first_stride: int=False, residual: bool=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation_function = activation_function
        self.first_stride = first_stride
        
        self.residual = residual
        
        if not self.first_stride:
            self.first_stride = 1
        
        self.block = nn.Sequential(
                                   PaddedConv2d(self.in_features, self.out_features, kernel_size=(3, 3), stride=self.first_stride),
                                   nn.BatchNorm2d(self.out_features),
                                   self.activation_function(inplace=True),
                                   
                                   PaddedConv2d(self.out_features, self.out_features, kernel_size=(3, 3), stride=1),
                                   nn.BatchNorm2d(self.out_features),
                                   self.activation_function(inplace=True)
                                   )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                                   output = self.block(x)
                                   
                                   if self.residual:
                                   output.add_(x)
            
            output = self.activation_function(inplace=True)(output)
                
    return output


class Layer(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, n_blocks: int=2, activation_function: Callable=nn.ReLU):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_blocks = n_blocks
        self.activation_function = activation_function
        
        self.layer = nn.Sequential(
                                   *[Block(in_features=self.in_features,
                                           out_features=self.out_features,
                                           activation_function=self.activation_function,
                                           first_stride=2,
                                           residual=False) if _ < 1 else Block(
                                                                               in_features=self.out_features,
                                                                               out_features=self.out_features,
                                                                               activation_function=self.activation_function,
                                                                               residual=True
                                                                               ) for _ in range(self.n_blocks)]
                                   )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ResNet(nn.Module):
    
    RES_34_LAYER_MAP = {
        64: 3,
        128: 4,
        256: 6,
        512: 3
    }
    
    def __init__(self, in_features: int, layer_map: Dict[int, int]=None):
        super().__init__()
        
        self.in_features = in_features
        self.layer_map = layer_map
        if self.layer_map is None:
            self.layer_map = ResNet.RES_34_LAYER_MAP
        
        self.image_input = nn.Sequential(
                                         nn.Conv2d(self.in_features, 64, kernel_size=(7, 7), stride=2),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         
                                         nn.MaxPool2d(kernel_size=(3, 3), stride=2)
                                         )
            
                                         self.layers = nn.ModuleList([])
                                         for i, (features, depth) in enumerate(self.layer_map.items()):
                                             if i > 0:
                                                 self.layers.append(Layer(in_features=self.layers[i-1].out_features,
                                                                          out_features=features, n_blocks=depth))
                                                     else:
                                                         self.layers.append(Layer(in_features=features, out_features=features, n_blocks=depth))
                                         
                                                     self.output_features = nn.Sequential(
                                                                                          nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1),
                                                                                          nn.BatchNorm2d(512),
                                                                                          nn.ReLU(),
                                                                                          
                                                                                          nn.AvgPool2d((1, 1)),
                                                                                          )
                                                         
                                                         self.fc = nn.Sequential(
                                                                                 nn.Flatten(),
                                                                                 
                                                                                 nn.Linear(2048, 1024),
                                                                                 nn.SELU(),
                                                                                 nn.Linear(1024, 1),
                                                                                 nn.SELU(),
                                                                                 nn.Sigmoid()
                                                                                 )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.image_input(x)
    for layer in self.layers:
        x = layer(x)
        
        x = self.output_features(x)
        x = self.fc(x)
        
        return x


def accuracy(model: nn.Module, test_loader: DataLoader):
    with torch.no_grad():
        model.eval()

        correct = 0
        for inputs, target in test_loader:
            if round(model(inputs).item()) == round(target.item()):
                correct += 1
        return correct/len(test_loader.dataset)

