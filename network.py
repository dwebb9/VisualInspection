import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets

class CNN(nn.Module):
    def __init__(self, dataset):
        super(CNN, self).__init__()
        x, y = dataset[0]
        c, h, w = x.size()
        output = 3
        init = "orthogonal"
        
        # TODO : make sure it outputs the correct size!
        self.net = nn.Sequential( #3x28x28
            nn.Conv2d(c, 40, (9, 9), padding=(4,4)), #40x28x28
            nn.ReLU(),
            nn.Conv2d(40, 60, (9, 9), padding=(4,4)), #60x28x28
            nn.ReLU(),
            nn.Conv2d(60, 80, (9, 9), padding=(3,3)), #80x26x26
            nn.ReLU(),
            nn.Conv2d(80, 60, (9, 9), padding=(3,3)), #60x24x24
            nn.ReLU(),
            nn.Conv2d(60, 40, (5, 5), padding=(1,1)), #40x22x22
            nn.ReLU(),
            nn.Conv2d(40, 20, (5, 5), padding=(1,1)), #40x20x20
            nn.ReLU(),
            nn.Conv2d(20, output, (20,20), padding=(0,0)) #10x1x1
        )
    
    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)
