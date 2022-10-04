import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential( #input shape(1,200,200)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ), #output shape(16,200,200)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) #shape(16,100,100)
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) #shape(32,50,50)
        self.out=nn.Linear(32*50*50,2)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(1,32*50*50)
        output=self.out(x)
        m=nn.Softmax()
        output1=m(output)
        return output1
    
cnn=CNN()
print(cnn)