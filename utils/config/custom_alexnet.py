import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla=None,image_size=None):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.image_size = image_size
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        layers = []
        layers.append(nn.Linear(2048,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid()) 

        self.fc3 = nn.Sequential(*layers)
        
        # self.taskcla = taskcla
        # self.fc3=torch.nn.ModuleList()
        # for t,n in self.taskcla:
        #     self.fc3.append(torch.nn.Linear(2048,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        if self.image_size is not None:
            x = x.reshape(self.image_size)
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        # y=[]
        # for t,i in self.taskcla:
        #     y.append(self.fc3[t](x))
        self.act['last_layer_activation'] = x

        y = self.fc3(x)
            
        return y

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

