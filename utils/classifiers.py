import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torchvision.models import resnet18
from collections import OrderedDict
from torch.nn import Module,Linear,ReLU,Sigmoid
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

import math
from utils.config.configurations import cfg


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

# class ANOSHIFT_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,500,150,50,10,1]
#         super(ANOSHIFT_FC, self).__init__()
#         layers = []

#         for i in range(0, len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < (len(sizes) - 2):
#                 layers.append(nn.ReLU())
#         layers.append(nn.Sigmoid())        

#         self.net = nn.Sequential(*layers)
#         self.net.apply(Xavier)

#     def forward(self, x):
#         return self.net(x)        

# inputsize = 1159
class APIGRAPH_FC(nn.Module):
    def __init__(self, inputsize, softmax=False):
        super(APIGRAPH_FC, self).__init__()

        self.softmax=softmax
        self.act=OrderedDict()

        self.hidden1 = Linear(inputsize,512,bias=False)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()

        self.hidden2 = Linear(512, 384,bias=False)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()

        self.hidden3 = Linear(384, 256,bias=False)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()

        self.hidden4 = Linear(256, 128,bias=False)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()

        self.hidden5 = Linear(128, 100,bias=False)
        xavier_uniform_(self.hidden5.weight)
        self.act5 = ReLU()

        self.hidden6 = Linear(100, 100,bias=False)
        xavier_uniform_(self.hidden6.weight)
        self.act6 = ReLU()

        self.hidden7 = Linear(100, 2,bias=False)
        xavier_uniform_(self.hidden7.weight)
        self.act7 = ReLU()


    def forward(self, X):
        X = X.float()

        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.act1(X)  

        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.act2(X) 

        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.act3(X) 

        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.act4(X) 

        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.act5(X) 

        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.act6(X) 

        self.act['hidden7']= X
        X = self.hidden7(X)
        X = self.act7(X) 

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()

        return X

# inputsize = 16978
class ANDROZOO_FC(nn.Module):
    def __init__(self, inputsize, softmax=False):
        super(ANDROZOO_FC, self).__init__()

        self.softmax=softmax
        self.act=OrderedDict()

        self.hidden1 = Linear(inputsize,512,bias=False)
        xavier_uniform_(self.hidden1.weight)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.act1 = ReLU()

        self.hidden2 = Linear(512, 384,bias=False)
        xavier_uniform_(self.hidden2.weight)
        self.bn2 = nn.BatchNorm1d(384)
        self.dropout2 = nn.Dropout(0.2)
        self.act2 = ReLU()

        self.hidden3 = Linear(384, 256,bias=False)
        xavier_uniform_(self.hidden3.weight)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)
        self.act3 = ReLU()

        self.hidden4 = Linear(256, 128,bias=False)
        xavier_uniform_(self.hidden4.weight)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)
        self.act4 = ReLU()

        self.hidden5 = Linear(128, 100,bias=False)
        xavier_uniform_(self.hidden5.weight)
        self.bn5 = nn.BatchNorm1d(100)
        self.dropout5 = nn.Dropout(0.2)
        self.act5 = ReLU()

        self.hidden6 = Linear(100, 100,bias=False)
        xavier_uniform_(self.hidden6.weight)
        self.bn6 = nn.BatchNorm1d(100)
        self.dropout6 = nn.Dropout(0.2)
        self.act6 = ReLU()

        self.hidden7 = Linear(100, 2,bias=False)
        xavier_uniform_(self.hidden7.weight)
        self.bn7 = nn.BatchNorm1d(2)
        self.dropout7 = nn.Dropout(0.2)
        self.act7 = ReLU()


    def forward(self, X):
        X = X.float()

        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(X)
        # X = self.act1(self.dropout1(X))
        

        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.bn2(X)
        # X = self.act2(self.dropout2(X))
        X = self.act2(X) 

        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.bn3(X)
        # X = self.act3(self.dropout3(X))
        X = self.act3(X) 

        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.bn4(X)
        # X = self.act4(self.dropout4(X))
        X = self.act4(X) 

        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.bn6(X)
        # X = self.act5(self.dropout5(X))
        X = self.act5(X) 

        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.bn6(X)
        # X = self.act6(self.dropout6(X))
        X = self.act6(X) 

        self.act['hidden7']= X
        X = self.hidden7(X)
        X = self.bn7(X)
        # X = self.act7(self.dropout7(X))
        X = self.act7(X) 

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()

        return X 

class ANOSHIFT_FC(nn.Module):
    def __init__(self,inputsize, softmax=False):
        sizes = [inputsize,100,500,150,50,10,1]
        super(ANOSHIFT_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        # #architecture with 134810
        drop_out=0.2
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 500,bias=False)
        self.bn2 = nn.BatchNorm1d(500)
        self.dropout2 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(500, 150,bias=False)
        self.bn3 = nn.BatchNorm1d(150)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.dropout3 = nn.Dropout(drop_out)
        self.act3 = ReLU()
        self.hidden4 = Linear(150, 50,bias=False)
        self.bn4 = nn.BatchNorm1d(50)
        self.dropout4 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(50, 10,bias=False)
        self.bn5 = nn.BatchNorm1d(10)
        self.dropout5 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden5.weight)
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(10,2,bias=False)
        # self.dropout6 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden6.weight,nonlinearity='relu')
        # xavier_uniform_(self.hidden6.weight)
        # self.double()
        # self.act6 = Sigmoid()
        self.act6 = ReLU()

        
    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        # X = self.hidden1(self.dropout1(X))
        X = self.hidden1(X)
        X = self.bn1(X)
        # X = self.act1(X)
        X = self.act1(self.dropout1(X))
        # second hidden layer
        self.act['hidden2']= X
        # X = self.hidden2(self.dropout2(X))
        X = self.hidden2((X))
        X = self.bn2(X)
        # X = self.act2(X)
        X = self.act2(self.dropout2(X))
        # third hidden layer and output
        self.act['hidden3']= X
        # # X = self.hidden3(self.dropout3(X))
        X = self.hidden3((X))
        X = self.bn3(X)
        # X = self.act3(X)
        X = self.act3(self.dropout3(X))
        # # third hidden layer and output
        self.act['hidden4']= X
        # # X = self.hidden4(self.dropout4(X))
        X = self.hidden4((X))
        X = self.bn4(X)
        # X = self.act4(X)
        X = self.act4(self.dropout4(X))
        # # third hidden layer and output
        self.act['hidden5']= X
        # # X = self.hidden5(self.dropout5(X))
        X = self.hidden5((X))
        X = self.bn5(X)
        # X = self.act5(X)
        X = self.act5(self.dropout5(X))
        # third hidden layer and output
        self.act['hidden6']= X
        # X = self.hidden6(self.dropout6(X))
        X = self.hidden6((X))
        X = self.act6(X)
        # X = self.act6(self.dropout6(X))

        # X = torch.softmax(X, dim=1)

        # self.act['hidden7']= X
        # X = self.hidden7(X)
        # X = self.act7(X)
        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()
        
        return X           

class ANOSHIFT_FC_SUPERVISED(nn.Module):
    def __init__(self,inputsize,softmax=False):
        sizes = [inputsize,100,500,150,50,10,1]
        super(ANOSHIFT_FC_SUPERVISED, self).__init__()
        layers = []
        self.softmax=softmax
        
        # #architecture with 134810
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.dropout1 = nn.Dropout(0.2)
        # kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 150,bias=False)
        self.dropout1 = nn.Dropout(0.2)
        # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # self.hidden3 = Linear(500, 150,bias=False)
        # # kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        # self.act3 = ReLU()
        self.hidden4 = Linear(150, 50,bias=False)
        # kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(50, 10,bias=False)
        # kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden5.weight)
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(10,2,bias=False)
        # kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
        xavier_uniform_(self.hidden6.weight)
        # self.double()
        # self.act6 = Sigmoid()
        self.act6 = ReLU()

        
    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        # self.act['hidden3']= X
        # X = self.hidden3(X)
        # X = self.act3(X)
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.act5(X)
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.act6(X)

        # X = torch.softmax(X, dim=1)

        # self.act['hidden7']= X
        # X = self.hidden7(X)
        # X = self.act7(X)

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()
        
        return X           

class ANOSHIFT_FC_STUDENT2(nn.Module):
    def __init__(self,inputsize,softmax=False):
        sizes = [inputsize,100,500,150,50,10,1]
        super(ANOSHIFT_FC_STUDENT2, self).__init__()
        layers = []
        self.softmax = softmax
        
        # #architecture with 134810
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        # kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 500,bias=False)
        # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(500, 150,bias=False)
        # # kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(150, 50,bias=False)
        # kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(50, 10,bias=False)
        # kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden5.weight)
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(10,2,bias=False)
        # kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
        xavier_uniform_(self.hidden6.weight)
        # self.double()
        # self.act6 = Sigmoid()
        self.act6 = ReLU()

        
    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.act3(X)
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.act5(X)
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.act6(X)

        # X = torch.softmax(X, dim=1)

        # self.act['hidden7']= X
        # X = self.hidden7(X)
        # X = self.act7(X)

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()
        
        return X           



# class CICIDS2017_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,250,500,150,50,1]
#         super(CICIDS2017_FC, self).__init__()
#         layers = []

#         for i in range(0, len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < (len(sizes) - 2):
#                 layers.append(nn.ReLU())
#         layers.append(nn.Sigmoid())        

#         self.net = nn.Sequential(*layers)
#         self.net.apply(Xavier)

#     def forward(self, x):
#         return self.net(x)

class CICIDS2017_FC(nn.Module):
    def __init__(self,inputsize, softmax=False):
        sizes = [inputsize,100,250,500,150,50,1]
        super(CICIDS2017_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden5.weight)
        self.act5 = ReLU()
        # third hidden layer and output
        # self.hidden6 = Linear(50,1,bias=False)
        # kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
        # #self.double()
        # self.act6 = Sigmoid()
        self.hidden6 = Linear(50,2,bias=False)
        # self.dropout6 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        # xavier_uniform_(self.hidden6.weight)
        self.act6 = ReLU()

        #for i in range(0, len(sizes) - 1):
         #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
          #  layers.append(nn.ReLU())
           # if i < (len(sizes) - 2):
            #    layers.append(nn.ReLU())
        #layers.append(nn.Sigmoid())        

        #self.net = nn.Sequential(*layers)
        #self.net.apply(Xavier)

    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        # X = nn.BatchNorm1d(100)(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))
        # X = self.act1((X))
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        # X = nn.BatchNorm1d(250)(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))
        # X = self.act2((X))
        # third hidden layer and output
        self.act['hidden3']= X
        X = self.hidden3(X)
        # X = nn.BatchNorm1d(500)(X)
        X = self.bn3(X)
        X = self.act3(self.dropout3(X))
        # X = self.act3((X))
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        # X = nn.BatchNorm1d(150)(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))
        # X = self.act4((X))
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        # X = nn.BatchNorm1d(50)(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))
        # X = self.act5((X))
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        # X = self.act6(self.dropout6(X))
        X = self.act6((X))
        
        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()

        return X




# class CICIDS2018_FC(nn.Module):
#     def __init__(self,inputsize, softmax=False):
#         sizes = [inputsize,100,250,500,200,50,10,1]
#         super(CICIDS2018_FC, self).__init__()
#         layers = []
#         self.softmax=softmax
        
        
#         # self.act=OrderedDict()
#         # self.hidden1 = Linear(inputsize,100,bias=False)
#         # kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
#         # self.act1 = ReLU()
#         # # second hidden layer
#         # self.hidden2 = Linear(100, 250,bias=False)
#         # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
#         # self.act2 = ReLU()
#         # self.hidden3 = Linear(250, 500,bias=False)
#         # kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
#         # self.act3 = ReLU()
#         # self.hidden4 = Linear(500, 200,bias=False)
#         # kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
#         # self.act4 = ReLU()
#         # self.hidden5 = Linear(200, 50,bias=False)
#         # kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
#         # self.act5 = ReLU()
#         # # third hidden layer and output
#         # self.hidden6 = Linear(50,10,bias=False)
#         # kaiming_uniform_(self.hidden6.weight,nonlinearity='relu')
#         # #self.double()
#         # self.act6 = ReLU()

#         # self.hidden7 = Linear(10,1,bias=False)
#         # kaiming_uniform_(self.hidden7.weight,nonlinearity='sigmoid')
#         # #self.double()
#         # self.act7 = Sigmoid()
#         self.act=OrderedDict()
#         self.hidden1 = Linear(inputsize,100,bias=False)
#         self.bn1 = nn.BatchNorm1d(100)
#         self.dropout1 = nn.Dropout(0.2)
#         kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
#         self.act1 = ReLU()
#         # second hidden layer
#         self.hidden2 = Linear(100, 250,bias=False)
#         self.bn2 = nn.BatchNorm1d(250)
#         self.dropout2 = nn.Dropout(0.2)
#         kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
#         self.act2 = ReLU()
#         self.hidden3 = Linear(250, 100,bias=False)
#         self.bn3 = nn.BatchNorm1d(100)
#         self.dropout3 = nn.Dropout(0.2)
#         kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
#         self.act3 = ReLU()
#         self.hidden4 = Linear(100, 200,bias=False)
#         self.bn4 = nn.BatchNorm1d(200)
#         self.dropout4 = nn.Dropout(0.2)
#         kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
#         self.act4 = ReLU()
#         self.hidden5 = Linear(200, 50,bias=False)
#         self.bn5 = nn.BatchNorm1d(50)
#         self.dropout5 = nn.Dropout(0.2)
#         kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
#         self.act5 = ReLU()
#         # third hidden layer and output
#         self.hidden6 = Linear(50,10,bias=False)
#         self.bn6 = nn.BatchNorm1d(10)
#         self.dropout6 = nn.Dropout(0.2)
#         kaiming_uniform_(self.hidden6.weight,nonlinearity='relu')
#         #self.double()
#         self.act6 = ReLU()

#         self.hidden7 = Linear(10,2,bias=False)
#         # kaiming_uniform_(self.hidden7.weight,nonlinearity='sigmoid')
#         kaiming_uniform_(self.hidden7.weight,nonlinearity='relu')
#         #self.double()
#         self.act7 = ReLU()
#         # self.act7 = Sigmoid()

#         #for i in range(0, len(sizes) - 1):
#          #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#           #  layers.append(nn.ReLU())
#            # if i < (len(sizes) - 2):
#             #    layers.append(nn.ReLU())
#         #layers.append(nn.Sigmoid())        

#         #self.net = nn.Sequential(*layers)
#         #self.net.apply(Xavier)

#     def forward(self, X):
#         X = X.float()
#         self.act['hidden1']= X
#         X = self.hidden1(X)
#         X = self.bn1(X)
#         X = self.act1(self.dropout1(X))
#         # X = self.act1(X)
#         # second hidden layer
#         self.act['hidden2']= X
#         X = self.hidden2(X)
#         X = self.bn2(X)
#         X = self.act2(self.dropout2(X))
#         # X = self.act2(X)
#         # third hidden layer and output
#         self.act['hidden3']= X
#         X = self.hidden3(X)
#         X = self.bn3(X)
#         X = self.act3(self.dropout3(X))
#         # X = self.act3(X)
#         # third hidden layer and output
#         self.act['hidden4']= X
#         X = self.hidden4(X)
#         X = self.bn4(X)
#         X = self.act4(self.dropout4(X))
#         # X = self.act4(X)
#         # third hidden layer and output
#         self.act['hidden5']= X
#         X = self.hidden5(X)
#         X = self.bn5(X)
#         X = self.act5(self.dropout5(X))
#         # X = self.act5(X)
#         # third hidden layer and output
#         self.act['hidden6']= X
#         X = self.hidden6(X)
#         X = self.bn6(X)
#         X = self.act6(self.dropout6(X))
#         # X = self.act6(X)

#         self.act['hidden7']= X
#         X = self.hidden7(X)
#         X = self.act7(X)

#         if self.softmax:
#             X = torch.softmax(X,dim=1).squeeze()
        
#         return X
    
class CICIDS2018_FC(nn.Module):
    def __init__(self,inputsize, softmax=False):
        sizes = [inputsize,100,250,500,200,50,10,1]
        super(CICIDS2018_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        
        # self.act=OrderedDict()
        # self.hidden1 = Linear(inputsize,100,bias=False)
        # kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # self.act1 = ReLU()
        # # second hidden layer
        # self.hidden2 = Linear(100, 250,bias=False)
        # kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # self.act2 = ReLU()
        # self.hidden3 = Linear(250, 500,bias=False)
        # kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # self.act3 = ReLU()
        # self.hidden4 = Linear(500, 200,bias=False)
        # kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # self.act4 = ReLU()
        # self.hidden5 = Linear(200, 50,bias=False)
        # kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        # self.act5 = ReLU()
        # # third hidden layer and output
        # self.hidden6 = Linear(50,10,bias=False)
        # kaiming_uniform_(self.hidden6.weight,nonlinearity='relu')
        # #self.double()
        # self.act6 = ReLU()

        # self.hidden7 = Linear(10,1,bias=False)
        # kaiming_uniform_(self.hidden7.weight,nonlinearity='sigmoid')
        # #self.double()
        # self.act7 = Sigmoid()
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()


        self.hidden3 = Linear(250, 100,bias=False)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()

        self.hidden31 = Linear(100, 300,bias=False)
        self.bn31 = nn.BatchNorm1d(300)
        self.dropout31 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden31.weight, nonlinearity='relu')
        self.act31 = ReLU()

        self.hidden32 = Linear(300, 100,bias=False)
        self.bn32 = nn.BatchNorm1d(100)
        self.dropout32 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden32.weight, nonlinearity='relu')
        self.act32 = ReLU()


        self.hidden4 = Linear(100, 200,bias=False)
        self.bn4 = nn.BatchNorm1d(200)
        self.dropout4 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        self.hidden5 = Linear(200, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(50,10,bias=False)
        self.bn6 = nn.BatchNorm1d(10)
        self.dropout6 = nn.Dropout(0.2)
        kaiming_uniform_(self.hidden6.weight,nonlinearity='relu')
        #self.double()
        self.act6 = ReLU()

        self.hidden7 = Linear(10,2,bias=False)
        # kaiming_uniform_(self.hidden7.weight,nonlinearity='sigmoid')
        kaiming_uniform_(self.hidden7.weight,nonlinearity='relu')
        #self.double()
        self.act7 = ReLU()
        # self.act7 = Sigmoid()

        #for i in range(0, len(sizes) - 1):
         #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
          #  layers.append(nn.ReLU())
           # if i < (len(sizes) - 2):
            #    layers.append(nn.ReLU())
        #layers.append(nn.Sigmoid())        

        #self.net = nn.Sequential(*layers)
        #self.net.apply(Xavier)

    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))
        # X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))
        # X = self.act2(X)
        # third hidden layer and output

        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.bn3(X)
        X = self.act3(self.dropout3(X))
        
        self.act['hidden31']= X
        X = self.hidden31(X)
        X = self.bn31(X)
        X = self.act31(self.dropout3(X))
        # X = self.act3(X)


        self.act['hidden32']= X
        X = self.hidden32(X)
        X = self.bn32(X)
        X = self.act32(self.dropout3(X))


        
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))
        # X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))
        # X = self.act5(X)
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.bn6(X)
        X = self.act6(self.dropout6(X))
        # X = self.act6(X)

        self.act['hidden7']= X
        X = self.hidden7(X)
        X = self.act7(X)

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()
        
        return X

# class CICIDS2018_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,250,500,200,50,10,1]
#         super(CICIDS2018_FC, self).__init__()
#         layers = []

#         for i in range(0, len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < (len(sizes) - 2):
#                 layers.append(nn.ReLU())
#         layers.append(nn.Sigmoid())        

#         self.net = nn.Sequential(*layers)
#         self.net.apply(Xavier)

#     def forward(self, x):
#         return self.net(x)
class UNSWNB15_FC(nn.Module):
    def __init__(self,inputsize,softmax=False):
        sizes = [inputsize,100,250,500,150,50,1]
        super(UNSWNB15_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        drop_out = 0.2
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(50,2,bias=False)
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        # kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
        #self.double()
        # self.act6 = Sigmoid()
        self.act6 = ReLU()

        #for i in range(0, len(sizes) - 1):
         #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
          #  layers.append(nn.ReLU())
           # if i < (len(sizes) - 2):
            #    layers.append(nn.ReLU())
        #layers.append(nn.Sigmoid())        

        #self.net = nn.Sequential(*layers)
        #self.net.apply(Xavier)

    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))
        # X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))
        # X = self.act2(X)
        # third hidden layer and output
        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.bn3(X)
        X = self.act3(self.dropout3(X))
        # X = self.act3(X)
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))
        # X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))
        # X = self.act5(X)
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.act6(X)

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()
        
        return X


# class UNSWNB15_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,250,500,150,50,1]
#         super(UNSWNB15_FC, self).__init__()
#         layers = []
        
        
#         self.act=OrderedDict()
#         self.hidden1 = Linear(inputsize,100,bias=False)
#         kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
#         self.act1 = ReLU()
#         # second hidden layer
#         self.hidden2 = Linear(100, 250,bias=False)
#         kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
#         self.act2 = ReLU()
#         self.hidden3 = Linear(250, 500,bias=False)
#         kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
#         self.act3 = ReLU()
#         self.hidden4 = Linear(500, 150,bias=False)
#         kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
#         self.act4 = ReLU()
#         self.hidden5 = Linear(150, 50,bias=False)
#         kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
#         self.act5 = ReLU()
#         # third hidden layer and output
#         self.hidden6 = Linear(50,1,bias=False)
#         kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
#         #self.double()
#         self.act6 = Sigmoid()

#         #for i in range(0, len(sizes) - 1):
#          #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#           #  layers.append(nn.ReLU())
#            # if i < (len(sizes) - 2):
#             #    layers.append(nn.ReLU())
#         #layers.append(nn.Sigmoid())        

#         #self.net = nn.Sequential(*layers)
#         #self.net.apply(Xavier)

#     def forward(self, X):
#         X = X.float()
#         self.act['hidden1']= X
#         X = self.hidden1(X)
#         X = self.act1(X)
#         # second hidden layer
#         self.act['hidden2']= X
#         X = self.hidden2(X)
#         X = self.act2(X)
#         # third hidden layer and output
#         self.act['hidden3']= X
#         X = self.hidden3(X)
#         X = self.act3(X)
#         # third hidden layer and output
#         self.act['hidden4']= X
#         X = self.hidden4(X)
#         X = self.act4(X)
#         # third hidden layer and output
#         self.act['hidden5']= X
#         X = self.hidden5(X)
#         X = self.act5(X)
#         # third hidden layer and output
#         self.act['hidden6']= X
#         X = self.hidden6(X)
#         X = self.act6(X)
        
#         return X


# class UNSWNB15_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,250,500,150,50,1]
#         super(UNSWNB15_FC, self).__init__()
#         layers = []

#         for i in range(0, len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < (len(sizes) - 2):
#                 layers.append(nn.ReLU())
#         layers.append(nn.Sigmoid())        

#         self.net = nn.Sequential(*layers)
#         self.net.apply(Xavier)

#     def forward(self, x):
#         return self.net(x)        

class CIDDS_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,500,250,50,1]
        super(CIDDS_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)   

class KDDCUP99_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,500,150,50,1]
        super(KDDCUP99_FC, self).__init__()
        layers = []
        
        
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(250, 500,bias=False)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 150,bias=False)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(50,1,bias=False)
        kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
        #self.double()
        self.act6 = Sigmoid()

        #for i in range(0, len(sizes) - 1):
         #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
          #  layers.append(nn.ReLU())
           # if i < (len(sizes) - 2):
            #    layers.append(nn.ReLU())
        #layers.append(nn.Sigmoid())        

        #self.net = nn.Sequential(*layers)
        #self.net.apply(Xavier)

    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.act3(X)
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.act5(X)
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.act6(X)
        
        return X



class CTU13_FC(nn.Module):
    def __init__(self,inputsize,softmax=False):
        sizes = [inputsize,100,250,500,150,50,1]
        super(CTU13_FC, self).__init__()
        layers = []
        self.softmax=softmax
        
        drop_out = 0
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.hidden1.weight)

        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 250,bias=False)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # self.hidden3 = Linear(250, 500,bias=False)
        # self.bn3 = nn.BatchNorm1d(500)
        # self.dropout3 = nn.Dropout(drop_out)
        # kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # # nn.init.xavier_uniform_(self.hidden3.weight)
        # self.act3 = ReLU()
        # self.hidden4 = Linear(500, 150,bias=False)
        self.hidden4 = Linear(250, 150,bias=False)
        self.bn4 = nn.BatchNorm1d(150)
        self.dropout4 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(150, 50,bias=False)
        self.bn5 = nn.BatchNorm1d(50)
        self.dropout5 = nn.Dropout(drop_out)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.hidden5.weight)
        self.act5 = ReLU()
        # third hidden layer and output
        self.hidden6 = Linear(50,2,bias=False)
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.hidden6.weight)
        # kaiming_uniform_(self.hidden6.weight,nonlinearity='sigmoid')
        #self.double()
        # self.act6 = Sigmoid()
        self.act6 = ReLU()

        #for i in range(0, len(sizes) - 1):
         #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
          #  layers.append(nn.ReLU())
           # if i < (len(sizes) - 2):
            #    layers.append(nn.ReLU())
        #layers.append(nn.Sigmoid())        

        #self.net = nn.Sequential(*layers)
        #self.net.apply(Xavier)

    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.bn1(X)
        X = self.act1(self.dropout1(X))
        # X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.bn2(X)
        X = self.act2(self.dropout2(X))
        # X = self.act2(X)
        # third hidden layer and output
        # self.act['hidden3']= X
        # X = self.hidden3(X)
        # X = self.bn3(X)
        # X = self.act3(self.dropout3(X))
        # X = self.act3(X)
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.bn4(X)
        X = self.act4(self.dropout4(X))
        # X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.bn5(X)
        X = self.act5(self.dropout5(X))
        # X = self.act5(X)
        # third hidden layer and output
        self.act['hidden6']= X
        X = self.hidden6(X)
        X = self.act6(X)

        if self.softmax:
            X = torch.softmax(X,dim=1).squeeze()
        
        return X


# class KDDCUP99_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,250,500,150,50,1]
#         super(KDDCUP99_FC, self).__init__()
#         layers = []

#         for i in range(0, len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < (len(sizes) - 2):
#                 layers.append(nn.ReLU())
#         layers.append(nn.Sigmoid())        

#         self.net = nn.Sequential(*layers)
#         self.net.apply(Xavier)

#     def forward(self, x):
#         return self.net(x) 


class custom_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(custom_LSTM, self).__init__()
        
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.output_size = output_size
        

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True) #lstm
        self.fc = nn.Linear(hidden_size, self.output_size) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(cfg.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(cfg.device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc(out) 
        
        return out



class CIFAR10_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,500,100,50,1]
        super(CIFAR10_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


# class MNIST_FC(nn.Module):
#     def __init__(self,inputsize):
#         sizes = [inputsize,100,150,50,10,1]
#         super(MNIST_FC, self).__init__()
#         layers = []

#         for i in range(0, len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1]))
#             if i < (len(sizes) - 2):
#                 layers.append(nn.ReLU())
#         layers.append(nn.Sigmoid())        

#         self.net = nn.Sequential(*layers)
#         self.net.apply(Xavier)

#     def forward(self, x):
#         return self.net(x)  


class MNIST_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,500,150,50,1]
        super(MNIST_FC, self).__init__()
        layers = []
        
        
        self.act=OrderedDict()
        self.hidden1 = Linear(inputsize,100,bias=False)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 150,bias=False)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(150, 50,bias=False)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(50, 10,bias=False)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        
        self.hidden5 = Linear(10,1,bias=False)
        kaiming_uniform_(self.hidden5.weight,nonlinearity='sigmoid')
        #self.double()
        self.act5 = Sigmoid()

        #for i in range(0, len(sizes) - 1):
         #   layers.append(nn.Linear(sizes[i], sizes[i + 1]))
          #  layers.append(nn.ReLU())
           # if i < (len(sizes) - 2):
            #    layers.append(nn.ReLU())
        #layers.append(nn.Sigmoid())        

        #self.net = nn.Sequential(*layers)
        #self.net.apply(Xavier)

    def forward(self, X):
        X = X.float()
        self.act['hidden1']= X
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        self.act['hidden2']= X
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        self.act['hidden3']= X
        X = self.hidden3(X)
        X = self.act3(X)
        # third hidden layer and output
        self.act['hidden4']= X
        X = self.hidden4(X)
        X = self.act4(X)
        # third hidden layer and output
        self.act['hidden5']= X
        X = self.hidden5(X)
        X = self.act5(X)
        
        
        return X
class SVHN_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,150,50,10,1]
        super(SVHN_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)          


class RESNET_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,50,1]
        super(RESNET_FC, self).__init__()
         
        model = resnet18(weights=None)       
        model.fc = nn.Sigmoid()
        self.net = model

        
    def forward(self, x):
        return self.net(x)       


class Weights_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,2,inputsize]
        super(Weights_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)    



class CLEAR10_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,150,100,50,1]
        super(CLEAR10_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)    


class SMAP_FC(nn.Module):
    def __init__(self,inputsize):
        sizes = [inputsize,100,250,150,50,1]
        super(SMAP_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)   





class ICARL_FC(nn.Module):
    def __init__(self,inputsize=512):
        sizes = [inputsize,100,50,1]
        super(ICARL_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())        

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)                              


              