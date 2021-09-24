import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallerNet(nn.Module):
    def __init__(self):
        super(SmallerNet,self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels = 1 , out_channels = 40, kernel_size = 8)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(40)
        self.conv1d_2 = nn.Conv1d(in_channels = 40 , out_channels = 40, kernel_size = 8)
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm1d(40)
        self.pool = nn.MaxPool1d(kernel_size = 128, stride = 128)
        self.relu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm1d(40)

        self.conv2d_1 = nn.Conv2d(in_channels = 1 , out_channels = 24, kernel_size = (6,6))
        self.relu4 = nn.ReLU()
        self.batchNorm4 = nn.BatchNorm2d(24)
        self.conv2d_2 = nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = (6,6))
        self.relu5 = nn.ReLU()
        self.batchNorm5 = nn.BatchNorm2d(24)
        self.conv2d_3 = nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = (5,5), stride = (2,2))
        self.relu6 = nn.ReLU()
        self.batchNorm6 = nn.BatchNorm2d(48)
        self.conv2d_4 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (5,5), stride = (2,2))
        self.relu7 = nn.ReLU()
        self.batchNorm7 = nn.BatchNorm2d(48)
        self.conv2d_5 = nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = (4,4), stride = (2,2))
        self.relu8 = nn.ReLU()
        self.batchNorm8 = nn.BatchNorm2d(64)

        self.batchNorm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout()
        self.fcl = nn.Linear(64,2)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.conv1d_1(x)
        x = self.relu1(x)
        #x = self.batchNorm1(x)
        x = self.conv1d_2(x)
        x = self.relu2(x)
        #x = self.batchNorm2(x)
        x = self.pool(x)
        x = self.relu3(x)
        #x = self.batchNorm3(x)

        #print(x.shape)
        x = x.unsqueeze(1)
        #print(x.shape)

        x = self.conv2d_1(x)
        x = self.relu4(x)
        #x = self.batchNorm4(x)
        x = self.conv2d_2(x)
        x = self.relu5(x)
        #x = self.batchNorm5(x)
        x = self.conv2d_3(x)
        x = self.relu6(x)
        #x = self.batchNorm6(x)
        x = self.conv2d_4(x)
        x = self.relu7(x)
        #x = self.batchNorm7(x)
        x = self.conv2d_5(x)
        x = self.relu8(x)
        x = torch.reshape(x,(-1,64))
        #x = self.batchNorm(x)
        x = self.dropout(x)
        x = self.softmax(self.fcl(x))

        return x
