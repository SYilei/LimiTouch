import torch
import numpy as np
import os
import sys
import pandas
import random
from pynput import keyboard
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import csv


################## Classes #################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 调用基类的__init__()函数
        '''torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True)'''

        '''torch.nn.MaxPool2d(kernel_size, stride=None, 
            padding=0, dilation=1, return_indices=False, ceil_mode=False)
            stride – the stride of the window. Default value is kernel_size'''
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 9, stride=1, padding=4),  # 卷积层 输入1通道，输出6通道，kernel_size=5*5
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(3, 3),
            nn.Conv2d(16, 32, 5, stride=1, padding=2),  # 卷积层 输入6通道，输出16通道，kernel_size=5*5
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 120, 5, stride=1, padding=2),  # 卷积层 输入16通道，输出120通道，kernel_size=5*5
            nn.ReLU(),
        )
        self.fc = nn.Sequential(  #全连接层
            #nn.Linear(1320, 512),
            nn.Linear(3960, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(),
        )

    def forward(self, x):  # 前向传播
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_test(data, step, size, touch):
    out_put = []
    labels = []
    start = 0
    while start + (size - 1) * step < (len(data) - 1):
        temp_data = np.array(data[start: start + size * step: step]).reshape((1, -1)).tolist()
        out_put.append(temp_data)
        if touch == 0:
            labels.append(0)
        elif touch == 1:
            labels.append(1)
        start = start + 20

    out_put = np.array(out_put).reshape((-1, 1, 9, size))
    out_put = Variable(torch.FloatTensor(out_put))
    labels = np.array(labels).astype('float64')
    labels = Variable(torch.LongTensor(labels))
    return out_put, labels

############# Start the code ###############
device = torch.device('cpu')

model_file = '../models/S1_step_1_size_250.txt'
step = int(model_file.split('_')[2])
size = int(model_file.split('_')[4].split('.')[0])
net = torch.load(model_file).to(device)
net.eval()

#-------------------------------------------------------------
data_path = '../data/Data_Study1_Difference/'
files = os.listdir(data_path)
#files = os.listdir('..\\limitouch_independent_test_data\\')
elbow_condition = ['elbow', 'free']
res_list = []

for elbow in elbow_condition:
    for file_name in files:
        if file_name.split('_')[0] != 'evan' and file_name.split('_')[0] != 'mel':
            temp_list = []
            if '_touch' in file_name and elbow in file_name:
                # -------------------------------------------------------------
                data_np = pandas.read_csv(data_path + file_name).values
                #data_np = pandas.read_csv('..\\limitouch_independent_test_data\\'+file_name).values
                t_data_test = data_np[4 * len(data_np) // 5:]
                # -------------------------------------------------------------
                (test_data, test_label) = get_test(t_data_test, step, size, 1)

                prediction = torch.max(F.softmax(net(test_data)), 1)[1]
                y_pred = prediction.data.numpy().squeeze()
                count = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == test_label[i]:
                        count += 1
                temp_list.append(file_name.split('_')[0])
                temp_list.append(file_name.split('_')[1])
                temp_list.append(file_name.split('_')[2])
                temp_list.append(file_name.split('_')[3].split('.')[0])
                temp_list.append(round(count / len(test_label), 3))
                res_list.append(temp_list)
                print('accuracy', round(count / len(test_label), 3), '\t', file_name)

            elif '_nontouch' in file_name and elbow in file_name:
                # -------------------------------------------------------------
                data_np = pandas.read_csv(data_path + file_name).values
                #data_np = pandas.read_csv('..\\limitouch_independent_test_data\\' + file_name).values
                nt_data_test = data_np[4 * len(data_np) // 5:]
                # -------------------------------------------------------------
                (test_data, test_label) = get_test(nt_data_test, step, size, 0)

                prediction = torch.max(F.softmax(net(test_data)), 1)[1]
                y_pred = prediction.data.numpy().squeeze()
                count = 0
                for i in range(len(y_pred)):
                    if y_pred[i] == test_label[i]:
                        count += 1
                temp_list.append(file_name.split('_')[0])
                temp_list.append(file_name.split('_')[1])
                temp_list.append(file_name.split('_')[2])
                temp_list.append(file_name.split('_')[3].split('.')[0])
                temp_list.append(round(count / len(test_label), 3))
                res_list.append(temp_list)
                #print('accuracy', round(count / len(test_label), 3), '\t', file_name)
                print('accuracy', count / len(test_label), '\t', file_name)

with open('res_user.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(res_list)

