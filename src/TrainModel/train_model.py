import torch
import torch.nn.functional as F
import random as rd
import numpy as np
import time
import os
import pandas
import random
from pynput import keyboard
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



################## Classes #################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 调用基类的__init__()函数
        '''torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True)'''

        '''torch.nn.MaxPool2d(kernel_size, stride=None, 
            padding=0, dilation=1, return_indices=False, ceil_mode=False)
            stride – the stride of the window. Default value is kernel_size'''
        self.conv = nn.Sequential(  # 顺序网络结构
            nn.Conv2d(1, 16, 9, stride=1, padding=2),  # 卷积层 输入1通道，输出6通道，kernel_size=5*5
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(3, 3),  # 最大池化，kernel_size=2*2，stride=2*2
            # 输出大小为14*14
            nn.Conv2d(16, 32, 5, stride=1, padding=2),  # 卷积层 输入6通道，输出16通道，kernel_size=5*5
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            # 输出大小为7*7
            nn.Conv2d(32, 120, 5, stride=1, padding=2),  # 卷积层 输入16通道，输出120通道，kernel_size=5*5
            nn.ReLU(),
        )
        self.fc = nn.Sequential(  # 全连接层
            nn.Linear(1200, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(),
        )

    def forward(self, x):  # 前向传播
        # print(x.shape)
        out = self.conv(x)
        out = out.view(out.size(0), -1)  # 展平数据为7*7=49的一维向量
        out = self.fc(out)
        return out


def load_data():
    files = os.listdir('Data_Study1_Difference/')
    t_data = []
    nt_data = []
    count = 0
    for file_name in files:
        print(count, 'Read: ' + file_name)
        if '_touch' in file_name:
            t_data.append(pandas.read_csv('Data_Study1_Difference/'+file_name).values)
        elif '_nontouch' in file_name:
            nt_data.append(pandas.read_csv('Data_Study1_Difference/'+file_name).values)
        count += 1
    return t_data, nt_data


def get_batch(batch_num, t_data, nt_data, step = 1, size = 100):
    out_put = []
    # if batch_num%2 == 1:
    #     batch_num = batch_num - 1
    for i in range(batch_num // 2):
        file = random.randint(0, len(t_data) - 1)
        index = random.randint(0, len(t_data[file]) - step * size - 1)
        temp_data = np.array(t_data[file][index: index + size * step: step]).reshape((1, -1)).tolist()
        out_put.append(temp_data)

    for i in range(batch_num // 2):
        file = random.randint(0, len(nt_data) - 1)
        index = random.randint(0, len(nt_data[file]) - step * size - 1)
        temp_data = np.array(nt_data[file][index : index + size * step: step]).reshape((1, -1)).tolist()
        out_put.append(temp_data)

    out_put = np.array(out_put).reshape((-1, 1, 9, 100))

    labels = np.ones(batch_num//2).tolist()
    labels += np.zeros(batch_num//2).tolist()

    # print(out_put)
    out_put = Variable(torch.FloatTensor(out_put))
    labels = np.array(labels).astype('float64')
    labels = Variable(torch.LongTensor(labels))
    return out_put, labels

def on_press(key):
    global loop

    eventkey = '{0}'.format(key)
    if eventkey == 'Key.esc':
        loop = False

def on_release(key):
    pass

t_data, nt_data = load_data()

net = LeNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()

loop = True
keyboard.Listener(on_press = on_press, on_release = on_release).start()

for i in range(10000):
    (data, label) = get_batch(1000, t_data, nt_data, step=3, size=100)
    # print(data)
    prediction = net(data)
    loss = loss_func(prediction, label)
    print(i, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ##test
    if i%5==0:
        (test_data, test_label) = get_batch(1000, t_data, nt_data, step=3, size=100)
        prediction = torch.max(F.softmax(net(test_data)), 1)[1]
        y_pred = prediction.data.numpy().squeeze()
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == test_label[i]:
                count += 1
        print(count / 1000)

    if not loop:
        break

torch.save(net, 'models/touch_nontouch')