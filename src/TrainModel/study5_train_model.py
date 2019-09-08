import torch
import numpy as np
import os
import sys
import pandas
import random
#from pynput import keyboard
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
            nn.Linear(3240, 512),
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


def load_data():
    global participants
    files = os.listdir(data_path)
    t_data_train = []
    t_data_test = []
    nt_data_train = []
    nt_data_test = []
    count = 0
    for file_name in files:
        if 'high_force' in file_name:
            data_np = pandas.read_csv(data_path + file_name).values
            t_data_train.append(data_np[:4 * len(data_np) // 5])
            t_data_test.append(data_np[4 * len(data_np) // 5:])
            print(count, 'Read: ' + file_name)
            count += 1
        elif 'low_force' in file_name:
            data_np = pandas.read_csv(data_path + file_name).values
            nt_data_train.append(data_np[:4 * len(data_np) // 5])
            nt_data_test.append(data_np[4 * len(data_np) // 5:])
            print(count, 'Read: ' + file_name)
            count += 1
    return t_data_train, nt_data_train, t_data_test, nt_data_test


def get_batch(batch_num, t_data, nt_data, step, size):
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

    out_put = np.array(out_put).reshape((-1, 1, 9, size))

    labels = np.ones(batch_num//2).tolist()
    labels += np.zeros(batch_num//2).tolist()

    out_put = Variable(torch.FloatTensor(out_put))
    labels = np.array(labels).astype('float64')
    labels = Variable(torch.LongTensor(labels))
    return out_put, labels


#def on_press(key):
#    global loop

 #   eventkey = '{0}'.format(key)
  #  if eventkey == 'Key.esc':
   #     loop = False


#def on_release(key):
#    pass


############# Start the code ###############

data_path = '../../data/Study5_force_derivative/'

loop = True
batch_size = 1000
train_num = 5000

data_step = 1
data_size = 250

participants = ['test','jiashuo']

net = LeNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()
t_data_train, nt_data_train, t_data_test, nt_data_test = load_data()

#keyboard.Listener(on_press=on_press, on_release=on_release).start()

for i in range(train_num):
    (data, label) = get_batch(batch_size, t_data_train, nt_data_train, data_step, data_size)
    # print(data)
    prediction = net(data)
    loss = loss_func(prediction, label)
    print(i, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ##test
    if i%5==0:
        (test_data, test_label) = get_batch(batch_size, t_data_test, nt_data_test, data_step, data_size)
        prediction = torch.max(F.softmax(net(test_data)), 1)[1]
        y_pred = prediction.data.numpy().squeeze()
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == test_label[i]:
                count += 1
        print(count / batch_size)

    if not loop:
        break

torch.save(net, '../../models/Study5/S1_'+'step_'+str(data_step)+'_size_'+str(data_size)+'.txt')








