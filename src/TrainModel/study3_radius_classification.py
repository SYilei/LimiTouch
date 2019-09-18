import torch
import numpy as np
import os
import sys
import pandas
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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
            nn.Linear(128, 6),
            nn.Softmax(),
        )

    def forward(self, x):  # 前向传播
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def load_data():
    path = '../../data/Study3_derivative/'
    files = os.listdir(path)
    data_train = [[],[],[],[],[],[]]
    data_test = [[],[],[],[],[],[]]
    gestures = {'microwave':0, 'small_bottle':1, 'large_bottle':2, 'door_handle':3, 'cabinet':4, 'door.csv':5}

    count = 0
    for gesture in gestures.keys():
        for file_name in files:
            if '_touch' in file_name and gesture in file_name and 'hussel_' in file_name:
                data_np = pandas.read_csv(path + file_name).values
                data_np = (data_np - data_np.mean(0)) / data_np.std(0)
                data_train[gestures[gesture]].append(data_np[:4 * len(data_np) // 5])
                data_test[gestures[gesture]].append(data_np[4 * len(data_np) // 5:])
                print(count, 'Read: ' + file_name)
                count += 1
    return data_train, data_test


def get_batch(batch_num, data, step, size):
    out_put = []
    labels = []
    for gesture in range(6):
        for i in range(batch_num):
            file = random.randint(0, len(data[gesture]) - 1)
            index = random.randint(0, len(data[gesture][file]) - step * size - 1)
            temp_data = np.array(data[gesture][file][index: index + size * step: step]).reshape((1, -1)).tolist()
            out_put.append(temp_data)
            labels.append(gesture)

    out_put = np.array(out_put).reshape((-1, 1, 9, size))

    out_put = Variable(torch.FloatTensor(out_put))
    labels = np.array(labels).astype('float64')
    labels = Variable(torch.LongTensor(labels))
    return out_put, labels

############# Start the code ###############

loop = True
batch_size = 200
train_num = 4000

data_step = 1
data_size = 250

gestures = ['cabinet','door_handle','door.csv','large_bottle','small_bottle','microwave']


net = LeNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()
data_train, data_test = load_data()

#keyboard.Listener(on_press=on_press, on_release=on_release).start()

for i in range(train_num):
    (data, label) = get_batch(batch_size, data_train, data_step, data_size)
    prediction = net(data)
    loss = loss_func(prediction, label)
    # if i%5 == 0:
    print(i, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ##test
    if i%5==0:
        (test_data, test_label) = get_batch(batch_size, data_test, data_step, data_size)
        prediction = torch.max(F.softmax(net(test_data)), 1)[1]
        y_pred = prediction.data.numpy().squeeze()
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == test_label[i]:
                count += 1
        print(count / len(y_pred))

    if not loop:
        break

# torch.save(net.state_dict(), '../../models/Study3/S3_'+'step_'+str(data_step)+'_size_'+str(data_size)+'.txt')

torch.save(net.state_dict(), '../../models/Study3/S3_gesture_classification_'+'step_'+str(data_step)+'_size_'+str(data_size)+'.txt')







