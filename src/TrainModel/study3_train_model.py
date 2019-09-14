import torch
import numpy as np
import os
import sys
import pandas
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from study1_modelclass import LeNet

def load_data(participant):
    global participants
    path = '../../data/Study3_derivative/'
    files = os.listdir(path)
    t_data_train = []
    t_data_test = []
    nt_data_train = []
    nt_data_test = []
    count = 0
    for file_name in files:
        if '_touch' in file_name and participant in file_name:
            data_np = pandas.read_csv(path + file_name).values
            t_data_train.append(data_np[:4 * len(data_np) // 5])
            t_data_test.append(data_np[4 * len(data_np) // 5:])
            print(count, 'Read: ' + file_name)
            count += 1
        elif '_nontouch' in file_name and participant in file_name:
            data_np = pandas.read_csv(path + file_name).values
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

############# Start the code ###############

loop = True
batch_size = 1000
train_num = 4000

data_step = 1
data_size = 250

participants = ['_evan','_pai','_chamod']

for participant in participants:
    net = LeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
    loss_func = torch.nn.CrossEntropyLoss()
    t_data_train, nt_data_train, t_data_test, nt_data_test = load_data(participant)

    #keyboard.Listener(on_press=on_press, on_release=on_release).start()

    for i in range(train_num):
        (data, label) = get_batch(batch_size, t_data_train, nt_data_train, data_step, data_size)
        # print(data)
        prediction = net(data)
        loss = loss_func(prediction, label)
        if i%5 == 0:
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
            print(count / 1000)

        if not loop:
            break

    # torch.save(net.state_dict(), '../../models/Study3/S3_'+'step_'+str(data_step)+'_size_'+str(data_size)+'.txt')

    torch.save(net.state_dict(), '../../models/Study3/S3_individual_'+participant+'_'+'step_'+str(data_step)+'_size_'+str(data_size)+'.txt')







