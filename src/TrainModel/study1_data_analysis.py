from study1_train_model import LeNet
import torch
import numpy as np
import os
import pandas
from torch.autograd import Variable
import torch.nn.functional as F

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
        start = start + 5

    out_put = np.array(out_put).reshape((-1, 1, 9, size))
    out_put = Variable(torch.FloatTensor(out_put))
    labels = np.array(labels).astype('float64')
    labels = Variable(torch.LongTensor(labels))
    return out_put, labels

net = LeNet()
net = torch.load('../../models/S1_step2_size100.txt')
net.eval()
files = os.listdir('../../data/Data_Study1_Difference/')
elbow_condition = ['elbow', 'free']
for elbow in elbow_condition:
    for file_name in files:
        if '_touch' in file_name and elbow in file_name:
            data_np = pandas.read_csv('../../data/Data_Study1_Difference/'+file_name).values
            t_data_test = data_np[4 * len(data_np) // 5:]
            (test_data, test_label) = get_test(t_data_test, 2, 100, 1)

            prediction = torch.max(F.softmax(net(test_data)), 1)[1]
            y_pred = prediction.data.numpy().squeeze()
            count = 0
            for i in range(len(y_pred)):
                if y_pred[i] == test_label[i]:
                    count += 1
            print('accuracy', round(count / len(test_label), 3), '\t', file_name)

        elif '_nontouch' in file_name and elbow in file_name:
            data_np = pandas.read_csv('../../data/Data_Study1_Difference/' + file_name).values
            nt_data_test = data_np[4 * len(data_np) // 5:]
            (test_data, test_label) = get_test(nt_data_test, 2, 100, 0)

            prediction = torch.max(F.softmax(net(test_data)), 1)[1]
            y_pred = prediction.data.numpy().squeeze()
            count = 0
            for i in range(len(y_pred)):
                if y_pred[i] == test_label[i]:
                    count += 1
            print('accuracy', round(count / len(test_label), 3), '\t', file_name)

exit(0)