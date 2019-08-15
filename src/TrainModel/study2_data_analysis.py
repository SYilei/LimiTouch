import torch
import numpy as np
import os
import pandas as pd
from torch.autograd import Variable
from study1_modelclass import LeNet
import torch.nn.functional as F

data_path = '../../data/DataProcessed_Study2/'
model_path = '../../models/new_step_1_size_250.txt'
name_list = []
count = 0
index = 0
data_list = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
data_files = os.listdir(data_path)
model = LeNet()
colname = ['letter','label']
letter = []


def analysis_test(data, step, size):
    label = []
    start = 0
    while start + (size - 1) * step < (len(data) - 1):
        temp_data = np.array(data[start: start + size * step: step]).reshape((1, -1)).tolist()
        #print(temp_data)
        temp_data = np.array(temp_data).reshape((-1, 1, 9, size))
        temp_data = Variable(torch.FloatTensor(temp_data))
        touch_prediction = torch.max(F.softmax(model(temp_data)), 1)[1]
        label.append([letter[start],touch_prediction.data.numpy().squeeze()])
        start = start + 5
    return label


#get name of participants
for filename in data_files:
    if filename.find('data') != -1:
        name = filename.split('_')[-1].split('.')[0]
        name_list.append(name)

for name in name_list:
    print(str(count) + '.' + name)
    count += 1


name_num = input("Input the num of the name data you want to process:")
print('----Begin to prepare ' + name_list[int(name_num)] + ' data----')
#get the data and dataindex file of participant
for filename in data_files:
    if filename.find(name_list[int(name_num)]) != -1:
        data_csv = pd.read_csv(data_path + filename)
        temp = data_csv[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']].values
        letter = data_csv[['letter']].values

print(letter)

model.load_state_dict(torch.load(model_path))
model.eval()
print('----Model load succesfully----')

touch_result = analysis_test(temp, 1, 250)

print('----Begin to save result as csv----')
result_csv = pd.DataFrame(columns=colname, data=touch_result)
result_csv.to_csv('data_touchlabel_' + name_list[int(name_num)] + '.csv', index=False)
