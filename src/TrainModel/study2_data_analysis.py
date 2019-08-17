import torch
import numpy as np
import os
import sys
import pandas as pd
from torch.autograd import Variable
from study1_modelclass import LeNet
import torch.nn.functional as F

def analysis_test(data, index_list, step, size, participant):
    label = []
    label_index = []
    count = 0
    for index in index_list:
        start = 0
        st = int(index.split(',')[1])
        en = int(index.split(',')[2]) + 1
        data_clip = data[st:en]
        while start + (size - 1) * step < len(data_clip):
            temp_data = data_clip[start: start + size * step: step].reshape((-1, 1, 9, size))
            temp_data = Variable(torch.FloatTensor(temp_data))
            touch_prediction = torch.max(F.softmax(model(temp_data)), 1)[1]
            label.append([index.split(',')[0],touch_prediction.data.numpy().squeeze(), st+start])
            start = start + 1

            if start%1000 == 0:
                print(participant, st+start, ' out of ', len(data))

        label_index.append([index.split(',')[0], count, count + start])
        count = count + start
    return label, label_index


# name_num = input("Input the num of the name data you want to process:")
# print('----Begin to prepare ' + name_list[int(name_num)] + ' data----')
#get the data and dataindex file of participant

data_path = '../../data/DataProcessed_Study2/'
index_path = '../../data/DataProindex_Study2/'
data_files = os.listdir(data_path)
index_files = os.listdir(index_path)

print(data_files, index_files)

model_path = '../../models/new_step_1_size_250.txt'
step = 1
size = 250

name_list = ['_chamod','_clarence','_evan','_hussel','_jing',\
            '_kaixing','_logan','_mevan','_sachith','_samantha'\
            ,'_samitha','_vipula','_yilei','_yiyue']

name_list = ['_vipula','_yilei','_yiyue']

if len(sys.argv) > 1:
    name_list = ['_'+sys.argv[1]]
    print(name_list)

count = 0
index = 0
model = LeNet()
letter = []

model.load_state_dict(torch.load(model_path))
model.eval()
for participant in name_list:
    for filename in data_files:
        if participant in filename:
            data_csv = pd.read_csv(data_path + filename)
            data_np = data_csv[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']].values
            for index in index_files:
                if participant in index and participant:
                    index_list = open(index_path + index).readlines()
                    print('Processing: ' + participant)
                    (label_result, label_index) = analysis_test(data_np, index_list[1:], step, size, participant)
                    pd.DataFrame(label_result).to_csv('../../data/DataLabels_Study2/label_'+participant+'.csv')
                    pd.DataFrame(label_index).to_csv('../../data/DataLabels_Study2/labelindex_'+participant+'.csv')

