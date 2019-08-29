import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, svm, tree
import numpy as np
import os
import sys
import random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import math


def extract_festure(data):
    feature_list = []
    separate_num = 20
    skip = 2
    data_separated = [data[i*len(data)//separate_num:(i+1)*len(data)//separate_num] for i in range(separate_num)]

    for item in data_separated:
        item.sort(0)
        length = len(item)
        st = skip * length // separate_num
        # st = 0
        en = (separate_num - skip) * length // separate_num
        feature_list += item[st: en, 0:3].max(0).tolist() # 73%
        feature_list += item[st: en, 0:3].min(0).tolist()   # 
        # 23%
        feature_list.append(np.sqrt((item[st: en,0:1]**2+item[st:en,2:3]**2).mean()) - np.sqrt((item[st:en,1:2]**2).mean()))
        feature_list.append(item[st: en, 0:1].std() + item[st: en, 2:3].std())
        feature_list.append(item[st: en, 1:2].std())
        feature_list += item[st: en, 3:9].mean(0).tolist() # 75%
        feature_list += item[st: en, 3:9].std(0).tolist() # 37%
        feature_list += np.sqrt((item * item)[st: en].mean(0)).tolist() #44%
    feature_list = np.array(feature_list)
    feature_list = feature_list / feature_list.max()
    return feature_list.tolist()


def separate_features(features, labels, info):
    test_x = []
    test_y = []
    train_x = []
    train_y = []
    test_info = []
    train_info = []

    index = [[j for j in range(len(features[i]))] for i in range(len(features))]
    for i in range(len(features)):
        length = len(index[i])
        for j in range(length // 3):
            k = rd.randint(0, len(index[i])-1)
            n = index[i].pop(k)
            test_x.append(features[i][n])
            test_y.append(labels[i][n])
            test_info.append(info[i][n])
    for i in range(len(features)):
        for j in range(len(index[i])):
            n = index[i].pop(0)
            train_x.append(features[i][n])
            train_y.append(labels[i][n])
            train_info.append(info[i][n])
    return train_x, train_y, test_x, test_y, test_info, train_info

def remove_static(data):
    processed_data = []
    return processed_data

data_path = '../../data/DataProcessed_Study2/'
gesturelabel_path = '../../data/DataLabels_Study2/'
column_list = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
# column_list = ['mx', 'my', 'mz']
data_label = []
New_file = True
data_train = []
label_train = []
template = []
count = 0

data_list = os.listdir(data_path)
gesturelabel_list = os.listdir(gesturelabel_path)
name_list = ['_chamod','_clarence','_hussel','_jing',\
            '_kaixing','_logan','_mevan','_sachith','_samantha'\
            ,'_samitha','_vipula','_yilei','_yiyue']

if len(sys.argv) > 1:
    name_list = ['_'+sys.argv[-1]]

gesture_dic = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,
                'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,
                'V':21,'W':22,'X':23,'Y':24,'Z':25,'delete':26,'space':27 }

feature_list = [[] for i in range(28)]
label_list = [[] for i in range(28)]
info = [[] for i in range(28)]

for name in name_list:
    print(name)
    data_file = pd.read_csv(data_path + 'data_processed' + name)[column_list].values
    letter_all = pd.read_csv(gesturelabel_path + 'gest_pos_index_'+name+'.csv')[['0']].values
    index_all = pd.read_csv(gesturelabel_path + 'gest_pos_index_'+name+'.csv')[['1','2']].values
    for i in range(len(index_all)):
        feature = extract_festure(data_file[index_all[i][0]:index_all[i][1],:])
        feature_list[gesture_dic[letter_all[i][0]]].append(feature)
        label_list[gesture_dic[letter_all[i][0]]].append(gesture_dic[letter_all[i][0]])
        info[gesture_dic[letter_all[i][0]]].append([name, index_all[i][0], index_all[i][1]])

# feature_np = np.array(feature_list)
# feature_np = (feature_np - feature_np.mean(0)) / feature_np.std(0)
# feature_list = feature_np.tolist()
    # for i in range(len(feature_list)):
    #     print(i, len(feature_list[i]))
    

train_x, train_y, test_x, test_y, test_info, train_info = separate_features(feature_list, label_list, info)

train_x = np.array(train_x)
train_y = np.array(train_y)

# model = KNeighborsClassifier(n_neighbors = 5)
model = LogisticRegression(tol=1e-5, C=0.3, solver='saga', class_weight='balanced', multi_class='multinomial')
# model = svm.SVC(C=5, tol=1e-5)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)

plt.ion()

axes = plt.gca()
line1, = axes.plot([], [], color = 'green')
line2, = axes.plot([], [], color = 'red')
line3, = axes.plot([], [], color = 'blue')
correct = 0

total_num = [0 for i in range(28)]
correct_num = [0 for i in range(28)]
result = [[0 for j in range(28)] for i in range(28)]
for i in range(len(test_y)):
    result[test_y[i]][y_pred[i]] += 1
    total_num[test_y[i]] += 1
    if test_y[i] == y_pred[i]:
        correct_num[test_y[i]] += 1
    # print('label: ', test_y[i], 'predict: ', y_pred[i])
    # else:
    #     data = pd.read_csv(data_path + 'data_processed' + test_info[i][0])[column_list].values
    #     line1.set_xdata([k for k in range(test_info[i][1], test_info[i][2])])
    #     line2.set_xdata([k for k in range(test_info[i][1], test_info[i][2])])
    #     # line3.set_xdata([k for k in range(test_info[i][1], test_info[i][2])])
 
    #     line1.set_ydata(np.sqrt(data[test_info[i][1]:test_info[i][2], 0:1]**2+data[test_info[i][1]:test_info[i][2], 2:3]**2))
    #     line2.set_ydata(np.sqrt(data[test_info[i][1]:test_info[i][2], 1:2]**2))
    #     # line3.set_ydata(data[test_info[i][1]:test_info[i][2], 2:3])
        
    #     plt.xlim([test_info[i][1], test_info[i][2]])
    #     plt.ylim([-100,100])
        
    #     plt.draw()
    #     plt.pause(1e-17)
    #     print('label: ', test_y[i], 'predict: ', y_pred[i])
    #     input('next')


print(result)
for i in range(len(total_num)):
    print(i, chr(97 + i), correct_num[i], total_num[i],'accuracy:', correct_num[i] / total_num[i])
print('Overall: ', sum(correct_num) / sum(total_num))



    


