from sklearn import svm
from os import listdir
import random as rd
import numpy as np
import torch
import time
import pandas as pds
from pyquaternion import Quaternion
import madgwickahrs as mg
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier

gesture_dic = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8,
    '9':9,
    'a':10,
    'b':11,
    'c':12,
    'd':13,
    'e':14,
    'f':15,
    'g':16,
    'h':17,
    'i':18,
    'j':19,
    'k':20,
    'l':21,
    'm':22,
    'n':23,
    'o':24,
    'p':25,
    'q':26,
    'r':27,
    's':28,
    't':29,
    'u':30,
    'v':31,
    'w':32,
    'x':33,
    'y':34,
    'z':35,
}

x_acc_min = -960.0
x_acc_max = 1008.0
y_acc_min = -970.0
y_acc_max = 995.0
z_acc_min = -1030.0
z_acc_max = 970.0

x_bias = (x_acc_max+x_acc_min)/2.0
x_scale = (x_acc_max-x_acc_min)/2.0
y_bias = (y_acc_max+y_acc_min)/2.0
y_scale = (y_acc_max-y_acc_min)/2.0
z_bias = (z_acc_max+z_acc_min)/2.0
z_scale = (z_acc_max-z_acc_min)/2.0

acc_bias = np.array([x_bias, y_bias, z_bias])
acc_scale = np.array([x_scale, y_scale, z_scale])

ma = mg.MadgwickAHRS(sampleperiod = 1.0 / 1100, beta = 5)
sample_period = 1.0 / 1100

def read_data():
    # Get the raw data
    data_pds = pds.read_csv("data.csv")
    data_np = data_pds[["ax","ay","az","gx","gy","gz","mx","my","mz"]].values
    data_np = data_np.astype("float64")
    f = open("data_index.csv")
    lines = f.readlines()
    data_processed = []
    labels = []
    for line in lines:
        index = list(map(int, line.split(',')[1:]))
        # data_no_g = remove_gravity(data_np[index[0]:index[1],:])
        features = get_features(data_np[index[0]:index[1],:6])
        data_processed.append(features)
        labels.append(gesture_dic[line.split(',')[0]])
    return (data_processed, labels)

def remove_gravity(data):
    global acc_bias
    global acc_scale
    data[:,:3] = (data[:,:3] - acc_bias) / acc_scale
    data[:,3:6]= data[:,3:6] / 100.0
    for i in range(1000):
        ma.update(data[0][3:6], data[0][0:3], data[0][6:9])
    for i in range(len(data)):
        ma.update(data[i][3:6], data[i][0:3], data[i][6:9])
        my_qua = Quaternion(ma.quaternion._q[0], ma.quaternion._q[1], ma.quaternion._q[2], ma.quaternion._q[3])
        r = np.array(my_qua.rotation_matrix)
        data[i][:3] = np.dot(r, np.transpose(data[i][0:3])) - np.array([0,0,1])
    for i in range(len(data)):
        print(i, np.round(data[i][:6], decimals=5))
    return data

def get_features(data):
    features = []

    data_processed = data[1:,:].astype("float64")
    data_processed[:,:3] = data[1:,:3] - data[:-1,:3]

    features += data_processed[:,3:6].min(0).tolist()
    features += data_processed[:,3:6].max(0).tolist()
    features += data_processed[:,3:6].std(0).tolist()
    features += data_processed[:,3:6].mean(0).tolist()
    # features += np.sqrt((data_processed*data_processed/len(data_processed)).sum(0)).tolist()

    gro_d = data_processed[1:,3:6] - data_processed[:-1,3:6]
    features += gro_d.min(0).tolist()
    features += gro_d.max(0).tolist()
    features += gro_d.std(0).tolist()
    features += gro_d.mean(0).tolist()
    # features += np.sqrt((gro_d*gro_d/len(gro_d)).sum(0)).tolist()

    return features

def separate_data(data, labels):
    label_set = list(set(labels))
    test_x = []
    test_y = []
    while len(label_set) > 0:
        label = label_set.pop(0)
        count = 0
        for i in range(len(data)):
            if labels[i] == label:
                count = count + 1
                test_x.append(data.pop(i))
                test_y.append(labels.pop(i))
            if count == 2:
                break
    return (data, labels, test_x, test_y)

(data, labels) = read_data()
(train_x, train_y, test_x, test_y) = separate_data(data, labels)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
# print("------------------train features-------------------")
# for i in range(len(train_x)):
#     print(train_x[i], train_y[i])
# print("------------------test features-------------------")
# for i in range(len(test_x)):
#     print(test_x[i], test_y[i])


# clf = svm.SVC(C=5, tol=1e-5)
knn = KNeighborsClassifier(n_neighbors = 2)

# clf.fit(train_x, train_y)
knn.fit(train_x, train_y)

# y_pred = clf.predict(test_x)
y_pred = knn.predict(test_x)

# print("------------------train features-------------------")
# for i in range(len(train_x)):
#     print(train_x[i], train_y[i])
# print("------------------test features-------------------")
# for i in range(len(test_x)):
#     print(test_x[i], test_y[i])

print(y_pred)
