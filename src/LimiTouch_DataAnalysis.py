import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import PrepareData as PD
import NN
import numpy as np
import sys

############ Set the parameter ############
LABELS = {"line":0, "sphere":1}
# LABELS = {"dot":0, "flat": 1, "still":2, "air":3}

N_OUTPUT = len(LABELS)

NUM_SAMPLE = 70
STEP_LENGTH = 1

NUM_HIDDEN = 80
NUM_BATCH = 10000
NUM_TRAIN = 1000

TRAIN_FILE_PATH = "processed/train_file.csv"
TEST_FILE_PATH = "processed/test_file.csv"
PARAMETERS_FILE_PATH = "processed/parameters.txt"



########### Get the data #############
pd = PD.PrepareData()

if len(sys.argv) == 1:
    para_file = open(PARAMETERS_FILE_PATH)
    parameters = para_file.readlines()
    pd.mean = np.array(list(map(float, parameters[0].rstrip()[1:-1].split(","))))
    pd.std = np.array(list(map(float, parameters[1].rstrip()[1:-1].split(","))))
    
# elif sys.argv[1] == "pd":

(train_data, test_data) = pd.prepare_data(NUM_SAMPLE, STEP_LENGTH, LABELS, TRAIN_FILE_PATH, TEST_FILE_PATH)

############# Save the parameters ####################
para_file = open(PARAMETERS_FILE_PATH,"w+")
para_file.write(str(pd.mean.tolist()) + "\n")
para_file.write(str(pd.std.tolist())+"\n")
para_file.close()

# print("Loading the data")
# data = pd.load_data(TRAIN_FILE_PATH)
# print("Getting the data")
(train_x, train_y) = pd.get_all(train_data)

net = NN.Net(n_feature = len(train_x[0]), n_hidden = NUM_HIDDEN, n_output = N_OUTPUT)
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()                                

for t in range(NUM_TRAIN):
    prediction = net(train_x)
    loss = loss_func(prediction, train_y)
    print(t,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

################ Test with the test set ##############
test_x = Variable(torch.FloatTensor(test_data[:,:-1]))
test_y = Variable(torch.LongTensor(test_data[:,-1]))

prediction = torch.max(F.softmax(net(test_x)),1)[1]
pred_y = prediction.data.numpy().squeeze()
y = test_y

print("------------------Check the accuracy------------------")
count = np.zeros((len(LABELS), len(LABELS)))
for i in range(len(pred_y)):
    count[test_y[i]][pred_y[i]] += 1
total = 0
for i in range(len(LABELS)):
    print("| accuracy " + str(i) + " : " + str(count[i][i] / count.sum(1)[i]) )
    total += count[i][i]
print("| accuracy total : ", total / count.sum())

time.sleep(1)


#################### real time detection ######################
import serial
ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
CHECKING_NUM = 10

# data = []
# while len(data) < 

while True:
    data = [[]]
    while len(data[0]) < 6 * NUM_SAMPLE:
        for i in range(STEP_LENGTH - 1): #Skip some of the data
            ser.readline()
        this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
        next_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
        data[0] += (np.array(next_read) - np.array(this_read)).tolist()
    
    data[0] = (np.array(data[0]) - pd.mean) / pd.std
    data = Variable(torch.FloatTensor(data))
    prediction = torch.max(F.softmax(net(data)),1)[1]
    pred_y = prediction.data.numpy().squeeze()
    print(pred_y)


