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
# LABELS = {'still':0, 'dot':1, 'flat':2, 'air': 3}

N_OUTPUT = len(LABELS)

NUM_SAMPLE = 50
STEP_LENGTH = 2

NUM_HIDDEN = 80
NUM_BATCH = 10000
NUM_TRAIN = 2000



TRAIN_FILE_PATH = "../data/processed/train_file.csv"
TEST_FILE_PATH = "../data/processed/test_file.csv"
PARAMETERS_FILE_PATH = "../models/parameters.txt"
SAVE_MODEL = "../models/model.txt"




########### Get the data #############
pd = PD.PrepareData()
(train_data, test_data) = pd.prepare_data(NUM_SAMPLE, STEP_LENGTH, LABELS, TRAIN_FILE_PATH, TEST_FILE_PATH)

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


#################### Save the data and the model ######################

torch.save(net, SAVE_MODEL)

para_file = open(PARAMETERS_FILE_PATH,"w+")
para_file.write(str(LABELS) + "\n")
para_file.write("NUM_SAMPLE:"+str(NUM_SAMPLE) + "\n")
para_file.write("STEP_LENGTH:"+str(STEP_LENGTH) + "\n")
para_file.write(str(pd.mean.tolist()) + "\n")
para_file.write(str(pd.std.tolist())+"\n")
para_file.close()


