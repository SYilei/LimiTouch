import json
import serial
import torch
import ast
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import NN

ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
CHECKING_NUM = 10
PARAMETERS_FILE_PATH = "../models/parameters.txt"
MODEL_PATH = "../models/model.txt"

##################### Load model and parameters ###########
para_file = open(PARAMETERS_FILE_PATH)
parameters = para_file.readlines()

for item in parameters:
    print(item)

LABELS = ast.literal_eval(parameters[0].rstrip())
NUM_SAMPLE = int(parameters[1].rstrip().split(":")[1])
STEP_LENGTH = int(parameters[2].rstrip().split(":")[1])
mean = np.array(list(map(float, parameters[3].rstrip()[1:-1].split(","))))
std = np.array(list(map(float, parameters[4].rstrip()[1:-1].split(",")))) 

net = torch.load(MODEL_PATH)
net.eval()

while True:
    data = [[]]
    while len(data[0]) < 6 * NUM_SAMPLE:
        for i in range(STEP_LENGTH - 1): #Skip some of the data
            ser.readline()
        this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
        next_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
        data[0] += (np.array(next_read) - np.array(this_read)).tolist()
    
    data[0] = (np.array(data[0]) - mean) / std
    data = Variable(torch.FloatTensor(data))
    prediction = torch.max(F.softmax(net(data)),1)[1]
    pred_y = prediction.data.numpy().squeeze()
    print(pred_y)