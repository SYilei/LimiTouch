from study1_modelclass import LeNet
import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cpu')
model = LeNet()
try:
	model.load_state_dict(torch.load('../../models/new_step_1_size_250.txt', map_location=device))
	print('load model successfully!')
except Exception as e:
	print('there is something wrong!')
	raise e


# path = '../../data/Study1_derivative/'
path = '../../data/Study3_derivative/'

data_files = os.listdir(path)
column = ['0','1','2','3','4','5','6','7','8'] 
for file in data_files:
	
	if '.csv' in file and '_touch_' in file:
		data = pd.read_csv(path + file)[column].values
		correct = 0
		total = 0
		for i in range(0, len(data) - 250, 10):
			x = np.array(data[i:i+250]).reshape((-1, 1, 9, 250))
			x = Variable(torch.FloatTensor(x))
			prediction = torch.max(F.softmax(model(x)), 1)[1]

			total = total + 1
			y_pred = prediction.data.numpy().squeeze()
			if y_pred == 1:
				correct += 1
		print(file, correct / total)
			
