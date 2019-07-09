import numpy as np
import pandas as pd
class test():

	mean = 0
	std = 0

	def add(self, a, b):
		return a+b
	def perform_add(self, a, b):
		return self.add(a, b)

# path = "processed/test.txt"
# f = open(path, "w+")

import time
# st = time.time()
# for i in range(10000):
# 	a = np.array(b)

# st1 = time.time()

# for i in range(10000):
# 	b = a.tolist()

# st2 = time.time()
# print("time1: ", st1 - st, " time2: ", st2 - st1)
start_time = time.time()
data_pd = pd.read_csv("ML_line_0.txt")
data_np = data_pd[["acc1","acc2","acc3","gro1","gro2","gro3","mag1","mag2","mag3"]].values
data_np = data_np[1:] - data_np[:-1]
data_list = data_np.tolist()

# sample_num = 3
# step = 2
# data = []
# for i in range(len(data_list) - sample_num * step):
# 	temp_data = []
# 	for j in range(sample_num):
# 		temp_data += data_list[i+j*step]
# 	data.append(temp_data)
# end_time = time.time()
# for i in range(len(data)):
# 	print(data[i])

# print(end_time - start_time)

a = pd.read_csv("processed/test.csv")
print(a[1:-1])

