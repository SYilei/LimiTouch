import os
import pandas as pd

path = '../../data/Study3_raw_data/'

files = os.listdir(path)

for file in files:
	data = pd.read_csv(path+file)
	data['mx'] = data['ax']
	data['my'] = data['ay']
	data['mz'] = data['az']
	data.to_csv(path+file)