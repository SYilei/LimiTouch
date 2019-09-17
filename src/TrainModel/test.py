import os
import pandas as pd

path = '../../data/Study3_raw_data/'

files = os.listdir(path)

for file in files:
	if '.csv' in file:
		data = pd.read_csv(path+file)
		new_data = data[['ax','ay','az','gx','gy','gz','mx','my','mz']]
		new_data.to_csv(path+file, index=False)
