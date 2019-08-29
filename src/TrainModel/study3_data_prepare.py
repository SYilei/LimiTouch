
import pandas as pd
import os
import numpy as np

data_path = '../../data/Study3_raw_data/'
dataprocessed_path = '../../data/Study3_derivative/'

data_files = os.listdir(data_path)
columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

for file in data_files:
    if '.csv' in file:
        derivative = np.ones((29999, 9))
        data = pd.read_csv(data_path+file)[columns].values
        derivative[:,:6] = data[1:, :] - data[:-1, :]
        derivative[:,6:9] = data[:-1, 3:6]

        data_pd = pd.DataFrame(derivative)
        data_pd.to_csv(dataprocessed_path + file, index = False)
        

