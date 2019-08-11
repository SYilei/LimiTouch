
import os
import numpy as np
import pandas

touch_condition = ['_touch','_nontouch']
elbow_condition = ['_elbow','_free']
orientation_condition = ['_horizontal', '_vertical', '_slash', '_backslash', '_circle', '_static']

data_path = '../../data/'

all_file_names = os.listdir(data_path+'Data_Study1/')

for touch in touch_condition:
    for elbow in elbow_condition:
        for orientation in orientation_condition:
            for file_name in all_file_names:
                if touch in file_name and elbow in file_name and orientation in file_name:
                    print('read file: ' + file_name)
                    temp_data_pd = pandas.read_csv(data_path + "Data_Study1/" + file_name)
                    temp_data_np = temp_data_pd[["ax","ay","az","gx","gy","gz"]].values
                    data_np_diff = np.zeros((len(temp_data_np)-1, 9))
                    data_np_diff[:,0:3] = temp_data_np[1:,0:3] - temp_data_np[:-1,0:3]
                    data_np_diff[:,3:6] = temp_data_np[1:,3:6] - temp_data_np[:-1,3:6]
                    data_np_diff[:,6:9] = temp_data_np[1:,3:6]
                    data_pandas = pandas.DataFrame(data_np_diff)
                    data_pandas.to_csv(data_path + 'Data_Study1_Difference/'+file_name, index= False)
                    print('finished: ' + file_name)
                    print('There are ' + str(len(data_pandas)) + 'lines of data')
            



