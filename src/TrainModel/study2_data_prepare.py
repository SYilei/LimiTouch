
import pandas as pd
import os

data_path = '../../data/Data_Study2/'
dataindex_path = '../../data/DataIndex_Study2/'
dataprocessed_path = '../../data/DataProcessed_Study2/'
name_list = []
count = 0
index = 0
data_files = os.listdir(data_path)
dataindex_files = os.listdir(dataindex_path)
data_list = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

#get name of participants
for filename in data_files:
    if filename.find('data') != -1:
        name = filename.split('_')[-1].split('.')[0]
        name_list.append(name)

for name in name_list:
    print(str(count) + '.' + name)
    count += 1


# name_num = input("Input the num of the name data you want to process:")
# print('----Begin to prepare ' + name_list[int(name_num)] + ' data----')
#get the data and dataindex file of participant

for name in name_list:
    data_file = ''
    index_file = ''
    for filename in data_files:
        if ('_'+name) in filename:
            data_csv = pd.read_csv(data_path + filename)
            
            print('found:', filename)
    for filename in dataindex_files:
        if ('_'+name) in filename:
            dataindex_csv = pd.read_csv(dataindex_path + filename, header=None)
            print('found:', filename)


    index = 0
    while index < dataindex_csv.shape[0]:
        dataindex_min = dataindex_csv[1][index]
        dataindex_max = dataindex_csv[2][index]
        temp = data_csv.loc[dataindex_min:dataindex_max-1] #select data by index
        temp.loc[:, ['mx']] = temp['gx']
        temp.loc[:, ['my']] = temp['gy']
        temp.loc[:, ['mz']] = temp['gz']
        for listname in data_list:
            temp[listname] = temp[listname] - temp[listname].shift(1)
            temp[listname] = temp[listname].shift(-1)
        
        temp = temp.drop(temp.index[len(temp)-1])
        
        if index == 0:
            temp.to_csv(dataprocessed_path+"data_processed_" + name, index=False)
        else:
            temp.to_csv(dataprocessed_path+"data_processed_" + name, mode='a', header=False, index=False)
        index += 1

        print(index)

