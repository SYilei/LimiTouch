from os import listdir
import random as rd
import numpy as np
import torch
import time
import pandas
from torch.autograd import Variable


################# Get the data##################
class PrepareData:
    def __init__(self):
        self.mean = 0
        self.std = 0

    def prepare_data(self, sample_num, step, labels, train_file_path, test_file_path):
        train_file = open(train_file_path,"w+")
        test_file = open(test_file_path,"w+")
        files = listdir()
        train_data = []
        test_data = []
        for file in files:
            if ("ML" in file) and (file.split("_")[1] in labels.keys()):
                st = time.time()
                
                # f = open(file, "r")
                # lines = f.readlines()
                # x = []
                # y = []
                # lines_num = len(lines) - (sample_num - 1) * step - 5
                # for i in range(lines_num):
                #     this_line = []
                #     next_line = []
                #     for j in range(sample_num):
                #         this_line += list(map(float, lines[i+step*j].split(",")[1:7]))
                #         next_line += list(map(float, lines[i+step*j + 1].split(",")[1:7]))
                #     x.append((np.array(next_line) - np.array(this_line)).tolist())
                #     y.append(labels[lines[i].rstrip().split(",")[-1]])

                data_pd = pandas.read_csv(file)
                data_np = data_pd[["acc1","acc2","acc3","gro1","gro2","gro3"]].values
                data_np = data_np[1:] - data_np[:-1]
                data_list = data_np.tolist()

                x = []
                for i in range(len(data_list) - sample_num * step):
                    temp_data = []
                    for j in range(sample_num):
                        temp_data += data_list[i + j * step]
                    x.append(temp_data)

                st1 = time.time()
                print("----------------------------")
                print("sorting data", (st1 - st))
                y = (np.ones(len(x)) * labels[file.split("_")[1]]).tolist()
                (train_data_temp, test_data_temp) = self.separate_data(x,y)
                
                train_data += train_data_temp
                test_data += test_data_temp
                
                print("separate data", time.time() - st1)
                print("Finished " + file)   

        print("Normalizing data")
        st = time.time()

        train_data = np.array(train_data).astype("float64")
        test_data = np.array(test_data).astype("float64")
        
        self.mean = train_data[:,:-1].mean(0)
        self.std = train_data[:,:-1].std(0)
        
        train_data[:,:-1] = (train_data[:,:-1] - self.mean) / self.std
        test_data[:,:-1] = (test_data[:,:-1] - self.mean) / self.std
        print("Normalizing uses: ", time.time() - st)
        return (train_data, test_data)

        # print("Saving the training data")
        # train_data = pandas.DataFrame(train_data)
        # test_data = pandas.DataFrame(test_data)
        # train_data.to_csv(train_file_path)
        
        # print("Saving the training data")
        # test_data.to_csv(test_file_path)

        # print("Finished save the data!")

        # data = ""
        # for i in range(len(train_data)):
        #     temp = str(train_data[i])
        #     data += str(i)+","+temp[1:len(temp)-1]+"\n"
        # train_file.write(data)    
        # train_file.close()

        # data = ""
        # for i in range(len(test_data)):
        #     temp = str(test_data[i])
        #     data += str(i)+","+temp[1:len(temp)-1]+"\n"
        # test_file.write(data)       
        # test_file.close()           
        
        
    
    def normalize(self, data):
        # mean = []
        # std = []
        # for i in range(len(data) - 1): 
        #     mean.append(np.mean(data[i]))
        #     std.append(np.std(data[i]))
        #     data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])

        self.mean = data[:,:-1].mean(0)
        self.std = data[:,:-1].std(0)

        data[:,:-1] = (data[:,:-1] - self.mean) / self.std
        return data

    def separate_data(self, x, y):
        train_data = []
        test_data = []

        while len(train_data) // 2 < len(x):
            a = rd.randint(0,len(x)-1)
            newdata = x.pop(a)
            newdata.append(y.pop(a))
            train_data.append(newdata)

        while len(x) > 0:
            a = rd.randint(0,len(x)-1)
            newdata = x.pop(a)
            newdata.append(y.pop(a))
            test_data.append(newdata)
        return (train_data, test_data)

    def load_data(self, path):
        # f = open(path, "r")
        return pandas.read_csv(path)

    def get_batch(self, data, batch_num):
        batch_x = []
        batch_y = []
        while len(batch_x) < batch_num:
            a = rd.randint(0, len(data) - 1)
            data_float = list(map(float, data[a].split(",")))
            batch_x.append(data_float[1:-1])
            batch_y.append(data_float[-1])
        batch_x = Variable(torch.FloatTensor(batch_x))
        batch_y = Variable(torch.LongTensor(batch_y))
        return (batch_x, batch_y)
    
    def get_all(self, data):
        x = Variable(torch.FloatTensor(data[:,0:-1].tolist()))
        y = Variable(torch.LongTensor(data[:,-1].tolist()))
        return (x, y)

    def get_test_data(self):
        f = open("processed/test_file.txt", "r")
        lines = f.readlines()
        x = []
        y = []
        for i in range(len(lines)):
            data = list(map(float,lines[i].split(",")))
            x.append(data[1:-1])
            y.append(data[-1])
        x = (np.array(x) - self.mean) / self.std
        x = np.array(x)
        y = np.array(y)

        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.LongTensor(y))

        return (x, y)

# labels = {"air":0, "flat":1}
# pd = PrepareData()
# pd.prepare_data(20, 5, labels)
