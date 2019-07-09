import serial
import time
import numpy as np
import pyqtgraph as pg

ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 

datas = [np.array([1,1,1,1,1,1]),np.array([1,1,1,1,1,1])]
changes = [max(abs(datas[1] - datas[0]))]


while True:
    print(list(map(int, ser.readline().decode('utf-8').split(",")))[0:6])

    # dataByte = list(ser.read(13))
    # print(dataByte)
    # data = []
    # while len(dataByte) > 1:
    #     data.append(int.from_bytes([dataByte.pop(0),dataByte.pop(0)], "big", signed = True))
    # temp = np.array(data)
    # print(data)
    

    # changes.append(max(abs(datas[1] - datas[0])))
    # if len(changes) > 200:
    #     changes.pop(0)
    # if changes[-1] > 1800:
    #     print(changes[-1])
    # pg.plot(changes)
# ax.clear()    
# ax.plot(changes)

