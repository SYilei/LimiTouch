import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
import random as rd
from pyquaternion import Quaternion
import madgwickahrs as mg
import serial
import time
import math
# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
import scipy.signal
import keyboard

# def Cube():
#     glBegin(GL_LINES)
#     for edge in edges:
#         for vertex in edge:
#             glVertex3fv(final_verticies[vertex])
#     glEnd()

# pygame.init()
# display = (800,600)
# pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
# gluPerspective(45, (display[0]/display[1]), 0.1, 10.0)
# glTranslatef(0.0,0.0, -5)


#### for calibration
def calibrate():
    data1 = [-930.0,111.0,-249.0]
    data2 = [224.0, 928.0,-322.0]
    data3 = [56.0, 570.0, 791.0]
    data4 = [175.0, 41.0, 958.0]

    data1.append(0)
    data2.append(0)
    data3.append(0)
    data4.append(0)

    data1 = np.array(data1).astype("float64")
    data2 = np.array(data2).astype("float64")
    data3 = np.array(data3).astype("float64")
    data4 = np.array(data4).astype("float64")

    abc_1 = (-2*data1) - (-2*data2)
    abc_2 = (-2*data2) - (-2*data3)
    abc_3 = (-2*data3) - (-2*data4)
    b_1 = np.sum(data1*data1) - np.sum(data2*data2)
    b_2 = np.sum(data2*data2) - np.sum(data3*data3)
    b_3 = np.sum(data3*data3) - np.sum(data4*data4)
    abc_1[3] = b_1
    abc_2[3] = b_2
    abc_3[3] = b_3

    abc_1 = abc_1 / abc_1[0]
    abc_2 = abc_2 / abc_2[0]
    abc_3 = abc_3 / abc_3[0]

    bc_1 = abc_1 - abc_2
    bc_2 = abc_2 - abc_3

    bc_1 = bc_1 / bc_1[1]
    bc_2 = bc_2 / bc_2[1]

    c_m = bc_1 - bc_2

    c = - c_m[3] / c_m[2]
    b = - bc_1[3] - bc_1[2] * c
    a = - abc_1[3] - abc_1[2] * c - abc_1[1] * b

    print(a, b, c)

# calibrate()
# exit()

def collect_data():

    acc_bias = np.array([x_bias, y_bias, z_bias])
    acc_scale = np.array([x_scale, y_scale, z_scale])

    file  = open("data.csv","w+")
    data = ""
    for i in range(500):
        print(i, ser.readline())
    st = time.time()
    for i in range(1000):
        this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
        this_read = np.array(this_read)
        this_read[:3] = -(this_read[:3] - acc_bias) / acc_scale
        this_read[3:] /= 100.0 
        switch = np.array([0,0,0]).astype("float64")
        switch += this_read[3:6] * 180.0 / math.pi 
        this_read[3:6] = this_read[3:6] - this_read[3:6] + this_read[:3]
        this_read[:3] = switch
        # print(this_read)
        file.write(str(this_read.tolist())[1:-1]+"\n")
        if i%100 == 0:
            print(i)
    en = time.time()
    print(en - st)

    file.close
    exit(0)

# collect_data()

def localize(acc):
    order = 1
    cut_off = 0.1
    [b, a] = scipy.signal.butter(order, (2*cut_off) / (1/sample_period), 'high')

    vel_lin = np.zeros((len(acc), 3))
    for i in range(len(acc)-1):
        vel_lin[i+1] = vel_lin[i] + acc[i+1] * sample_period
    vel_hp = scipy.signal.filtfilt(b, a, vel_lin, axis=0)

    pos_lin = np.zeros((len(vel_hp),3))
    for i in range(len(vel_hp)-1):
        pos_lin[i+1] = pos_lin[i] + vel_hp[i+1] * sample_period
    pos_hp = scipy.signal.filtfilt(b, a, pos_lin, axis=0)
    return pos_hp


############## Setup the parameters ###############

ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
ma = mg.MadgwickAHRS(sampleperiod= 1.0 / 226.54)
sample_period = 1.0 / 226.54
count = 0

l = 0.5
verticies = [[l, -l, -l],[l, l, -l],[-l, l, -l],[-l, -l, -l],[l, -l, l],[l, l, l],[-l, -l, l],[-l, l, l]]
final_verticies = [[1, -1, -1],[1, 1, -1],[-1, 1, -1],[-1, -1, -1],[1, -1, 1],[1, 1, 1],[-1, -1, 1],[-1, 1, 1]]
edges = [[0,1],[0,3],[0,4],[2,1],[2,3],[2,7],[6,3],[6,4],[6,7],[5,1],[5,4],[5,7]]


x_acc_min = -960.0
x_acc_max = 1008.0
y_acc_min = -970.0
y_acc_max = 995.0
z_acc_min = -1030.0
z_acc_max = 970.0

x_bias = (x_acc_max+x_acc_min)/2.0
x_scale = (x_acc_max-x_acc_min)/2.0
y_bias = (y_acc_max+y_acc_min)/2.0
y_scale = (y_acc_max-y_acc_min)/2.0
z_bias = (z_acc_max+z_acc_min)/2.0
z_scale = (z_acc_max-z_acc_min)/2.0

acc_bias = np.array([x_bias, y_bias, z_bias])
acc_scale = np.array([x_scale, y_scale, z_scale])

############## Start the code ###############
collect_data()
exit(0)


data = []
status = False
status_pre = False

while True:
    status_pre = status
    status = keyboard.is_pressed('q')
    if status:
        try:
            data.append(list(map(float, ser.readline().decode('utf-8').split(",")))[0:9])
        except:
            continue
    
    elif status_pre:
        data_np = np.array(data) 
        data_np[:,0:3] = (data_np[:,0:3] - acc_bias) / acc_scale 
        data_np = data_np / 100.0
        new_acc = np.zeros((len(data_np), 3))
        for i in range(len(data_np)):
            ma.update(data_np[i][3:6], data_np[i][0:3], data_np[i][6:9])
            my_qua = Quaternion(ma.quaternion._q[0], ma.quaternion._q[1], ma.quaternion._q[2], ma.quaternion._q[3])
            r = np.array(my_qua.rotation_matrix)
            new_acc[i] = (np.dot(r, np.transpose(data_np[i][0:3])) - np.array([0,0,1])) * 9.81
        # make1_qua = Quaternion(axis = [0,0,1], angle = -3.14159265 / 2.0)
        # make2_qua = Quaternion(axis = [1,0,0], angle = -3.14159265 / 2.0)
        # print(len(new_acc))
        positions = localize(new_acc)
        positions = np.array(positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:,0],positions[:,1],positions[:,2], marker='o')
        plt.show()
    
    elif not status_pre:
        print(ser.readline())
        data.clear()

    # print(my_qua)


    ####### Visualization #####
    # for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()
    # count += 1
    # if count % 5 == 0:
    #     acc_list = np.array(acc_list)
    #     vel_list = np.array(vel_list)
    #     pos_list = np.array(pos_list)
    #     print(count)
    #     dx = 0
    #     dy = 0
    #     dz = 0
    #     if count > 2000:
    
    #     for i in range(len(final_verticies)):
    #         final_verticies[i] = my_qua.rotate(verticies[i])
    #         final_verticies[i] = make1_qua.rotate(final_verticies[i])
    #         final_verticies[i] = make2_qua.rotate(final_verticies[i])
    #     glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    #     Cube()
    #     pygame.display.flip()
        # pygame.time.wait(10) 
        
        
    