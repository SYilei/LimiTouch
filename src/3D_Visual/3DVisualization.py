import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
import random as rd
from pyquaternion import Quaternion
import madgwickahrs as mg
import time
import math
<<<<<<< HEAD
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
=======
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from utils import get_serial
from double_integral import DoubleIntegrator3D
import atexit

# prints the double integrated position to the console
def print_trajectory():
    coord_by_axis = tuple(zip(*trajectory))
    print("\n".join(map(repr, trajectory)))
    print("\nRange: [{}, {}], [{}, {}], [{}, {}]".format(
        min(coord_by_axis[0]),
        max(coord_by_axis[0]),
        min(coord_by_axis[1]),
        max(coord_by_axis[1]),
        min(coord_by_axis[2]),
        max(coord_by_axis[2]),
        ))
# Uncomment the statement below to print the double integration results
# when the program exits.
#atexit.register(print_trajectory)

ser = get_serial(None, 115200)
sample_period = 1.0 / 200
ma = mg.MadgwickAHRS(sampleperiod=sample_period)
count = 0

x_bias = 29.69433
y_bias = 3.20569
z_bias = -22.26908

l = 0.5
verticies = [[l, -l, -l],[l, l, -l],[-l, l, -l],[-l, -l, -l],[l, -l, l],[l, l, l],[-l, -l, l],[-l, l, l]]
# ini_qua = Quaternion(0.05, -0.116, 0.984, 0.001)
# for i in range(len(verticies)):
#     verticies[i] = ini_qua.rotate(verticies[i])
final_verticies = [[1, -1, -1],[1, 1, -1],[-1, 1, -1],[-1, -1, -1],[1, -1, 1],[1, 1, 1],[-1, -1, 1],[-1, 1, 1]]
edges = [[0,1],[0,3],[0,4],[2,1],[2,3],[2,7],[6,3],[6,4],[6,7],[5,1],[5,4],[5,7]]



def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(final_verticies[vertex])
    glEnd()



pygame.init()
display = (800,600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
gluPerspective(45, (display[0]/display[1]), 0.1, 10.0)
glTranslatef(0.0,0.0, -5)
>>>>>>> d5f6e9d92c857463a9cab968c01033398a3b410a


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

<<<<<<< HEAD
    vel_lin = np.zeros((len(acc), 3))
    for i in range(len(acc)-1):
        vel_lin[i+1] = vel_lin[i] + acc[i+1] * sample_period
    vel_hp = scipy.signal.filtfilt(b, a, vel_lin, axis=0)
=======
itg = DoubleIntegrator3D(spacing=sample_period)
trajectory = tuple()

while True:
    en = time.time()
    print(1.0 / (en - st))
    st = time.time()
>>>>>>> d5f6e9d92c857463a9cab968c01033398a3b410a

    pos_lin = np.zeros((len(vel_hp),3))
    for i in range(len(vel_hp)-1):
        pos_lin[i+1] = pos_lin[i] + vel_hp[i+1] * sample_period
    pos_hp = scipy.signal.filtfilt(b, a, pos_lin, axis=0)
    return pos_hp

<<<<<<< HEAD

############## Setup the parameters ###############
=======
    ####### 3D tracking #######
    x_ini = [1,0,0]
    y_ini = [0,1,0]
    z_ini = [0,0,1]
    
    g_ini = np.array([0,0,-1])

    x = np.array(my_qua.rotate(x_ini))
    y = np.array(my_qua.rotate(y_ini))
    z = np.array(my_qua.rotate(z_ini))

    angle_x = math.acos( np.sum(x * g_ini) / (math.sqrt(np.sum(x * x)) * math.sqrt(np.sum(g_ini * g_ini))))
    angle_y = math.acos( np.sum(y * g_ini) / (math.sqrt(np.sum(y * y)) * math.sqrt(np.sum(g_ini * g_ini))))
    angle_z = math.acos( np.sum(z * g_ini) / (math.sqrt(np.sum(z * z)) * math.sqrt(np.sum(g_ini * g_ini))))
    # print("----------------------")
    # print(angle_x, angle_y, angle_z)

    g = 985.0
    acc_x = this_read[0] + g * math.cos(angle_x) - 15
    acc_y = this_read[1] + g * math.cos(angle_y) - 10
    acc_z = this_read[2] + g * math.cos(angle_z) + 20

    dx = v_x
    dy = v_y
    dz = v_z

    if acc_x > 30:
        v_x += acc_x
        dx += 0.5 * acc_x
    if acc_y > 30:
        v_y += acc_y
        dy += 0.5 * acc_y
    if acc_z > 30:
        v_z += acc_z
        dz += 0.5 * acc_z
    
    p_x += dx
    p_y += dy
    p_z += dz
    print(round(acc_x, 2), "\t", round(acc_y, 2), "\t", round(acc_z, 2))
    # print(angle_x, math.cos(angle_x), angle_y, math.cos(angle_y), angle_z, math.cos(angle_z))
    # print(this_read)
    # print("--------------------------")

    print("acc: ", round(acc_x,2),round(acc_y,2),round(acc_z,2))
    # print("vel: ", round(v_x, 2), round(v_y, 2), round(v_z, 2))
    # print("d_x: ", round(dx, 2), round(dy, 2), round(dz, 2))
    # print("pos: ", round(p_x, 2), round(p_y, 2), round(p_z, 2))
>>>>>>> d5f6e9d92c857463a9cab968c01033398a3b410a

ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
ma = mg.MadgwickAHRS(sampleperiod= 1.0 / 226.54)
sample_period = 1.0 / 226.54
count = 0

<<<<<<< HEAD
l = 0.5
verticies = [[l, -l, -l],[l, l, -l],[-l, l, -l],[-l, -l, -l],[l, -l, l],[l, l, l],[-l, -l, l],[-l, l, l]]
final_verticies = [[1, -1, -1],[1, 1, -1],[-1, 1, -1],[-1, -1, -1],[1, -1, 1],[1, 1, 1],[-1, -1, 1],[-1, 1, 1]]
edges = [[0,1],[0,3],[0,4],[2,1],[2,3],[2,7],[6,3],[6,4],[6,7],[5,1],[5,4],[5,7]]

=======
    # double integration to obtain position
    acc = (acc_x, acc_y, acc_z)  # 3D vector of acceleration
    pos = itg.update(acc)
    trajectory += tuple([pos])
>>>>>>> d5f6e9d92c857463a9cab968c01033398a3b410a

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
        
        
    
