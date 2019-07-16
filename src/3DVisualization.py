import numpy as np
import ahrs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd
from pyquaternion import Quaternion
import madgwickahrs as mg
import serial
import time
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
ma = mg.MadgwickAHRS(sampleperiod= 1.0 / 213.0)
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

# while True:

#     data = list(map(float, ser.readline().decode('utf-8').split(",")))[0:3]
#     value = (data[0]-17.34766)*(data[0]-17.34766)+(data[1]-2.85044)*(data[1]-2.85044)+(data[2]+20.40367)*(data[2]+20.40367)
#     count += 1
#     if count == 100:
#         count = 0
#         print(math.sqrt(value))

mag_off = np.array([423.0, 2500.5, -2600.0])
mag_scale = np.array([5201.0, 5173.5, 5352.0])
st = 0
en = 0

v_x = 0
v_y = 0
v_z = 0

p_x = 0
p_y = 0
p_z = 0

# for i in range(2500):
#     print(ser.readline())

while True:
    en = time.time()
    # print(en - st)
    st = time.time()

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
    # print(this_read)
    ma.update(np.array([this_read[3]/100, this_read[4]/100, this_read[5]/100]), [this_read[0] - x_bias, this_read[1] - y_bias, this_read[2] - z_bias], (np.array(this_read[6:9]) - mag_off) / mag_scale)
    # ma.update_imu([this_read[3]/100, this_read[4]/100, this_read[5]/100], [this_read[0], this_read[1], this_read[2]])
    my_qua = Quaternion(ma.quaternion._q[0], ma.quaternion._q[1], ma.quaternion._q[2], ma.quaternion._q[3])
    make1_qua = Quaternion(axis = [0,0,1], angle = -3.14159265 / 2.0)
    make2_qua = Quaternion(axis = [1,0,0], angle = -3.14159265 / 2.0)
    # print(my_qua)

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
    acc_x = this_read[0] + g * math.cos(angle_x) - 30
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
    print(round(acc_x, 2), "/t", round(acc_y, 2), "\t", round(acc_z, 2))
    # print(angle_x, math.cos(angle_x), angle_y, math.cos(angle_y), angle_z, math.cos(angle_z))
    # print(this_read)
    # print("--------------------------")

    # print("acc: ", round(acc_x,2),round(acc_y,2),round(acc_z,2))
    # print("vel: ", round(v_x, 2), round(v_y, 2), round(v_z, 2))
    # print("d_x: ", round(dx, 2), round(dy, 2), round(dz, 2))
    # print("pos: ", round(p_x, 2), round(p_y, 2), round(p_z, 2))




    ####### Visualization #####
    count += 1
    if count == 5:
        count = 0
        for i in range(len(final_verticies)):
            final_verticies[i] = my_qua.rotate(verticies[i])
            final_verticies[i] = make1_qua.rotate(final_verticies[i])
            final_verticies[i] = make2_qua.rotate(final_verticies[i])
            # final_verticies[i][0] += (x / 1000.0)
            # final_verticies[i][1] += (y / 1000.0)
            # final_verticies[i][2] += (z / 1000.0)
        # print(final_verticies)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        # pygame.time.wait(10)  

        
    