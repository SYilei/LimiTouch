import torch
import numpy as np
import os
import sys
import pandas as pd
import random
from study1_modelclass import LeNet
import matplotlib.pyplot as plt
import plotly.graph_objects as go

touch_data = pd.read_csv('../../data/Data_Study1/chamod_touch_free_circle.csv')[['ax','ay','az','gx','gy','gz']].values
nontouch_data = pd.read_csv('../../data/Data_Study1/chamod_nontouch_free_circle.csv')[['ax','ay','az','gx','gy','gz']].values

# touch_data = pd.read_csv('../../data/Data_Study1_Difference/chamod_touch_free_circle.csv')[['0','1','2','3','4','5']].values
# nontouch_data = pd.read_csv('../../data/Data_Study1_Difference/chamod_nontouch_free_circle.csv')[['0','1','2','3','4','5']].values

n = 2000
random_x = [i for i in range(n)]
# random_y0 = touch_data[:n,0]
# random_y1 = touch_data[:n,1]
# random_y2 = touch_data[:n,2]
# random_y3 = touch_data[:n,3]
# random_y4 = touch_data[:n,4]
# random_y5 = touch_data[:n,5]

random_y0 = nontouch_data[:n,0]
random_y1 = nontouch_data[:n,1]
random_y2 = nontouch_data[:n,2]
random_y3 = nontouch_data[:n,3]
random_y4 = nontouch_data[:n,4]
random_y5 = nontouch_data[:n,5]

# print(random_y0)
# quit(0)
# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='acc_x'))

fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines',
                    name='acc_y'))

fig.add_trace(go.Scatter(x=random_x, y=random_y2,
                    mode='lines',
                    name='acc_z'))

fig.add_trace(go.Scatter(x=random_x, y=random_y3,
                    mode='lines',
                    name='gyro_x'))

fig.add_trace(go.Scatter(x=random_x, y=random_y4,
                    mode='lines',
                    name='gyro_y'))

fig.add_trace(go.Scatter(x=random_x, y=random_y5,
                    mode='lines',
                    name='gyro_z'))
fig.show()
