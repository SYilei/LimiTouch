import serial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 

plt.ion()

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = []
ys = []
zs = []
while True:
	for i in range(10):
		ser.readline()
	this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
	xs.append(this_read[6])
	ys.append(this_read[7])
	zs.append(this_read[8])

	x = np.array(xs)
	y = np.array(ys)
	z = np.array(zs)

	x_off = (max(x)+min(x))/2
	y_off = (max(y)+min(y))/2
	z_off = (max(z)+min(z))/2

	x_range = (max(x)-min(x)) / 2
	y_range = (max(y)-min(y)) / 2
	z_range = (max(z)-min(z)) / 2

	ave_scale = (x_range+y_range+z_range) / 3
	
	plt.scatter((x - x_off) / x_range, (y - y_off) / y_range, c='tab:red', s = 4)
	plt.scatter((y - y_off) / y_range, (z - z_off) / z_range, c='tab:green', s = 4)
	plt.scatter((x - x_off) / x_range, (z - z_off) / z_range, c='tab:blue', s = 4)
	# plt.scatter(ave_scale / (x - x_off), ave_scale / (y - y_off), c='tab:red', s = 4)
	# plt.scatter(ave_scale / (y - y_off), ave_scale / (z - z_off), c='tab:green', s = 4)
	# plt.scatter(ave_scale / (x - x_off), ave_scale / (z - z_off), c='tab:blue', s = 4)
	plt.draw()
	plt.pause(0.001)
	plt.clf()

	print(x_off, y_off, z_off, x_range, y_range, z_range)


#53.0 2405.5 -2744.5 5193.333333333333