import serial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 


data = [[],[],[]]
for i in range(500):
	this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
	next_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
	all_read = np.array(next_read) - np.array(this_read)
	data[0].append(all_read[0])
	data[1].append(all_read[1])
	data[2].append(all_read[2])
while True:
	for i in range(100):
		this_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
		next_read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:6]
		all_read = np.array(next_read) - np.array(this_read)
		data[0].append(all_read[0])
		data[0].pop(0)
		# data[1].append(all_read[1])
		# data[2].append(all_read[2])
	
	data_fft1 = np.fft.rfft(data[0])
	# data_fft2 = np.fft.rfft(data[1])
	# data_fft3 = np.fft.rfft(data[2])
	plt.cla()
	plt.ylim(-2000, 2000)
	plt.plot(data_fft1.real)
	plt.pause(0.001)
plt.show()
	# plt.plot( data=data_fft2, color='green', linewidth=1)
	# plt.plot( data=data_fft3, color='blue', linewidth=1)
# plt.show()


