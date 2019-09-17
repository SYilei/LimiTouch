import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy.fft

path = '../../data/Study5_force_derivative/'
names = ['chamod','clarence','haimo','hussel','jiashuo','logan','sachith','samitha','shamane','tharindu','vipula','yilei']
numbers = ['1','2','3','4','5','6']
files = os.listdir(path)

for name in names:
	for number in numbers:
		print(path+name+'_high_force'+number+'.csv', path+name+'_low_force'+number+'.csv')
		data1 = pd.read_csv(path+name+'_high_force'+number+'.csv').values
		data2 = pd.read_csv(path+name+'_low_force'+number+'.csv').values
		fft1 = numpy.fft.fft(data1[:,0])
		fft2 = numpy.fft.fft(data2[:,0])

		plt.plot(fft1, label='high')
		plt.plot(fft2, label='low')
		plt.ylabel('some numbers')
		plt.show()