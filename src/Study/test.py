import matplotlib.pyplot as plt
import time
import random as rd
 
 
x = [i+10 for i in range(10)]
y1 = [rd.random() for i in range(10)]
y2 = [rd.random() for i in range(10)]
plt.ion()

axes = plt.gca()
line1, = axes.plot(x, y1, 'r-')
line2, = axes.plot(x, y2, 'r-')
for i in range(100):
	y1 = [rd.random() for i in range(10)]
	y2 = [rd.random() for i in range(10)]
	line1.set_ydata(y1)
	line2.set_ydata(y2)
	plt.draw()
	plt.pause(1e-17)
	# time.sleep(0.1)
	input('next?')