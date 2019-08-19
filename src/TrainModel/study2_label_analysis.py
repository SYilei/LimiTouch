import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

def find_period(data):
	zero = 0
	zero_one = 0
	one = 0
	count = 0
	zero_flag = False
	count_flag = False
	st = 0
	en = 0
	gest_st_en = []
	for i in range(len(data)):
		if data[i] == 1:
			one += 1
			if not count_flag:
				st = i
				count_flag = True
			if zero_flag:
				zero_one += 1
		
		if count_flag:
			if data[i] == 0:
				zero += 1
				if zero_flag == False:
					zero_flag = True
					en = i

			if zero_flag:
				if zero > 100:
					if one > 400:
						count += 1
						gest_st_en.append([st, i, en])
					one = 0
					zero = 0
					zero_one = 0
					zero_flag = False
					count_flag = False

				elif zero_one > zero:
					zero = 0
					zero_one = 0
					zero_flag = False
	# print(one)
	# print(zero_flag)
	if zero_flag and one > 400:
		gest_st_en.append([st, en, en])
		count += 1
	return count, gest_st_en


files = os.listdir('../../data/DataLabels_Study2/')
name_list = ['_chamod','_clarence','_evan','_hussel','_jing',\
            '_kaixing','_logan','_mevan','_sachith','_samantha'\
            ,'_samitha','_vipula','_yilei','_yiyue']

if len(sys.argv) > 1:
	name_list = ['_'+sys.argv[-1]]

plt.ion()

axes = plt.gca()
line1, = axes.plot([], [], color = 'green')
line2, = axes.plot([], [], color = 'red')
line3, = axes.plot([], [], color = 'blue')

ymin = -100
ymax = 100
check_plot = True
print('-------------------------')
for name in name_list:
	for data_file in files:
		if name in data_file and 'label_' in data_file:
			label_list = pd.read_csv('../../data/DataLabels_Study2/' + data_file)[['1']].values
			original_index = pd.read_csv('../../data/DataLabels_Study2/' + data_file)[['2']].values
			original_data = pd.read_csv('../../data/DataProcessed_Study2/data_processed' + name)[['ax','ay','az']].values
			for index_file in files:
				if name in index_file and 'labelindex_' in index_file:
					data_info = open('../../data/DataLabels_Study2/' + index_file).readlines()[1:]
					# (correct, fail_list, more_list) = process(label_list, data_info,name,original_index,original_data)
					fail_list = []
					more_list = []
					good_list = []
					correct = 0
					for line in data_info:
						letter = line.split(',')[1]
						st = int(line.split(',')[2])
						en = int(line.split(',')[3])
						data = label_list[st : en]
						gest_num, gest_st_en = find_period(data)
						final_gest_num = 0
						final_gest_st_en = []

						if gest_num > 1:
							original_dataclip = (np.array(gest_st_en) + original_index[st]).tolist()
							for i in range(len(original_dataclip)):
								t = (original_data[original_dataclip[i][0]:original_dataclip[i][1]]**2).sum(1)
								t.sort()
								sm = t[:int(0.8*len(t))].mean()
								if sm > 10:
									final_gest_num += 1
									final_gest_st_en.append([gest_st_en[i][0]+250,gest_st_en[i][2]])
						elif gest_num == 1:
							final_gest_num = gest_num
							final_gest_st_en.append([gest_st_en[0][0]+250,gest_st_en[0][2]])
						

						if final_gest_num == 0:
							fail_list.append([letter, st, final_gest_num])
							# for i in range(len(data)):
							# 	print(i+st, letter, data[i], name, st, 'label file index: ', i + st, 'original file index: ', i + original_index[st])
							if check_plot:
								print(len(fail_list)+len(more_list)+correct, letter, 'no gesture found')
						
						elif final_gest_num > 1:
							more_list.append([letter, st, final_gest_num])
							original_dataclip = (np.array(final_gest_st_en) + original_index[st]).tolist()
							# for i in range(len(data)):
							# 	print(i+st, letter, data[i], name, st, 'label file: ',(np.array(gest_st) + st).tolist(), 'original file: ', original_dataclip)
							if check_plot:
								for i in range(len(original_dataclip)):
									plt.scatter(original_dataclip[i], [-80+i*10, -80+i*10])
								# print(letter, original_dataclip)
								print(len(fail_list)+len(more_list)+correct, letter, 'more gesture found')
						
						elif final_gest_num == 1:
							correct += 1
							temp_index = np.array(gest_st_en) + original_index[st]
							good_list.append([letter, temp_index[0][0], temp_index[0][1]])
							if check_plot:
								original_dataclip = (np.array(final_gest_st_en) + original_index[st]).tolist()
								for i in range(len(original_dataclip)):
									plt.scatter(original_dataclip[i], [-80+i*10, -80+i*10])
								print(len(fail_list)+len(more_list)+correct, letter, 'good gesture found')
						

						if check_plot:
							# plt.plot(data*50, [i+original_index[st] for i in range(len(data))])
							line1.set_xdata([i+original_index[st] for i in range(len(data))])
							line2.set_xdata([i+original_index[st] for i in range(len(data))])
							line3.set_xdata([i+original_index[st] for i in range(len(data))])

							line1.set_ydata(original_data[original_index[st][0]:original_index[st][0]+len(data),0:1])
							line2.set_ydata(original_data[original_index[st][0]:original_index[st][0]+len(data),1:2])
							line3.set_ydata(original_data[original_index[st][0]:original_index[st][0]+len(data),2:3])
							
							plt.xlim([original_index[st], original_index[st]+len(data)])
							plt.ylim([ymin,ymax])
							
							plt.draw()
							plt.pause(1e-17)
							input('next')
					print(name+'   \t', 'correct number',correct, '- the one that fail: ',len(fail_list), '- the one that more: ',len(more_list))
					pd.DataFrame(good_list).to_csv('../../data/DataLabels_Study2/gest_pos_index'+'_'+name+'.csv', index=False)

























