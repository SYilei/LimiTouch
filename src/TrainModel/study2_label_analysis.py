import sys
import os
import numpy as np
import pandas as pd


files = os.listdir('../../data/DataLabels_Study2/')

# name = '_' + sys.argv[1]

def get_next(label_list,letter_list):
	label = []
	letter = []
	while len(letter_list) > 0:
		temp_label = [label_list.pop(0)]
		temp_letter = [letter_list.pop(0)]
		print(letter_list[0], temp_letter[0], letter_list[0] == temp_letter[0])
		count = 0 
		while letter_list[0] == temp_letter[0]:
			print(count)
			count = count + 1
			temp_label.append(label_list.pop(0))
			temp_letter.append(letter_list.pop(0))
		label.append(temp_label)
		label.append(temp_letter)
	return label, letter

# def count_gestures(label_list,letter_list):
for file in files:
	data_pd = pd.read_csv('../../data/DataLabels_Study2/' + file)
	data = data_pd['1'].values.tolist()
	letter = data_pd['0'].values.tolist()
	(label, letter) = get_next(data, letter)
	print(pd.DataFrame(label), pd.DataFrame(letter))
exit(0)
for file in files:
	data_pd = pd.read_csv('../../data/DataLabels_Study2/' + file)
	data = data_pd['1'].values.tolist()
	letter = data_pd['0'].values.tolist()

	zero = 0
	zero_one = 0
	zero_zero = 0
	one = 0
	count = 0
	zero_flag = False
	count_flag = False
	for i in range(len(data)):
		if data[i] == 1:
			one += 1
			count_flag = True
			if zero_flag:
				zero_one += 1
		
		if count_flag:
			if data[i] == 0:
				zero += 1
				zero_zero += 1
				zero_flag = True

			if zero_flag:
				if zero_zero > 100:
					if one > 800:
						one = 0
						zero = 0
						zero_one = 0
						zero_zero = 0
						zero_flag = False
						count_flag = False
						count += 1
					else:
						one = 0
						zero = 0
						zero_one = 0
						zero_zero = 0
						zero_flag = False
						count_flag = False

				if zero_one > zero_zero:
					zero = 0
					zero_one = 0
					zero_zero = 0
					zero_flag = False


	print(count, file)

