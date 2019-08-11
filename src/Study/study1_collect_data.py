import serial
import csv
import random
import time
import sys
from pynput import keyboard


def write_to_csv(file_path, file_list):
    csv_file = open(file_path, 'w+')
    csv_file.write('ax,ay,az,gx,gy,gz,mx,my,mz\n')
    for item in file_list:
        csv_file.write(str(item)[1:-1] + '\n')
    csv_file.close()


def on_press(key):
    global loop
    global read_data
    global gestures

    eventkey = '{0}'.format(key)
    # print(eventkey)
    # print(eventkey[0], eventkey[1], eventkey[2])
    if eventkey == 'Key.esc':
        record = open('../../data/Data_Study1/record_'+sys.argv[1]+'.txt','w+')
        for item in gestures:
            record.write(item+',')
        record.close()
        loop = False

    elif eventkey == 'Key.ctrl':
        if read_data == False:
            print('.........reading data........')
            read_data = True


def on_release(key):
    pass


def generate_gesture_order():
    gesture_order = []
    list_touch = ['touch', 'nontouch']
    while len(list_touch) > 0:
        touch = list_touch.pop(random.randint(0, len(list_touch) - 1))
        list_condition = ['free', 'elbow']
        while len(list_condition) > 0:
            condition = list_condition.pop(random.randint(0, len(list_condition) - 1))
            list_orientation = ['horizontal', 'vertical', 'slash', 'backslash', 'circle', 'static']
            while len(list_orientation) > 0:
                orientation = list_orientation.pop(random.randint(0, len(list_orientation) - 1))
                gesture_order.append((touch+'_'+condition+'_'+orientation))
    final_gesture_list = []
    while len(gesture_order) > 0:
        final_gesture_list.append(gesture_order.pop(random.randint(0, len(gesture_order) - 1)))
    return final_gesture_list


#############Start the code####################
ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
id_user = sys.argv[1]

gestures = []
if len(sys.argv) == 2:
    gestures = generate_gesture_order()
elif len(sys.argv) > 2:
    file = open('../../data/Data_Study1/record_' + sys.argv[1]+'.txt', 'r')
    line = file.readline()
    gestures = line.split(',')[:-1]

for i in range(len(gestures)):
    print(gestures[i])
print('Number of gestures: ', len(gestures))

count = 0
loop = True
read_data = False
data_list = []

keyboard.Listener(on_press = on_press, on_release = on_release).start()

print('')
print('-----------------------------')
print('Gesture'+ str(25 - len(gestures)) +': ' + gestures[0])
while loop:
    data = []
    try:
        data = list(map(float, ser.readline().decode('utf-8').strip().split(',')))
    except:
        print('Read data error!')
        continue

    if read_data:
        data_list.append(data)
        count = count + 1
    
    if count == 30000:
        write_to_csv('../../data/Data_Study1/'+id_user + '_' + gestures[0] + '.csv', data_list)
        read_data = False
        data_list.clear()
        count = 0
        gestures.pop(0)
        if len(gestures) > 0:
            print('')
            print('-----------------------------')
            print('Gesture'+ str(25 - len(gestures))+': ' + gestures[0])
        else:
            break

print('Study finished!!!!!!')