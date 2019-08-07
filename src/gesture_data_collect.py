import serial
import time
# import keyboard as kb
from pynput import keyboard

ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 

# Files to record the data
participant = input('Name of the participant: ')
mode = input('Study mode? air / touch? ')
file_data = open("Data/data_"+participant+"_"+mode+".csv", "w+")
file_data.write('index,letter,ax,ay,az,gx,gy,gz,mx,my,mz\n')
file_index = open("Data/data_index_"+participant+"_"+mode+".csv", "w+")

# Variables to time the time
time_st = 0
time_en = 0

# Other variables
count = 0                               #record the data index
data = []
read_data = False
index_start = 0
index_end = 0
data_info = []
eventkey = 'none'
gesture_key = ''
loop = True

end = 'Key.alt'

gesture_dic = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8,
    '9':9,
    'a':10,
    'b':11,
    'c':12,
    'd':13,
    'e':14,
    'f':15,
    'g':16,
    'h':17,
    'i':18,
    'j':19,
    'k':20,
    'l':21,
    'm':22,
    'n':23,
    'o':24,
    'p':25,
    'q':26,
    'r':27,
    's':28,
    't':29,
    'u':30,
    'v':31,
    'w':32,
    'x':33,
    'y':34,
    'z':35,
}

def on_press(key):
    global eventkey
    global loop
    global count
    global index_start
    global index_end
    global read_data
    global gesture_key
    global data_info
    global file_index
    global file_data
    eventkey = '{0}'.format(key)
    # print(eventkey)
    # print(eventkey[0], eventkey[1], eventkey[2])
    if eventkey == 'Key.esc':
        for item in data_info:
            file_index.write(item)
        loop = False

    elif eventkey == 'Key.ctrl':
        if read_data == False and len(data_info) > 0:
            data_info.pop(-1)
            print("Data has been deleted; Please restart the session!")

        elif read_data == True:
            read_data = False
            index_end = count - 1
            print("Data has been deleted; Please restart the session!")

    elif eventkey == 'Key.alt' and read_data == True:
        print('stop reading data!')
        read_data = False
        index_end = count - 1
        data_info.append(gesture_key+','+str(index_start)+','+str(index_end)+'\n')

    elif len(eventkey) == 3 and eventkey[1] in gesture_dic.keys() and read_data == False:
        print('--------------------------------')
        print('start to read data!')
        gesture_key = eventkey[1]
        index_start = count
        read_data = True




def on_release(key):
    pass

keyboard.Listener(on_press = on_press, on_release = on_release).start()

read = []
print('Ready to collect the data:')
while loop:
    try:
        read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
    except:
        print("Read data error")
        continue
    
    if read_data:
        file_data.write(str(count) + ',' + gesture_key + ',' + str(read)[1:-1] + '\n')
        count = count + 1

file_data.close()
file_index.close()
