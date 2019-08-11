import serial
import random as rd
from pynput import keyboard
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


def on_press(key):
    global eventkey
    global loop
    global count
    global index_start
    global index_end
    global read_data
    global gesture_list
    global data_info
    global file_index
    global new_gesture
    global start_session
    global end_session
    global start_key
    global end_key
    global delete_key
    eventkey = '{0}'.format(key)
    # print(eventkey)
    if eventkey == 'Key.esc':
        print(gesture_list, len(gesture_list))
        loop = False
    elif eventkey == delete_key:
        if read_data == False and len(data_info) > 0:
            data_info.pop(-1)
            gesture_list.append(new_gesture)
            print("Data has been deleted; Please restart the session!")

        elif read_data == True:
            read_data = False
            gesture_list.append(new_gesture)
            print("Data has been deleted; Please restart the session!")
    elif eventkey == start_key and start_session == False and read_data == False:
        new_gesture = gesture_list.pop(-1)
        start_session = True
        
    elif eventkey == end_key and end_session == False and read_data == True:
        end_session = True


def on_release(key):
    pass



if __name__ == '__main__':
    ser = serial.Serial('/dev/cu.usbmodem14201', 115200) 
    participant = input('Name of the participant: ')
    file_data = open("../../data/Data_Study2/data_"+participant+".csv", "w+")
    file_data.write('index,letter,ax,ay,az,gx,gy,gz,mx,my,mz\n')
    file_index = open("../../data/Data_Study2/data_index_"+participant+".csv", "w+")
    # plt.ion()

    # Other variables
    count = 0                               #record the data index
    data = []
    read_data = False
    start_session = False
    end_session = False
    index_start = 0
    index_end = 0
    data_info = []
    new_gesture = ''
    loop = True
    start_key = 'Key.alt'
    end_key = 'Key.ctrl'
    delete_key = 'Key.backspace'
    gesture = ['A', 'B', 'C', 'D', 'E', 'F', 'G',\
                'H', 'I', 'J', 'K', 'L', 'M', 'N',\
                'O', 'P', 'Q', 'R', 'S', 'T',\
                'U', 'V', 'W', 'X', 'Y', 'Z', 'delete', 'space']

    gesture_list = []
    for i in range(10):
        gesture_list += gesture
    rd.shuffle(gesture_list)


    gesture_dic = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16,
        'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23,
        'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29,
        'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35,
    }

    keyboard.Listener(on_press=on_press, on_release=on_release).start()

    read = []
    print('Ready to collect the data:')

    while loop:
        try:
            read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
        except:
            print("Read data error")
            continue

        if start_session:
            index_start = count
            for i in range(300):
                try:
                    read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
                    file_data.write(str(count) + ',' + new_gesture + ',' + str(read)[1:-1] + '\n')
                    count = count + 1
                except:
                    print("Read data error")
                    continue
            # img = mpimg.imread('../Study/study2_figure/'+ new_gesture +'.png')
            # plt.imshow(img)
            print('------------------------')
            print('Left:', len(gesture_list), '---','Next: ' + new_gesture)
            print('.........reading........')
            start_session = False
            read_data = True
        
        if end_session:
            for i in range(300):
                try:
                    read = list(map(float, ser.readline().decode('utf-8').split(",")))[0:9]
                    file_data.write(str(count) + ',' + new_gesture + ',' + str(read)[1:-1] + '\n')
                    count = count + 1
                except:
                    print("Read data error")
                    continue

            index_end = count
            data_info.append(new_gesture+','+str(index_start)+','+str(index_end)+'\n')
            end_session = False
            read_data = False
            print('...........stop.........')

            if len(gesture_list) == 0:
                loop = False
        
        if read_data:
            file_data.write(str(count) + ',' + new_gesture + ',' + str(read)[1:-1] + '\n')
            count = count + 1


    for item in data_info:
        file_index.write(item)

    file_data.close()
    file_index.close()








