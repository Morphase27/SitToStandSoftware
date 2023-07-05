import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

from scipy.signal import butter, lfilter, find_peaks

import pywt




def readData(filename):

    data = []

    #Open .csv file for read        
    with open(filename +'.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Create objects for each header
        # Assign the name of each header to the objects
        for row in reader:
            for i in range(len(row)): data.append([row[i]])
            break
        
        # Assign the value of each column to a different object
        for row in reader:
            for i in range(len(data)):
                if row[i] != "":
                    data[i].append(float(row[i]))
             
    return data


     

def StartTransition(signal_R, signal_L, init_start, init_end):

    median_R = np.median(signal_R[:init_start])
    median_L = np.median(signal_L[:init_start])

    max_R_idx = np.argmax(signal_R[:init_end])
    max_L_idx = np.argmax(signal_L[:init_end])

    range_R = abs(signal_R[max_R_idx] - median_R)
    range_L = abs(signal_L[max_L_idx] - median_L)

    for i in range(len(signal_R[:init_end])-1):
        if signal_R[i] > (median_R + range_R*0.02) and signal_L[i] > (median_L + range_L*0.02):
            return i



def StartSitting(signal, init_start):

    min_index = np.argmin(signal[init_start:]) + init_start
    median = np.median(signal[min_index:])
    max_index = np.argmax(signal[init_start:]) + init_start

    range = abs(signal[max_index] - median)

    for i, value in enumerate(signal[max_index:]):
        if value < (signal[max_index] - range*0.95):
            break

    return i + max_index



def HipAngle(acc_WZ, acc_WY, gyro_WX,acc_TZ,acc_TX, gyro_TY, init_end):

    dt = 1/500
    init_0 = 250
    new_0 = 245

    gyro_WX = np.array(gyro_WX) * 2 * np.pi / 360
    gyro_TY = np.array(gyro_TY) * 2 * np.pi / 360

    
    integral = np.sum(gyro_TY[:init_end]) - gyro_TY[0]*init_end
    if integral < 0:
        gyro_TY = -gyro_TY
        print("inverted gyro thigh")

    gyro_W0 = 0
    gyro_T0 = 0
    tilt_W0 =0
    tilt_T0 = 0
    for i in range(50):
        gyro_W0 = gyro_W0 + gyro_WX[i]/init_0
        gyro_T0 = gyro_T0 + gyro_TY[i]/init_0
        tilt_W0 = tilt_W0 + np.arctan2(acc_WZ[i],acc_WY[i])/init_0
        tilt_T0 = tilt_T0 + np.arctan2(acc_TZ[i],acc_TX[i])/init_0

    tilt_W = np.zeros(len(gyro_WX)-1)
    tilt_T = np.zeros(len(gyro_WX)-1)
    angle_hip = np.zeros(len(gyro_WX)-1)
    for i in range(len(gyro_WX)-1):
        
        if i < init_0:
            tilt_W[i] = tilt_W0
            tilt_T[i] = tilt_T0
            angle_hip[i] = tilt_W0 - tilt_T0
        else:
        
            tiltW_deg = tilt_W[i-new_0:i]*180/np.pi
            tiltT_deg = tilt_T[i-new_0:i]*180/np.pi

            #variance_W = math.sqrt((np.sum((tiltW_deg**2)*dt) - np.sum(tiltW_deg*dt)**2)/new_0)
            #variance_T = math.sqrt((np.sum((tiltT_deg**2)*dt) - np.sum(tiltT_deg*dt)**2)/new_0)
            
            variance_W = np.sum((tiltW_deg - np.sum(tiltW_deg)/new_0)**2)/new_0
            variance_T = np.sum((tiltT_deg - np.sum(tiltT_deg)/new_0)**2)/new_0

            #variance_W = math.sqrt((np.sum((tilt_W[i-new_0:i]**2)*dt) - np.sum(tilt_W[i-new_0:i]*dt)**2)/new_0)
            #variance_T = math.sqrt((np.sum((tilt_T[i-new_0:i]**2)*dt) - np.sum(tilt_T[i-new_0:i]*dt)**2)/new_0)
            
            #variance_W = np.sum((tilt_W[i-new_0:i] - np.sum(tilt_W[i-new_0:i])/new_0)**2)/new_0
            #variance_T = np.sum((tilt_T[i-new_0:i] - np.sum(tilt_T[i-new_0:i])/new_0)**2)/new_0
            

            tilt_W[i] = tilt_W[i-1] + (gyro_WX[i] - gyro_W0 ) * dt
            tilt_T[i] = tilt_T[i-1] + (gyro_TY[i] - gyro_T0) * dt
            angle_hip[i] = tilt_W[i] - tilt_T[i]


        angle_hip[i] = angle_hip[i] * 360 / (2 * np.pi)

    tilt_W = tilt_W * 360 / (2 * np.pi)
    
    return angle_hip, tilt_W  



def SiStTransHipAngle(signal_R, signal_L, init_start, init_end, knee_start, hip_tilt_signal):


    max_R_idx = np.argmax(signal_R[:init_end])
    max_L_idx = np.argmax(signal_L[:init_end])

    # Index where the trunk end leaning forward
    # Do it by finding the max angle forward
    max_idx = max(max_R_idx, max_L_idx)

    #Find the index where the trunk start leaning forward
    # Do it by finding when the hip angle is 5% of the range between the static angle and the max angle forward
 
    delta = 100
    treshold = 0.05

    check_R = 0
    check_L = 0

    for i in range(max_idx):
        
        gradient_R = abs(signal_R[i+delta] - signal_R[i])/(delta)
        gradient_L = abs(signal_L[i+delta] - signal_L[i])/(delta)

        if gradient_R > treshold:
            check_R = 1
        if gradient_L > treshold:
            check_L = 1
        
        if check_R == 1 and check_L == 1:
            break

    start_trans = i


    # Find the index where the trunk stops leaning backward
    # Do it by finding when the slop decrease

    delta = 50
    treshold = 0.01

    check_tilt = 0
    check_R = 0
    check_L = 0

    max_hip_tilt_idx= np.argmax(hip_tilt_signal[:init_end])

    for i in range(max_hip_tilt_idx,init_end*2):

        gradient_tilt = abs(hip_tilt_signal[i+delta] - hip_tilt_signal[i])/(delta)
        gradient_R = abs(signal_R[i+delta] - signal_R[i])/(delta)
        gradient_L = abs(signal_L[i+delta] - signal_L[i])/(delta)
        
        if gradient_tilt < treshold:
            check_tilt = 1
        if gradient_R < treshold:
            check_R = 1
        if gradient_L < treshold:
            check_L = 1
        
        if (check_R == 1 and check_tilt == 1) or (check_L == 1 and check_tilt == 1):
            break

    end_trans = i


    diff_start = abs(signal_R[end_trans] - signal_L[end_trans])
    check_diff = 0
    for i in range(end_trans, end_trans+ 5000):
        diff_end = abs(signal_R[i] - signal_L[i])
        if diff_start < 25:
            if diff_end > 25:
                break
        else:     
            if diff_end > diff_start*1.5:
                break
            if diff_end < diff_start/10 and check_diff == 0:
                check_diff = 1
            if check_diff == 1 and diff_end > diff_start*1.2:
                break 

    end_stand = i


    return start_trans, max_idx, end_trans, end_stand
    


def StSiTransHipAngle(signal_R, signal_L, init_start, init_end, knee_end, hip_tilt_signal):

    peak_indices_R, _ = find_peaks(np.array(signal_R[init_start:]), prominence=5)
    peak_indices_L, _ = find_peaks(np.array(signal_L[init_start:]), prominence=5)
    peak_indices_tilt, _ = find_peaks(np.array(hip_tilt_signal[init_start:]))

    diff = [abs(init_start + value - knee_end) for _, value in enumerate(peak_indices_R)]
    max_R_idx = peak_indices_R[diff.index(min(diff))]
    diff = [abs(init_start + value - knee_end) for _, value in enumerate(peak_indices_L)]
    max_L_idx = peak_indices_L[diff.index(min(diff))]
    diff = [abs(init_start + value - knee_end) for _, value in enumerate(peak_indices_tilt)]
    max_tilt_idx = peak_indices_tilt[diff.index(min(diff))]

    max_idx = max(max_R_idx, max_L_idx) + init_start

    """
    for i in range(len(signal_R[max_idx:])-1):
        if signal_R[i] < (median_R + range_R*0.02) and signal_L[i] < (median_L + range_L*0.02):
            break   
    
    end_trans = i + max_idx
    """

    # Find the index where the trunk stops leaning backward
    # Do it by finding when the slop decrease

    min_R_end = np.argmin(signal_R[max_idx:])
    min_L_end = np.argmin(signal_L[max_idx:])
    min_tilt_end = np.argmin(hip_tilt_signal[max_idx:])

    check = 0
    if min_R_end > len(signal_R[max_idx:]) - 100:
        check = check +1
    if min_L_end > len(signal_R[max_idx:])- 100:
        check = check +1
    if min_tilt_end > len(signal_R[max_idx:])- 100:
        check = check +1

    delta = 150

    if check >= 2:
        treshold = 0.03
    else:
        treshold = 0.01

    check_tilt = 0
    check_R = 0
    check_L = 0    

    for i in range(max_idx + 100, len(signal_R)-1):
        
        if i + delta >= len(signal_R)-2:
            i = len(signal_R)-2
            break

        gradient_tilt = abs(hip_tilt_signal[i+delta] - hip_tilt_signal[i])/(delta)
        gradient_hip_R = abs(signal_R[i+delta] - signal_R[i])/(delta)
        gradient_hip_L = abs(signal_L[i+delta] - signal_L[i])/(delta)

        if gradient_tilt < treshold:
            check_tilt = 1
        if gradient_hip_R < treshold:
            check_R = 1
        if gradient_hip_L < treshold:
            check_L = 1

        if check_tilt == 1 and check_R == 1 and check_L == 1:
            break

    end_trans = i


    # Find the index where the trunk start leaning forward
    # Do it by finding when the hip tilt slop decrease. 
    # Starts from the end of the trunk leaning forward and goes backward
    
    max_hip_tilt = np.argmax(hip_tilt_signal[init_start:]) +init_start - 1
    max_hip_tilt = max_tilt_idx + init_start

    delta = 50
    treshold_grad = 0.008
    treshold_range = 0.3

    check_tilt = 0
    check_R = 0
    check_L = 0

    min_tilt = np.min(hip_tilt_signal[init_start:max_hip_tilt])
    min_R = np.min(signal_R[init_start:max_R_idx+ init_start])
    min_L = np.min(signal_L[init_start:max_L_idx+ init_start])
    
    range_tilt = abs(tilt_hip[max_hip_tilt] - min_tilt)
    range_R = abs(signal_R[max_R_idx+ init_start] - min_R)
    range_L = abs(signal_L[max_L_idx+ init_start] - min_L)

    for i in range(max_hip_tilt - init_start + 1000):
        
        n = max_hip_tilt - i
        gradient_tilt = abs(tilt_hip[n-delta] - tilt_hip[n])/(delta)
        gradient_hip_R = abs(signal_R[n-delta] - signal_R[n])/(delta)
        gradient_hip_L = abs(signal_L[n-delta] - signal_L[n])/(delta)

        if gradient_tilt < treshold_grad and tilt_hip[n] < (min_tilt + range_tilt*treshold_range):
            check_tilt = 1
        if gradient_hip_R < treshold_grad and signal_R[n] < (min_R + range_R*treshold_range):
            check_R = 1
        if gradient_hip_L < treshold_grad and signal_L[n] < (min_L + range_L*treshold_range):
            check_L = 1

        if check_R +check_L + check_tilt >= 2:
            break

    start_trans = n

    return start_trans, max_idx, end_trans






#-------------------Main-------------------#


col_end_trans = 35
col_end_trans_gyro = 29
col_plot_data = 35
sensor = "accelerometer"  #accelerometer, gyroscope, IMU, EMG, all
SiSt = 1
StSi = 0
multi = 0


thisFolderParent = os.getcwd()

candidate = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]
#c = 6

# Initialise the lists
Sit = []
Stand = []
all_windows = []

for c in range(0,len(candidate)):
    print("Candidate " + candidate[c])

    # Loop through all the circuits of each candidate
    for circ in range(1 ,51): 
        print("Circuit " + str(circ))
        if circ<10:
            filename  = thisFolderParent + "/Data/Source/AB" + candidate[c] + "/Processed/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_post"
        else:
            filename  = thisFolderParent + "/Data/Source/AB" + candidate[c] + "/Processed/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_post"
        
        #filename  = thisFolderParent + "/Data/AB156/Processed/AB156_Circuit_002_post"

        if os.path.exists(filename + ".csv"):

            allData = readData(filename)

            for i in range(len(allData)):
                if allData[i][0] == "Mode":
                    m = i 
                    break
            
            indexChange = []
            prev_label = int(allData[m][1])
            for i in range(2,len(allData[m])):
                current_label = int(allData[m][i])
                if current_label != prev_label:
                    indexChange.append(i)
                    if len(indexChange) >= 2:
                        break
                prev_label = current_label

            prev_label = int(allData[m][len(allData[m])-1])
            for i in range(2,len(allData[m])):
                n = len(allData[m])-i
                current_label = int(allData[m][n])
                if current_label != prev_label:
                    indexChange.append(n)
                    if len(indexChange) >= 4:
                        break
                prev_label = current_label

            idx4 = indexChange[3]
            idx3 = indexChange[2]
            indexChange[2] = idx4
            indexChange[3] = idx3



            idx_start_trans_SiSt = StartTransition(allData[45][1:indexChange[1]], allData[47][1:indexChange[1]], indexChange[0], indexChange[1])
            index_end_trans_StSi = StartSitting(allData[45], indexChange[2])


            hip_angle_R, tilt_hip = HipAngle(allData[26][1:], allData[25][1:], allData[29][1:], allData[8][1:], allData[6][1:], allData[9][1:], indexChange[1]) 
            hip_angle_L, _ = HipAngle(allData[26][1:], allData[25][1:], allData[29][1:], allData[20][1:], allData[18][1:], allData[21][1:], indexChange[1])


            SiSt_sit_end, SiSt_lean_end, SiSt_stand_start, SiSt_stand_end = SiStTransHipAngle(hip_angle_R[1:],hip_angle_L[1:], indexChange[0], indexChange[1], idx_start_trans_SiSt, tilt_hip)

            StSi_stand_end, StSi_lean_end, StSi_sit_start  = StSiTransHipAngle(hip_angle_R[1:],hip_angle_L[1:], indexChange[2], indexChange[3], index_end_trans_StSi, tilt_hip)


            freq = 500
            SiStTime = [i/freq for i in range(len(allData[0][1:])-1)]

            """
            # create a new figure
            fig, ax = plt.subplots()
            
            ax.plot(SiStTime, hip_angle_R, label= "hip angle right")
            ax.plot(SiStTime, hip_angle_L, label= "hip angle left")
            ax.plot(SiStTime, tilt_hip, label= "hip tilt")

            plt.axvline(x = SiStTime[SiSt_sit_end], color = 'y', linestyle = 'dashed', label = 'SiSt start transition hip')
            plt.axvline(x = SiStTime[SiSt_lean_end], color = 'r', linestyle = 'dashed', label = 'SiSt end leaning forward')
            plt.axvline(x = SiStTime[SiSt_stand_start], color = 'g', linestyle = 'dashed', label = ' SiSt end transiton')
            plt.axvline(x = indexChange[1]/freq, color = 'b', linestyle = 'dashed', label = 'start walking')
            plt.axvline(x = SiStTime[SiSt_stand_end], color = 'b', label = ' SiSt end standing')

            plt.axvline(x = indexChange[2]/freq, color = 'b', linestyle = 'dashed', label = 'end walking')
            plt.axvline(x = SiStTime[StSi_stand_end], color = 'y', linestyle = 'dashed', label = 'StSi end standing')
            plt.axvline(x = SiStTime[StSi_lean_end], color = 'r', linestyle = 'dashed', label = 'StSi end leaning forward')
            plt.axvline(x = SiStTime[StSi_sit_start], color = 'g', linestyle = 'dashed', label = 'StSi start sitting')

            plt.show()
            """


            #-------------------Write output file-------------------#
            if circ<10:
                out_file_name  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_SiSt"
            else:
                out_file_name  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_SiSt"

            file = open(out_file_name +'.csv', 'w')
            file.truncate()

            for i in range(m):
                file.write(allData[i][0] + ",")
            file.write("right_hip_angle,")
            file.write("left_hip_angle,")
            file.write("hip_tilt,")
            file.write("mode")
            file.write("\n")

            for n in range(1,SiSt_stand_end):
                for i in range(m):
                    file.write(str(allData[i][n]) + ",")
                file.write(str(hip_angle_R[n]) + ",")
                file.write(str(hip_angle_L[n]) + ",")
                file.write(str(tilt_hip[n]) + ",")
                if n < SiSt_sit_end:
                    file.write("1\n")
                elif n < SiSt_lean_end:
                    file.write("121\n")
                elif n < SiSt_stand_start:
                    file.write("122\n")
                else:
                    file.write("2\n")

            file.close()

            if circ<10:
                out_file_name  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_StSi"
            else:
                out_file_name  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_StSi"

            file = open(out_file_name +'.csv', 'w')

            for i in range(m):
                file.write(allData[i][0] + ",")
            file.write("right_hip_angle,")
            file.write("left_hip_angle,")
            file.write("hip_tilt,")
            file.write("mode")
            file.write("\n")

            for n in range(indexChange[2], len(allData[0])-2):
                for i in range(m):
                    file.write(str(allData[i][n]) + ",")
                file.write(str(hip_angle_R[n]) + ",")
                file.write(str(hip_angle_L[n]) + ",")
                file.write(str(tilt_hip[n]) + ",")
                if n > StSi_sit_start:
                    file.write("1\n")
                elif n > StSi_lean_end:
                    file.write("212\n")
                elif n > StSi_stand_end:
                    file.write("211\n")
                else:
                    file.write("2\n")


            file.close()