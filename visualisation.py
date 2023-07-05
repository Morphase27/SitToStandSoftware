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




def Main(filename):

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


def processed(data):
    
    #Find number of columns
    i = 0
    for col in data:
        if col[0] == "Mode":
            break
        i = i + 1

    #Add headers
    dataPost = []
    for col in range(i+1): 
        dataPost.append([data[col][0]])
    dataPost.append(['Step'])
    dataPost.append(['Changes'])
    
    #Add data
    n=0
    for row in range(1,len(data[i])):

        #Add data corresponding to sit or stand
        if int(data[i][row]) == 0 or int(data[i][row]) == 6:
            for col in range(i+1): 
                if col == i:
                    dataPost[col].append(int(data[col][row]))
                else:
                    dataPost[col].append(data[col][row])
            dataPost[i+1].append(row)  #Add step number    

            if (row > 1) and (data[i][row] != data[i][row-1]):  #If there is a change in position
                dataPost[i+2].append(1)                         #Mark change as 1
                
                if row - dataPost[i+1][n] >100:                 #If the last standing position was more than 100 steps ago,
                    dataPost[i+2][n] = 1                        #It means it that it the start of standing to sitting
            else:
                dataPost[i+2].append(0)                         #Add 0 if there is no change
            n = n+1

    return dataPost
     

def MaxPeakToPeak(signal):
    """
    Finds when the magnitude between the positive and negative peaks in a noisy oscillating signal
    goes below a certain threshold. Returns the index of the data point where the threshold is crossed.
    """
    # Find the indices of the positive and negative peaks in the signal
    pos_peak_indices, _ = find_peaks(np.array(signal))
    neg_peak_indices, _ = find_peaks(-np.array(signal))
    
    # Calculate the magnitudes between each positive and negative peak
    magnitudes = []
    for pos_index in pos_peak_indices:
        for neg_index in neg_peak_indices:
            if neg_index > pos_index:
                magnitudes.append(signal[pos_index] - signal[neg_index])
                break
    
    max_mag = 0
    # Iterate through the magnitudes and find the index where the threshold is crossed
    for i, magnitude in enumerate(magnitudes):
        if magnitude > max_mag:
            max_mag = magnitude
            max_mag_index = i
    
    #return index of max peak to peak
    return pos_peak_indices[max_mag_index]


def MedianPeakToPeak(signal):
    """
    Finds when the magnitude between the positive and negative peaks in a noisy oscillating signal
    goes below a certain threshold. Returns the index of the data point where the threshold is crossed.
    """
    # Find the indices of the positive and negative peaks in the signal
    pos_peak_indices, _ = find_peaks(np.array(signal))
    neg_peak_indices, _ = find_peaks(-np.array(signal))
    
    # Calculate the magnitudes between each positive and negative peak
    magnitudes = []
    for pos_index in pos_peak_indices:
        for neg_index in neg_peak_indices:
            if neg_index > pos_index:
                magnitudes.append(signal[pos_index] - signal[neg_index])
                break
    
    """
    sum = 0
    # Sum of the magnitude
    for i, magnitude in enumerate(magnitudes):
        sum = sum + magnitude
    
    avg = sum / i
    """
    #find median of peak to peak magnitude
    median = np.median(magnitudes)

    #return average peak to peak magnitude
    return median
    



def PeakToPeakTreshold(signal, threshold):
    """
    Finds when the magnitude between the positive and negative peaks in a noisy oscillating signal
    goes below a certain threshold. Returns the index of the data point where the threshold is crossed.
    """
    # Find the indices of the positive and negative peaks in the signal
    pos_peak_indices, _ = find_peaks(np.array(signal))
    neg_peak_indices, _ = find_peaks(-np.array(signal))
    
    # Calculate the magnitudes between each positive and negative peak
    magnitudes = []
    for pos_index in pos_peak_indices:
        for neg_index in neg_peak_indices:
            if neg_index > pos_index:
                magnitudes.append(signal[pos_index] - signal[neg_index])
                break
    
    # Iterate through the magnitudes and find the index where the threshold is crossed
    for i, magnitude in enumerate(magnitudes):
        """
        sum = 0
        for n in range(1,5):
            if i+ n > len(magnitudes)-1:
                break
            sum = sum + magnitudes[i+n]
        """
        median = np.median(magnitudes[i:i+10])
        if median < threshold*20:
            return pos_peak_indices[i]
    
    # If the threshold is never crossed, return None
    return None

def EndMaxPeak(signal, threshold):
    """
    Finds where the signal crosses a certain threshold that is the median valuenof the sitting phase.
    Returns the index of the data point where the threshold is crossed.
    """
    # Find the indices of the positive and negative peaks in the signal
    pos_peak_indices, _ = find_peaks(np.array(signal))

    peak = 0
    # Iterate through the peaks and find the maximum peak
    for pos_index in pos_peak_indices:
        if signal[pos_index] > peak:
            peak = signal[pos_index]
            peak_index = pos_index
    
    
    # Iterate through the signal after the peak and find the index where the threshold is crossed
    for i, value in enumerate(signal[peak_index:]):
        if value < threshold:
            return i + peak_index
    
    # If the threshold is never crossed, return None
    return None


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



def EndTransition(signal):

    peak_index = np.argmax(signal)
    range = abs(signal[0] - signal[peak_index])

    for i, value in enumerate(signal):
        if value > (signal[0] + range*0.90):
            return i

    return 0


def EndTransition2(signal_R, signal_L):

    peak_indices_R, _ = find_peaks(np.array(signal_R))
    peak_indices_L, _ = find_peaks(np.array(signal_L))

    check_R =0
    for _,idx in enumerate(peak_indices_R):
        if signal_R[idx] > -50:
            peak_R = idx
            check_R = 1
            break
    
    if check_R ==0:
        peak_R = np.argmax(signal_R)
    
    check_L =0
    for _,idx in enumerate(peak_indices_L):
        if signal_L[idx] > -50:
            peak_L = idx
            check_L = 1
            break
    if check_L ==0:
        peak_L = np.argmax(signal_L)

    
    idx_end_trans = min(peak_R, peak_L)

    if idx_end_trans == peak_R:
        test = np.where(peak_indices_R ==peak_R)[0][0]
        idx_end_standing = peak_indices_R[test+1]
    else:
        test = np.where(peak_indices_L ==peak_L)[0][0]
        idx_end_standing = peak_indices_L[test+1]


    return idx_end_trans, idx_end_standing





def StartSitting(signal):

    min_index = np.argmin(signal)
    median = np.median(signal[min_index:])

    max_index = np.argmax(signal)

    test = signal[max_index]
    test2 = signal[min_index]
    range = abs(signal[max_index] - median)

    for i, value in enumerate(signal[max_index:]):
        if value < (signal[max_index] - range*0.95):
            return i + max_index

    return None


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def HipAngle(acc_WZ, acc_WY, gyro_WX,acc_TZ,acc_TX, gyro_TY, init_end):

    dt = 1/500
    init_0 = 250
    new_0 = 245

    gyro_WX = np.array(gyro_WX) * 2 * np.pi / 360
    gyro_TY = np.array(gyro_TY) * 2 * np.pi / 360
 

    max_gyro_WX = np.argmax(gyro_WX[:init_end])

    """     max_TY = abs(np.max(gyro_TY[:init_end]- gyro_TY[0]))
    min_TY = abs(np.min(gyro_TY[:init_end]- gyro_TY[0]))
    if max_TY < min_TY:
        gyro_TY = -gyro_TY
        print("chocolat") """

    integral = np.sum(gyro_TY[:init_end]) - gyro_TY[0]*init_end
    if integral < 0:
        gyro_TY = -gyro_TY
        print("chocolat")
    
    
    static_W= []
    static_T = []

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
            
            """
            if variance_W < 0.1:
                gyro_W0 = np.sum(gyro_WX[i-new_0:i])/new_0
                #if i> 4000:
                    #print("W ", i)
                    #print("gyro_W0 ", gyro_W0)
                static_W.append(i)
            if variance_T < 0.1:
                gyro_T0 = np.sum(gyro_TY[i-new_0:i])/new_0
                #if i> 4000:
                    #print("T ", i)
                    #print("gyro_T0 ", gyro_T0)
                static_T.append(i)
            """
        
            tilt_W[i] = tilt_W[i-1] + (gyro_WX[i] - gyro_W0 ) * dt
            tilt_T[i] = tilt_T[i-1] + (gyro_TY[i] - gyro_T0) * dt
            angle_hip[i] = tilt_W[i] - tilt_T[i]


        angle_hip[i] = angle_hip[i] * 360 / (2 * np.pi)

    tilt_W = tilt_W * 360 / (2 * np.pi)

    sin_angle_hip = np.sin(angle_hip*2*math.pi/360)
    

    """
    # Define the wavelet type and level of decomposition
    wavelet = 'db1'
    level = 2

    # Perform the DWT
    coeffs = pywt.wavedec(sin_angle_hip, wavelet, level=level)

    # Filter the signal by thresholding the detail coefficients
    threshold = 2  # Set the threshold value
    new_coeffs = [coeffs[0]] + [pywt.threshold(detail_coeff, threshold) for detail_coeff in coeffs[1:]]

    # Reconstruct the signal
    filtered_signal = pywt.waverec(new_coeffs, wavelet)
    """

    return angle_hip, tilt_W  #filtered_signal[1:]    #sin_angle_hip     # filtered_signal[1:]


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
    
    for i in range(end_trans, end_trans+ 5000):
        diff_end = abs(signal_R[i] - signal_L[i])
        if diff_end > diff_start*2:
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
        treshold = 0.05
    else:
        treshold = 0.01

    check_tilt = 0
    check_R = 0
    check_L = 0    

    for i in range(max_idx + 100, len(signal_R)-1):
        
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
    treshold_grad = 0.05
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

    for i in range(max_hip_tilt - init_start):
        
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
col_plot_data = 1
sensor = "accelerometer"  #accelerometer, gyroscope, IMU, EMG, all
SiSt = 1
StSi = 0
multi = 0


thisFolderParent = os.getcwd()

candidate = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]
c = 5

# Initialise the lists
Sit = []
Stand = []
all_windows = []

# Loop through all the circuits of each candidate
for i in range(35,36): 
    print("Circuit " + str(i))

    hardrive_path = "/Volumes/UnionSine/IMEE 2022/Semester2/FYP/FYPcode"

    if i<10:
        filename  = hardrive_path + "/Source/AB" + candidate[c] + "/Processed/AB" + candidate[c] + "_Circuit_00" + str(i) + "_post"
    else:
        filename  =  hardrive_path + "/Source/AB" + candidate[c] + "/Processed/AB" + candidate[c] + "_Circuit_0" + str(i) + "_post"
    
    #filename  = thisFolderParent + "/Data/AB156/Processed/AB156_Circuit_002_post"

    allData = Main(filename)

    STprocessed = processed(allData)

    nbCol = len(STprocessed)
    nbRow = len(STprocessed[nbCol-3])
    
    indexChange = [STprocessed[nbCol-2][idx] for idx, value in enumerate(STprocessed[nbCol-1]) if value == 1]
    #for i in range(len(indexChange)): print(STprocessed[nbCol - 2][indexChange[i]])



    # separate the data into Sit to Stand and Stand to Sit
    SiSt_data = []
    StSi_data = []    
    for col in range(nbCol-2):
        #Add header
        SiSt_data.append([STprocessed[col][0]])
        StSi_data.append([STprocessed[col][0]])
       
        #Add data
       
        # Normalise every measurement to start at 0 (Not good results)
        # if STprocessed[col][1] > 0:
        #     for row in range(1,nbRow):
        #         if row < indexChange[1]:
        #             SiSt_data[col].append(STprocessed[col][row] - STprocessed[col][1])
        #         else:
        #             StSi_data[col].append(STprocessed[col][row] - STprocessed[col][1])
        # else:
        #     for row in range(1,nbRow):
        #         if row < indexChange[1]:
        #             SiSt_data[col].append(STprocessed[col][row]+ STprocessed[col][1])
        #         else:
        #             StSi_data[col].append(STprocessed[col][row] + STprocessed[col][1])

        for row in range(1,nbRow):
            if row < indexChange[1]:
                SiSt_data[col].append(STprocessed[col][row])
            else:
                StSi_data[col].append(STprocessed[col][row])


fs = 500  # Sampling frequency
cutoff = 50  # Desired cutoff frequency in Hz
order = 6  # Filter order

filtered_SiSt = butter_lowpass_filter(SiSt_data[col_end_trans][1:], cutoff, fs, order)



median_pk2pk = MedianPeakToPeak(filtered_SiSt[0:indexChange[0]])
max_pk2pk_index = MaxPeakToPeak(filtered_SiSt)
Threshold_Index = PeakToPeakTreshold(filtered_SiSt[max_pk2pk_index:], median_pk2pk) + max_pk2pk_index
print("EMG index: " + str(Threshold_Index))


median_pk2pk_gyro = MedianPeakToPeak(SiSt_data[col_end_trans_gyro][1:])
Threshold_Index_gyro = EndMaxPeak(SiSt_data[col_end_trans_gyro][1:], median_pk2pk_gyro)
print("Gyro index: " + str(Threshold_Index_gyro))

idx_start_trans_SiSt = StartTransition(SiSt_data[45][1:indexChange[1]], SiSt_data[47][1:indexChange[1]], indexChange[0], indexChange[1])


index_end_trans_SiSt = EndTransition(SiSt_data[45][indexChange[0]:]) + indexChange[0]
index_end_trans_SiSt2, idx_end_standing = EndTransition2(allData[45][1:], allData[47][1:])

print("gonio index: " +str(index_end_trans_SiSt))

index_end_trans_StSi = StartSitting(StSi_data[45][indexChange[3]-indexChange[2]:]) + indexChange[3]

#print(indexChange[3])


hip_angle_R, tilt_hip = HipAngle(allData[26][1:], allData[25][1:], allData[29][1:], allData[8][1:], allData[6][1:], allData[9][1:], indexChange[1]) #Right allData[8][1:], allData[6][1:], allData[9]
hip_angle_L, _ = HipAngle(allData[26][1:], allData[25][1:], allData[29][1:], allData[20][1:], allData[18][1:], allData[21][1:], indexChange[1])
# Left allData[20][1:], allData[18][1:], allData[21][1:]

"""
SiSt_sit_end, SiSt_lean_end, SiSt_stand_start, SiSt_stand_end = SiStTransHipAngle(hip_angle_R[1:],hip_angle_L[1:], indexChange[0], indexChange[1], idx_start_trans_SiSt, tilt_hip)

StSi_stand_end, StSi_lean_end, StSi_sit_start  = StSiTransHipAngle(hip_angle_R[1:],hip_angle_L[1:], indexChange[2], indexChange[3], index_end_trans_StSi, tilt_hip)
"""

# Find the indices of the positive and negative peaks in the signal
#pos_peak_indices, _ = find_peaks(np.array(filtered_SiSt))
#neg_peak_indices, _ = find_peaks(-np.array(filtered_SiSt))


#----------------------Plot the data----------------------#


SiStData = allData[col_plot_data][1:len(allData[col_plot_data])-1]

"""
SiStData = allData[35][1:len(allData[col_plot_data])-1]
SiStData2 = allData[42][1:len(allData[col_plot_data])-1]

SiStData3 = allData[45][1:len(allData[col_plot_data])-1]
SiStData4 = allData[47][1:len(allData[col_plot_data])-1]

for i in range(len(SiStData3)):
    test = SiStData3[i]
    SiStData3[i] = SiStData3[i]/100
    SiStData4[i] = SiStData4[i]/100
"""

#SiStData = filtered_SiSt


if multi == 1:
    
    SiStDataY = allData[col_plot_data+1][1:len(allData[col_plot_data])-1]
    SiStDataZ = allData[col_plot_data+2][1:len(allData[col_plot_data])-1]

freq = 500
SiStTime = [i/freq for i in range(len(SiStData))]




# create a new figure
fig, ax = plt.subplots()

ax.plot(SiStTime, SiStData)




""" # plot the data
ax.plot(SiStTime, SiStData, label= allData[35][0])
ax.plot(SiStTime, SiStData2, label= allData[42][0])

ax.plot(SiStTime, SiStData3, label= allData[45][0])
ax.plot(SiStTime, SiStData4, label= allData[47][0]) """

"""
ax.plot(SiStTime, hip_angle_R, label= "hip angle right")
ax.plot(SiStTime, hip_angle_L, label= "hip angle left")
ax.plot(SiStTime, tilt_hip, label= "hip tilt")
"""

#ax.plot(SiStTime, allData[29][1:len(allData[col_plot_data])-1], label= "gyroWX")
#ax.plot(SiStTime, allData[21][1:len(allData[col_plot_data])-1], label= "gyro Thigh Left") 
#ax.plot(SiStTime, allData[9][1:len(allData[col_plot_data])-1], label= "gyro Thigh Right") 


"""
# Add a cross at each peak index
for index in staticW:
    plt.plot(SiStTime[index], hip_angle_R[index], 'rx')
for index in staticT:
    plt.plot(SiStTime[index], hip_angle_R[index], 'ro')
"""


"""
# Add a cross at each peak index
for index in peak_indices_R:
    plt.plot(SiStTime[index], hip_angle_R[index], 'rx')
for index in peak_indices_L:
    plt.plot(SiStTime[index], hip_angle_L[index], 'ro')
"""
    
if multi == 1:
    ax.plot(SiStTime, SiStDataY, label= allData[col_plot_data+1][0])
    ax.plot(SiStTime, SiStDataZ, label= allData[col_plot_data+2][0])

#Plot change of state with a vertical dash line
#plt.axvline(x = indexChange[0]/freq, color = 'b', linestyle = 'dashed', label = 'start standing')
#plt.axvline(x = indexChange[1]/freq, color = 'b', linestyle = 'dashed', label = 'start walking')
#plt.axvline(x = indexChange[2]/freq, color = 'b', linestyle = 'dashed', label = 'end walking')
#plt.axvline(x = indexChange[3]/freq, color = 'b', linestyle = 'dashed', label = 'start sitting')
#plt.axvline(x = SiStTime[idx_start_trans_SiSt], color = 'r', label = 'new start transition')
#plt.axvline(x = SiStTime[idx_end_standing], color = 'r', linestyle = 'dashed', label = 'new end standing')


plt.axvline(x = indexChange[0]/freq, color = 'b', linestyle = 'dashed')
plt.axvline(x = indexChange[1]/freq, color = 'b', linestyle = 'dashed')
plt.axvline(x = indexChange[2]/freq, color = 'b', linestyle = 'dashed')
plt.axvline(x = indexChange[3]/freq, color = 'b', linestyle = 'dashed')


"""
plt.axvline(x = SiStTime[SiSt_sit_end], color = 'y', linestyle = 'dashed', label = 'SiSt start transition hip')
plt.axvline(x = SiStTime[SiSt_lean_end], color = 'r', linestyle = 'dashed', label = 'SiSt end leaning forward')
plt.axvline(x = SiStTime[SiSt_stand_start], color = 'g', linestyle = 'dashed', label = ' SiSt end transiton')
plt.axvline(x = SiStTime[SiSt_stand_end], color = 'b', label = ' SiSt end standing')
"""

#plt.axvline(x = SiStTime[StSi_stand_end], color = 'y', linestyle = 'dashed', label = 'StSi end standing')
#plt.axvline(x = SiStTime[StSi_lean_end], color = 'r', linestyle = 'dashed', label = 'StSi end leaning forward')
#plt.axvline(x = SiStTime[StSi_sit_start], color = 'g', linestyle = 'dashed', label = 'StSi start sitting')

#plt.axvline(x = SiStTime[idx_start_trans_SiSt], color = 'r', label = 'knee band')
idx_start_trans_SiSt
#plt.axvline(x = 2500/500, color = 'g', linestyle = 'dashed', label = 'test')


#plt.axvline(x = SiStTime[max_pk2pk_index], color = 'r', linestyle = 'dashed', label = 'max peak to peak')
#plt.axvline(x = SiStTime[Threshold_Index], color = 'g', linestyle = 'dashed', label = 'end transition')
#plt.axvline(x = SiStTime[Threshold_Index_gyro], color = 'r', linestyle = 'dashed', label = 'end transition gyro')
#plt.axvline(x = SiStTime[index_end_trans_SiSt], color = 'y', linestyle = 'dashed', label = 'end transition gonio')
#plt.axvline(x = SiStTime[index_end_trans_SiSt2], color = 'r', linestyle = 'dashed', label = 'end transition gonio')

#plt.axvline(x = SiStTime[index_end_trans_StSi], color = 'g', linestyle = 'dashed', label = 'end transition gonio')


handles, labels = ax.get_legend_handles_labels()

# reverse the order
ax.legend(handles[::-1], labels[::-1], loc='upper right')

# set the x and y axis labels
ax.set_xlabel('time (s)')

# set the title of the plot
ax.set_title(allData[col_plot_data][0])

# display the plot
plt.show()


#plt.plot(time1, plot1)
#plt.yticks(labels)
#plt.show

print("done")









"""
col = 0

SiStData = STprocessed[col][1:indexChange[1]]
StSiData = STprocessed[col][indexChange[2]:nbRow -1]

if multi == 1:
    
    SiStDataY = STprocessed[col+1][1:indexChange[1]]
    SiStDataZ = STprocessed[col+2][1:indexChange[1]]
    StSiDataY = STprocessed[col+1][indexChange[2]:nbRow -1]
    StSiDataZ = STprocessed[col+2][indexChange[2]:nbRow -1]

SiStSteps = np.array(STprocessed[nbCol- 2][1:indexChange[1]])
StSiSteps = np.array(STprocessed[nbCol-2][indexChange[2]:nbRow -1])


#Create a time axis
freq = 500
SiStTime = SiStSteps/freq
StSiTime = StSiSteps/freq

# create a new figure
fig, ax = plt.subplots()

if SiSt == 1:
    # plot the data
    ax.plot(SiStTime, SiStData, label= STprocessed[col][0])
    if multi == 1:
        ax.plot(SiStTime, SiStDataY, label= STprocessed[col+1][0])
        ax.plot(SiStTime, SiStDataZ, label= STprocessed[col+2][0])

    #Plot change of state with a vertical dash line
    plt.axvline(x = STprocessed[nbCol-2][indexChange[0]]/freq, color = 'b', linestyle = 'dashed', label = 'change of position')
    plt.axvline(x = STprocessed[nbCol-2][indexChange[1]]/freq, color = 'b', linestyle = 'dashed', label = 'start walking')
else:
    # plot the data
    ax.plot(StSiTime, StSiData, label= STprocessed[col][0])
    if multi == 1:
        ax.plot(StSiTime, StSiDataY,label= STprocessed[col+1][0])
        ax.plot(StSiTime, StSiDataZ, label= STprocessed[col+2][0])

    #Plot change of state with a vertical dash line
    plt.axvline(x = STprocessed[nbCol-2][indexChange[3]]/freq, color = 'b', linestyle = 'dashed', label = 'change of position')
    plt.axvline(x = STprocessed[nbCol-2][indexChange[2]]/freq, color = 'b', linestyle = 'dashed', label = 'start walking')

handles, labels = ax.get_legend_handles_labels()

# reverse the order
ax.legend(handles[::-1], labels[::-1])


# set the x and y axis labels
ax.set_xlabel('time')
ax.set_ylabel('IMU')

# set the title of the plot
ax.set_title('test')

# display the plot
plt.show()


#plt.plot(time1, plot1)
#plt.yticks(labels)
#plt.show

print("done")


"""


