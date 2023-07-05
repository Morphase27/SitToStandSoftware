import csv
import os
import numpy as np
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



def Main(filename):

    data = []

    #Open .csv file for read        
    with open(filename +'.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Create objects for each header
        # Assign the name of each header to the objects
        for row in reader:
            for i in range(len(row)):
                data.append([row[i]])
                if row[i] == "mode":
                    break
            break
        
        # Assign the value of each column to a different object
        x = 1
        for row in reader:
            for i in range(len(data)):
                if row[i] != "":
                    data[i].append(float(row[i]))
             
    return data


def featureExtraction(data, window_time, freq, mode, sensor, activity):
    
    #Indicate which columns to use for each sensor
    if sensor == "accelerometer":
        cols = [0,2,6,8,12,14,18,20,25,26]
    elif sensor == "gyroscope":
        cols = [3,9,15,21,29]
    elif sensor == "IMU":
        cols = [*[0,2,6,8,12,14,18,20,25,26],*[3,9,15,21,29]]
    elif sensor == "EMG":
        cols = [*range(30,44)]
    elif sensor == "all":
        cols = [*[0,2,6,8,12,14,18,20,25,26],*[3,9,15,21,29],*range(30,44), *range(52,55)]

    window_size = int(window_time * freq)

    #Create a list of dictionaries
    w = 0
    windows = [{}]

    #Create a dictionary for each sensor (X,Y,Z and Positions) for the first window
    for col in cols:
        windows[w].update({data[col][0]:{"data" : np.zeros(window_size, dtype = float)} })
    if mode == "train":
        windows[w].update({"label" : np.zeros(window_size) }) 

    #Loop through every window
    n = 0
    for i in range(1, len(data[0])):
        j = i-1-(w * window_size)

        #Initialize a new window
        if j >= window_size:
            w = w + 1
            windows.append({})
            for col in cols:
                windows[w].update({data[col][0]:{"data" : np.zeros(window_size, dtype = float)} })
            if mode == "train":
                windows[w].update({"label" : np.zeros(window_size)}) 
            j = 0

        #Add data to the current window
        if j < window_size:
            for col in cols:
                windows[w][data[col][0]]["data"][j] = data[col][i] 
            if mode == "train":
                windows[w]["label"][j] = data[len(data)-1][i]

    windows.pop()
    # The last window is not full of data, so most of it is 0


    #Calculate features for each window
    for i in range(len(windows)):
        for col in cols:
            windows[i][data[col][0]]["mean"] = np.mean(windows[i][data[col][0]]["data"])        #Mean
            windows[i][data[col][0]]["max"] = np.max(windows[i][data[col][0]]["data"])          #Maximum
            windows[i][data[col][0]]["min"] = np.min(windows[i][data[col][0]]["data"])          #Minimum
            windows[i][data[col][0]]["std"] = np.std(windows[i][data[col][0]]["data"])          #Standard deviation
            windows[i][data[col][0]]["var"] = np.var(windows[i][data[col][0]]["data"])          #Variance
            windows[i][data[col][0]]["rms"] = np.sqrt(np.mean(windows[i][data[col][0]]["data"]**2))                                                 #Root mean square
            windows[i][data[col][0]]["skew"] = np.mean((windows[i][data[col][0]]["data"] - np.mean(windows[i][data[col][0]]["data"]))**3)           #Skewness
            windows[i][data[col][0]]["kurt"] = np.mean((windows[i][data[col][0]]["data"] - np.mean(windows[i][data[col][0]]["data"]))**4)           #Kurtosis
            windows[i][data[col][0]]["iqr"] = np.subtract(*np.percentile(windows[i][data[col][0]]["data"], [75, 25]))                               #Interquartile range
            windows[i][data[col][0]]["mad"] = np.mean(np.absolute(windows[i][data[col][0]]["data"] - np.mean(windows[i][data[col][0]]["data"])))    #Mean absolute deviation
            windows[i][data[col][0]]["ptp"] = np.ptp(windows[i][data[col][0]]["data"])                                                              #Peak to peak
            windows[i][data[col][0]]["energy"] = np.sum(windows[i][data[col][0]]["data"]**2)                                                        #Energy
            #windows[i][data[col][0]]["arCoeff"] = np.polyfit(np.arange(0,window_size), windows[i][data[col][0]]["data"], 1)                        #Linear regression
            #windows[i][data[col][0]]["arCoeff"] = windows[i][data[col][0]]["arCoeff"][0]                                                           #Slope of the linear regression               
        

        #Give a label to the window
        if mode ==  "train":
            sit = 0
            trans1 = 0
            trans2 = 0
            stand = 0
            if activity == "SiSt":
                for n in range(window_size):
                    if int(windows[i]["label"][n]) == 1:
                        sit = sit + 1
                    elif int(windows[i]["label"][n]) == 2:
                        stand = stand + 1
                    elif int(windows[i]["label"][n]) == 121:
                        trans1 = trans1 + 1
                    elif int(windows[i]["label"][n]) == 122:
                        trans2 = trans2 + 1

                if max(sit, stand, trans1, trans2) == sit:
                    windows[i]["label"] = 1
                elif max(sit, stand, trans1, trans2) == stand:
                    windows[i]["label"] = 2
                elif max(sit, stand, trans1, trans2) == trans1:
                    windows[i]["label"] = 121
                elif max(sit, stand, trans1, trans2) == trans2:
                    windows[i]["label"] = 122

            elif activity == "StSi":
                for n in range(window_size):
                    if int(windows[i]["label"][n]) == 1:
                        sit = sit + 1
                    elif int(windows[i]["label"][n]) == 2:
                        stand = stand + 1
                    elif int(windows[i]["label"][n]) == 211:
                        trans1 = trans1 + 1
                    elif int(windows[i]["label"][n]) == 212:
                        trans2 = trans2 + 1

                if max(sit, stand, trans1, trans2) == sit:
                    windows[i]["label"] = 1
                elif max(sit, stand, trans1, trans2) == stand:
                    windows[i]["label"] = 2
                elif max(sit, stand, trans1, trans2) == trans1:
                    windows[i]["label"] = 211
                elif max(sit, stand, trans1, trans2) == trans2:
                    windows[i]["label"] = 212

    return windows, len(cols)




def idxLabelChange(Data):

    for i in range(len(Data)):
        if Data[i][0] == "mode":
            m = i 
            break
    
    ChangeIdxs = []
    prev_label = int(Data[m][1])
    for i in range(2,len(Data[m])):
        current_label = int(Data[m][i])
        if current_label != prev_label:
            ChangeIdxs.append(i)
            if len(ChangeIdxs) >= 3:
                break
        prev_label = current_label

    return ChangeIdxs






#-------------------Main-------------------#


col = 0
sensor = "gyroscope"  #accelerometer, gyroscope, IMU, EMG, all
SiSt = 0
multi = 1
window_len = 0.025 #in seconds
feature_or_data = "Data"  #Features


thisFolderParent = os.getcwd()

candidate = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]

first_candidate = 0
last_candidate = 9

first_circuit = 1
last_circuit = 50


all_sensor = ["accelerometer", "gyroscope", "IMU", "EMG", "all"]


for sensor in all_sensor:

    all_window_len = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]

    for window_len in all_window_len:

        print("Sensor: " + sensor)
        print("Window length: " + str(window_len) + "s")

        for act in range(0,2):
            if act ==  1:
                activity = "SiSt"
            elif act == 0:
                activity = "StSi"

            for c in range(first_candidate, last_candidate+1):
                print("Candidate " + candidate[c])
                all_windows = []

                # Loop through all the circuits of each candidate
                for circ in range(first_circuit, last_circuit+1): 
                    #print("Circuit " + str(circ))
                    """
                    if act == 1:
                        if circ<10:
                            filename  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_SiSt"
                        else:
                            filename  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_SiSt"
                    else:
                        if circ<10:
                            filename  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_StSi"
                        else:
                            filename  = thisFolderParent + "/Data/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_StSi"    
                    """
                    hardrive_path = "/Volumes/UnionSine/IMEE 2022/Semester2/FYP/FYPcode"

                    if act == 1:
                        if circ<10:
                            filename  =  hardrive_path + "/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_SiSt"
                        else:
                            filename  =  hardrive_path + "/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_SiSt"
                    else:
                        if circ<10:
                            filename  =  hardrive_path + "/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + str(circ) + "_StSi"
                        else:
                            filename  =  hardrive_path + "/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + str(circ) + "_StSi"    


                    if os.path.exists(filename + ".csv"):

                        allData = Main(filename)

                        label_change = idxLabelChange(allData)

                        windows, nb_sensors = featureExtraction(allData, window_len, 500, "train", sensor, activity)


                        for n in range(len(windows)):
                            all_windows.append(windows[n])


                if feature_or_data == "Features":
                    #Transform dictionnary in array
                    features_only = []
                    labels = []
                    for i in range(len(all_windows)):
                        features_only.append([])
                        test = all_windows[i]
                        for j, header in enumerate(all_windows[i]):
                            if  header != "label":
                                for n, featur in enumerate(all_windows[i][header]):
                                    if featur != "data":
                                        test2 = all_windows[i][header][featur]
                                        features_only[i].append(all_windows[i][header][featur])
                            else:
                                labels.append(all_windows[i]["label"])
                    
                    headers = []
                    for j, header in enumerate(all_windows[0]):
                        if  header != "label":
                            for n, featur in enumerate(all_windows[i][header]):
                                if featur != "data":
                                    test2 = all_windows[i][header][featur]
                                    headers.append(header + "_" + featur)

                    nb_features = int(len(headers)/nb_sensors)

                else:
                    #Transform dictionnary in array
                    data_only = []
                    labels = []
                    for i in range(len(all_windows)):
                        data_only.append([])
                        test = all_windows[i]
                        for j, header in enumerate(all_windows[i]):
                            if  header != "label":
                                for n, featur in enumerate(all_windows[i][header]):
                                    if featur == "data":
                                        test2 = all_windows[i][header][featur]
                                        for d in range(len(all_windows[i][header][featur])):
                                            data_only[i].append(all_windows[i][header][featur][d])
                            else:
                                labels.append(all_windows[i]["label"])

                    headers = []
                    for j, header in enumerate(all_windows[0]):
                        if  header != "label":
                            for h in range(int(window_len*500)):
                                headers.append(header)

                    nb_features = int(int(window_len*500)*nb_sensors)


                #-------------------Write output file-------------------#

                
                if act == 1:

                    if feature_or_data == "Features":
                        end_name  = "AB" + candidate[c] + "_SiSt_" + str(window_len) +"s_" + str(nb_features) + "f_" + sensor
                    else:
                        end_name  = "AB" + candidate[c] + "_SiSt_" + str(window_len) +"s_data_" + sensor

                    if first_circuit<10:
                        #out_file_name  = thisFolderParent + "/Data/Window/"+ feature_or_data+"/SiSt/AB" + candidate[c] + "/" + end_name

                        out_file_name  = hardrive_path + "/Window/"+ feature_or_data+"/SiSt/AB" + candidate[c] + "/" + end_name

                    file = open(out_file_name +'.csv', 'w')
                    file.truncate()

                    for i in range(len(headers)):
                        file.write(headers[i] + ",")
                    file.write("label\n")

                    if feature_or_data == "Features":
                        for n in range(len(features_only)):
                            for i in range(len(headers)):
                                file.write(str(features_only[n][i]) + ",")
                            file.write(str(labels[n]) + "\n")

                    else:
                        for n in range(len(data_only)):
                            for i in range(len(headers)):
                                file.write(str(data_only[n][i]) + ",")
                            file.write(str(labels[n]) + "\n")

                    file.close()

                elif act == 0:

                    if feature_or_data == "Features":
                        end_name  = "AB" + candidate[c] + "_StSi_" + str(window_len) +"s_" + str(nb_features) + "f_" + sensor
                    else:
                        end_name  = "AB" + candidate[c] + "_StSi_" + str(window_len) +"s_data_" + sensor

                    if first_circuit<10:

                        #out_file_name  = thisFolderParent + "/Data/Window/"+ feature_or_data+"/StSi/AB" + candidate[c] + "/" + end_name
                        
                        out_file_name  = hardrive_path + "/Window/"+ feature_or_data+"/StSi/AB" + candidate[c] + "/" + end_name

                    file = open(out_file_name +'.csv', 'w')

                    file = open(out_file_name +'.csv', 'w')
                    file.truncate()

                    for i in range(len(headers)):
                        file.write(headers[i] + ",")
                    file.write("label\n")

                    if feature_or_data == "Features":
                        for n in range(len(features_only)):
                            for i in range(len(headers)):
                                file.write(str(features_only[n][i]) + ",")
                            file.write(str(labels[n]) + "\n")

                    else:
                        for n in range(len(data_only)):
                            for i in range(len(headers)):
                                file.write(str(data_only[n][i]) + ",")
                            file.write(str(labels[n]) + "\n")

                    file.close()


