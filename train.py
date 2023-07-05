import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# For PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# For SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle

# For MLP
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

import numpy as np
from sklearn.model_selection import RandomizedSearchCV


# For CNN
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# For KNN
from sklearn.neighbors import KNeighborsClassifier

# For RF
from sklearn.ensemble import RandomForestClassifier


# For cross validation
from sklearn.model_selection import KFold

# General
from keras.utils import to_categorical
import time
from joblib import dump, load
import random



# Save the trained model to a file
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load a trained model from a file
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)



def readData(filename):

    data = []
    labels = []

    #Open .csv file for read        
    with open(filename +'.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Create objects for each header
        # Assign the name of each header to the objects
        for row in reader:
            for i in range(len(row)): 
                if row[i] == "label":
                    label_col = i
                    break
            break
        
        # Assign the value of each column to a different object
        for r,row in enumerate(reader):
            data.append([])
            for i in range(label_col+1):
                if row[i] != "":
                    if i == label_col :
                        labels.append(int(row[i]))
                    else:
                        data[r].append(float(row[i]))
             
    return data, labels


def reLabel(label, action):
    new_label = 0
    if action == 0:
        if label == 1:
            new_label = 0
        elif label == 2:
            new_label = 1
        elif label == 211:
            new_label = 2
        elif label == 212:
            new_label = 3
        else:
            new_label = label
    else:
        if label == 1:
            new_label = 0
        elif label == 2:
            new_label = 1
        elif label == 121:
            new_label = 2
        elif label == 122:
            new_label = 3
        elif label == 211:
            new_label = 4
        elif label == 212:
            new_label = 5
        else:
            new_label = label
        
    return new_label




# Function to create the MLP model
def create_mlp_model(len_input, hidden_nodes, activation):
    model = Sequential()
    model.add(Input(len_input))
    model.add(Dense(hidden_nodes, activation=activation))
    model.add(Dense(nb_class, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])


    return model




#-------------------Main-------------------#


action =  0  # 0 for StSi, 1 for SiSt, 2 for both
window_len = 0.5 #s
window_type = "Data"  #Data, Features
nb_features = 13
sensor = "EMG"  #accelerometer, gyroscope, IMU, EMG, all
method = "SVM"   #SVM, RF, KNN, CNN, DT, LR, NB, MLP
new = 1       # 1 foor new model, 0 for loading an old model
validation_method = "candidate"  # candidate, circuit



if action ==  0:
    a = 0
    b = 1
    activity = "StSi"
    nb_class = 4
elif action == 1:
    a = 1
    b = 2
    activity = "SiSt"
    nb_class = 4
elif action == 2:
    a = 0
    b = 2
    activity = "both"
    nb_class = 6

thisFolderParent = os.getcwd()

candidates = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]

first_candidate = 0
last_candidate = 9

methods = ["MLP"]

for method in methods:
    #all_sensor = ["accelerometer", "gyroscope", "IMU", "EMG", "all"]
    all_sensor = ["all"]


    for sensor in all_sensor:


        all_window_len = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
        if method == "MLP":
            all_window_len = [0.05]

        for window_len in all_window_len:

            print("method: " + method)
            print("Sensor: " + sensor)
            print("Window length: " + str(window_len) + "s")

            # Initialise the lists
            all_data = []
            all_labels = []


            for c in range(first_candidate, last_candidate+1):
                #print("Candidate " + candidate[c])

                if action == 2:
                    both_data = []
                    both_label = []

                for act in range(a, b):
                    if act ==  1:
                        act_feature = "SiSt"
                    elif act == 0:
                        act_feature = "StSi"
                    
                    hardrive_path = "/Volumes/UnionSine/IMEE 2022/Semester2/FYP/FYPcode"
                    #filename  = hardrive_path + "/Window/"+ window_type +"/"+ act_feature +"/AB" + candidates[c] + "/AB" + candidates[c] + "_" + act_feature + "_" + str(window_len) +"s_" + str(nb_features) + "f_" + sensor
                    if window_type == "Features":
                        filename  = hardrive_path + "/Window/"+ window_type +"/"+ act_feature +"/AB" + candidates[c] + "/AB" + candidates[c] + "_" + act_feature + "_" + str(window_len) +"s_" + str(nb_features) + "f_" + sensor
                    else:
                        filename  = hardrive_path + "/Window/"+ window_type +"/StSi/AB" + candidates[c] + "/AB" + candidates[c] + "_" + act_feature + "_" + str(window_len) +"s_data_" + sensor


                    data, labels = readData(filename)
                    
                    if validation_method == "none":
                        for i in range(len(data)):
                            all_data.append(data[i])

                            new_label = reLabel(labels[i], action)
                            all_labels.append(new_label)
                    else:
                        if action == 2:
                            for i in range(len(data)):
                                both_data.append(data[i])
                                labels[i] = reLabel(labels[i], action)
                                both_label.append(labels[i])
                        else:
                            all_data.append(data)

                            for i in range(len(labels)):
                                labels[i] = reLabel(labels[i], action)

                            all_labels.append(labels)

                if action == 2:
                    all_data.append(both_data)
                    all_labels.append(both_label)


            accuracy_max = 0
            accuracy_min = 100
            tot_accuracy = 0
            conf_matrix_tot = np.zeros((nb_class,nb_class))
            time_one_tot = 0
            nb_validations = 1
            for val in range(nb_validations):

                # Configure KFold cross-validation
                nb_fold = 10 # The number of folds for cross-validation
                kf = KFold(n_splits=nb_fold, shuffle=True, random_state=42)

                fold = 0
                for train_index, test_index in kf.split(all_data):
                    fold = fold + 1
                    print("Fold " + str(fold))

                    if validation_method == "none": 
                    
                        X_train, X_test = all_data[train_index], all_data[test_index]
                        Y_train, Y_test = all_labels[train_index], all_labels[test_index]
                    
                    else:

                        X_train = []
                        X_test = []
                        Y_train = []
                        Y_test = []
                        for candidate in train_index:
                            for w in range(len(all_data[candidate])):
                                X_train.append(all_data[candidate][w])
                                Y_train.append(all_labels[candidate][w])
                            
                        for candidate in test_index:
                            for w in range(len(all_data[candidate])):
                                X_test.append(all_data[candidate][w])
                                Y_test.append(all_labels[candidate][w])

                    rng = random.Random(42)
                    rng.shuffle(X_train)
                    rng = random.Random(42)
                    rng.shuffle(Y_train)


                    # Perform PCA
                    pca = PCA(n_components=0.99)       #define the number of components
                    pca.fit(X_train)               #Find the componet for our data
                    X_train_pca = pca.transform(X_train)
                    X_test_pca = pca.transform(X_test)

                    if method == "CNN":
                        if X_train_pca.shape[1] < 4:
                            pca = PCA(n_components=4)       #define the number of components
                            pca.fit(X_train)               #Find the componet for our data
                            X_train_pca = pca.transform(X_train)
                            X_test_pca = pca.transform(X_test)

                    #----------------------plot PCA variance ----------------------#
                    # Determine amount of variance explained by components
                    #print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))

                    """
                    # Plot the explained variance
                    plt.plot(pca.explained_variance_ratio_)
                    plt.title('Variance Explained by Extracted Componenents')
                    plt.ylabel('Variance')
                    plt.xlabel('Principal Components')
                    plt.show() 
                    """

                    # Normalise the data sets
                    min_max_scaler = MinMaxScaler() 
                    min_max_scaler.fit(X_train_pca)   #Normalise the components
                    X_train_pca_norm = min_max_scaler.transform(X_train_pca)
                    X_test_pca_norm = min_max_scaler.transform(X_test_pca)



                    print("X_train length: ", len(X_train))

                    if method == "SVM":
                        
                        """
                        # Create a pipeline to train the SVM model using the RBF kernel
                        svm_pipeline = Pipeline([
                            ('svm', SVC(kernel='rbf', C=1, gamma='scale', cache_size=1000, probability=True))
                        ])

                        # Perform grid search for hyperparameter tuning
                        param_grid = {
                            'svm__C': [0.1, 1, 10],
                            'svm__gamma': ['scale', 'auto', 0.1, 1, 10]
                        }
                        
                        grid_search = GridSearchCV(svm_pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)
                        grid_search.fit(X_train_pca_norm, Y_train)



                        # Train the SVM model with the best parameters
                        model = grid_search.best_estimator_

                        """
                        
                        # Create the SVM model
                        model = SVC(kernel='rbf', C=1, gamma='scale', cache_size=1000, probability=True)

                        # Train the model
                        model.fit(X_train_pca_norm, Y_train)


                
                    elif method == "MLP":
                        
                        X_train_list = []
                        for i in range(len(X_train_pca_norm)):
                            X_train_list.append(X_train_pca_norm[i].tolist())   


                        Y_train = tf.keras.utils.to_categorical(Y_train, nb_class)
                        Y_train = Y_train.astype(int)
                        Y_test = tf.keras.utils.to_categorical(Y_test, nb_class)
                        Y_test = Y_test.astype(int)
                        Y_train = Y_train.tolist()
                        Y_test = Y_test.tolist()

                        #model=create_mlp_model(len(X_train_list[0]), 16, 'relu')

                        model = Sequential()    #Define the type of model
                        model.add(Input(shape = len(X_train_list[0])))      # Input layer
                        model.add(Dense(units = 32, activation = 'relu'))       # Hidden layer
                        model.add(Dense(units = nb_class, activation = 'softmax'))      # Output layer
                        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])        # Create the model
                        model.fit(X_train_list, Y_train, epochs = 10, batch_size = 15)      # Train the model




                    elif method == "CNN":

                        Y_train = tf.keras.utils.to_categorical(Y_train, nb_class)
                        Y_train = Y_train.astype(int)
                        Y_test = tf.keras.utils.to_categorical(Y_test, nb_class)
                        Y_test = Y_test.astype(int)

                        # Reshape the input for the CNN (batch_size, num_features, 1)
                        X_train_pca_norm = X_train_pca_norm.reshape(X_train_pca_norm.shape[0], X_train_pca_norm.shape[1], 1)
                        X_test_pca_norm = X_test_pca_norm.reshape(X_test_pca_norm.shape[0], X_test_pca_norm.shape[1], 1)


                        # Reshape the input for the CNN (batch_size, window_size, num_features)
                        test = X_train_pca_norm.shape[0]
                        test1 = X_train_pca_norm.shape[1]

                        input_shape = ( X_train_pca_norm.shape[1], X_train_pca_norm.shape[2])

                        # Create a CNN model
                        model = Sequential()       #Define the type of model
                        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))    # Input layer
                        model.add(MaxPooling1D(pool_size=2))        # Pooling layer
                        model.add(Flatten())        # Flatten the output of the convolutional layer
                        model.add(Dense(units = 32, activation='relu'))
                        model.add(Dropout(0.5))
                        model.add(Dense(nb_class, activation='softmax'))
                        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])   # Compile the model
                        model.fit(X_train_pca_norm, Y_train, epochs=10, batch_size=32, verbose=1)    # Train the model



                    elif method == "KNN":

                        # Create the KNN model
                        model = KNeighborsClassifier(n_neighbors=200 , metric='euclidean')
                        
                        # Train the model
                        model.fit(X_train_pca_norm, Y_train)




                    elif method == "RF":

                        # Create the Random Forest model
                        model = RandomForestClassifier(n_estimators=200, max_depth=40, min_samples_split=5, min_samples_leaf=1, random_state=42)

                        # Train the model
                        model.fit(X_train_pca_norm, Y_train)




                    start_time = time.time()                     # Start the timer
                    Y_predict = model.predict(X_test_pca_norm)   # Predict classes
                    time_all = time.time() - start_time          # Stop the timer
                    time_one = time_all/len(X_test_pca_norm)     # time for one prediction   

                
                    if method == "CNN" or method == "MLP":

                        # Convert the softmax probabilities into one-hot encoded vectors
                        y_pred = (Y_predict == Y_predict.max(axis=1)[:,None]).astype(int)
                        Y_pred_dec = []
                        Y_test_dec = []
                        for j in range(len(y_pred)):
                            Y_pred_dec.append(np.argmax(y_pred[j]))
                            Y_test_dec.append(np.argmax(Y_test[j]))
                        Y_predict = Y_pred_dec
                        Y_test = Y_test_dec


                    # Evaluate the model
                    accuracy = metrics.accuracy_score(Y_test, Y_predict)
                    conf_matrix = metrics.confusion_matrix(Y_test, Y_predict)
                    report = metrics.classification_report(Y_test, Y_predict)


                    print("test candiudate: ", test_index)
                    print("Accuracy = {}".format(np.round(accuracy,4)))
                    #print("Confusion Matrix:")
                    #print(conf_matrix)
                    #print("Classification Report:")
                    #print(report)




                    
                    tot_accuracy = tot_accuracy + accuracy
                    conf_matrix_tot = conf_matrix_tot + conf_matrix
                    time_one_tot = time_one_tot + time_one


                    if accuracy < accuracy_min:
                        accuracy_min = accuracy
                        worst_test_index = test_index


                    if accuracy > accuracy_max:
                        
                        accuracy_max = accuracy
                        best_time = time_one
                        conf_matrix_save = conf_matrix
                        report_save = report
                        best_test_index = test_index

                        X_test_save = X_test_pca_norm
                        Y_test_save = Y_test

                        
                        if window_type == "Features":
                            model_name = activity + "_" + method + "_" + sensor + "_" + str(nb_features) + "f_"+ str(window_len) + "s"
                        else:
                            model_name = activity + "_" + method + "_" + sensor + "_data_"+ str(window_len) + "s"

                        window_folder = str(window_len).replace(".","_")

                        thisFolderParent = '/Users/theodorebedos/Documents/IMEE 2022/Semester2/FYP/FYPcode/'

                        test_features_name = thisFolderParent + "/Models/" + method + "/" + window_folder + "/" + window_type + "/test_features_" + model_name + ".npy"
                        np.save(test_features_name, X_test_save)    

                        test_labels_name = thisFolderParent + "/Models/" + method  + "/" + window_folder + "/" + window_type + "/test_labels_" + model_name + ".npy"
                        np.save(test_labels_name, Y_test_save)

                        # Save the trained PCA model
                        pca_model_name = thisFolderParent + "/Models/" + method  + "/" + window_folder + "/" + window_type + "/PCA_" + model_name + ".joblib"
                        dump(pca, pca_model_name)

                        # Save the trained normalization model
                        norm_model_name = thisFolderParent + "/Models/" + method  + "/" + window_folder + "/" + window_type + "/Norm_" + model_name + ".joblib"
                        dump(min_max_scaler, norm_model_name)

                        model_save = thisFolderParent + "/Models/" + method + "/" + window_folder + "/" + window_type + "/" + method + "_" + model_name + "f_" + sensor

                        if method == "SVM" or method == "KNN" or method == "RF":
                            # Save the model
                            save_model(model, model_save)
                        
                        else:
                            # Save the model
                            model.save(model_save + '.h5')
                    

            
            avg_accuracy = tot_accuracy / (nb_fold * nb_validations)

            avg_time_one = time_one_tot / (nb_fold * nb_validations)

            # Normalize the confusion matrix by row
            row_sums = conf_matrix_tot.sum(axis=1, keepdims=True)
            normalized_conf_matrix = conf_matrix_tot / row_sums

            # Convert to percentage
            percentage_conf_matrix = normalized_conf_matrix * 100


            if window_type == "Features":
                model_name = activity + "_" + str(window_len) +"s_" + str(nb_features) + "f_" + sensor 
            else:
                model_name = activity + "_" + str(window_len) +"s_" + "data_" + sensor

            result_file = thisFolderParent + "/Models/" + method + "/" + window_folder + "/" + window_type + "/result_" + model_name + ".csv"


            with open(result_file, 'w', newline='') as file:
                file.truncate()

                file.write("Average Accuracy\n")
                file.write(str(np.round(avg_accuracy* 100,4)).replace(".",",")+ "\n")
                file.write("Average Time\n")
                file.write(str(format(np.round(avg_time_one* 1000,10))).replace(".",",")+ "\n")
                file.write("Total Confusion Matrix:\n")
                for row in conf_matrix_tot:
                    file.write(str(row) + "\n")
                file.write("Total Confusion Matrix in percentage:\n")
                for row in percentage_conf_matrix:
                    file.write(str(row) + "\n")

                file.write("Best Accuracy\n")
                file.write(str(format(np.round(accuracy_max * 100,4))).replace(".",",")+ "\n")
                file.write("Best test index\n")
                file.write(str(*best_test_index)+ "\n")
                file.write("Best time\n")
                file.write(str(format(np.round(best_time* 1000,10))).replace(".",",")+ "\n")
                file.write("Worst Accuracy\n")
                file.write(str(format(np.round(accuracy_min* 100,4))).replace(".",",")+ "\n")
                file.write("Worst test index\n")
                file.write(str(*worst_test_index)+ "\n")
                file.write("Best Confusion Matrix:\n")
                for row in conf_matrix_save:
                    file.write(str(row) + "\n")
        

            
            print("\n FINAL: \n\n")
            print("Best Confusion Matrix:")
            print(conf_matrix_save)
            print("Best Classification Report:")
            print(report_save)
            print("Best Accuracy = {}".format(np.round(accuracy_max,4)))
            print("Average Accuracy = {}".format(np.round(avg_accuracy,4)))
            print("Best time = {}".format(np.round(best_time,10)))
            print("Best test index = {}".format(*best_test_index))

            print("Worst Accuracy = {}".format(np.round(accuracy_min,4)))
            print("Worst test index = {}".format(*worst_test_index))
        
