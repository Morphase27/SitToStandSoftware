import tkinter as tk
from tkinter import ttk  # ttk for modern themed widgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os, csv
from PIL import Image, ImageTk  # for image processing
from tkinter import messagebox

from joblib import dump, load
import pickle
from keras.models import load_model
import time

from sklearn import metrics
import threading

def do_nothing():
    pass


def nb_to_text(label):
    if label == 0:
        text = "Sit"
    elif label == 1:
        text = "Stand"
    elif label == 2:
        text = "Trans 1"
    else:   
        text = "Trans 2"

    return text

def data_collection(filename):

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
                    windows[i]["label"] = 0
                elif max(sit, stand, trans1, trans2) == stand:
                    windows[i]["label"] = 1
                elif max(sit, stand, trans1, trans2) == trans1:
                    windows[i]["label"] = 2
                elif max(sit, stand, trans1, trans2) == trans2:
                    windows[i]["label"] = 3

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
                    windows[i]["label"] = 0
                elif max(sit, stand, trans1, trans2) == stand:
                    windows[i]["label"] = 1
                elif max(sit, stand, trans1, trans2) == trans1:
                    windows[i]["label"] = 2
                elif max(sit, stand, trans1, trans2) == trans2:
                    windows[i]["label"] = 3

    return windows, len(cols)






# Load a trained model from a file
def Load_Model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)



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


def find_best_index(model, activity, window_size, format, sensor):
        
        thisFolderParent = "/Users/theodorebedos/Documents/IMEE 2022/Semester2/FYP/FYPcode/"

        if activity == 'Sit-to-Stand':
            act = "SiSt"
        else:
            act = "StSi"

        if format == "Features":
            model_name = act + "_" + str(window_size) + "s_13f_" + sensor
            format = "Features"
        else:
            model_name = act + "_" + str(window_size) + "s_data_" + sensor
            format = "Data"

        window_folder = str(window_size).replace(".","_")

        result_file = thisFolderParent + "/Models/" + model + "/" + window_folder + "/" + format + "/result_" + model_name + ".csv"

        if os.path.exists(result_file):

            with open(result_file, 'r') as csvfile:
                reader = csv.reader(csvfile)

                # Create objects for each header
                # Assign the name of each header to the objects
                for  i,row in enumerate(reader):
                    if row[0] == 'Best test index':
                        break
                
                for  i,row in enumerate(reader):
                    best_idx = int(row[0])
                    break
        


        return best_idx


def check_exist_circuit(activity, circuit, candidate_idx):

    candidates = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]

    candidate = candidates[candidate_idx]

    if activity == "Sit-to-Stand":
        act = "SiSt"
    else:
        act = "StSi"

    hardrive_path = "/Volumes/UnionSine/IMEE 2022/Semester2/FYP/FYPcode"
    if int(circuit)<10:
        filename  = hardrive_path + "/New/AB" + candidate + "/AB" + candidate + "_Circuit_00" + circuit + "_" + act
    else:
        filename  = hardrive_path + "/New/AB" + candidate+ "/AB" + candidate+ "_Circuit_0" + circuit + "_" + act

    #filename  = thisFolderParent + "/Data/AB156/Processed/AB156_Circuit_002_post"

    if os.path.exists(filename + ".csv"):
        return True
    else:
        return False




def check_exist_model(activity, model, window_size, format, sensor):



    thisFolderParent = "/Users/theodorebedos/Documents/IMEE 2022/Semester2/FYP/FYPcode/"

    if activity == 'Sit-to-Stand':
        act = "SiSt"
    else:
        act = "StSi"

    if format == "Features":
        model_name = act + "_" + str(window_size) + "s_13f_" + sensor
        format = "Features"
    else:
        model_name = act + "_" + str(window_size) + "s_data_" + sensor
        format = "Data"

    window_folder = str(window_size).replace(".","_")

    result_file = thisFolderParent + "/Models/" + model + "/" + window_folder + "/" + format + "/result_" + model_name + ".csv"

    if os.path.exists(result_file):
        return True
    else:
        return False



class MainWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Graph in GUI')
        self.geometry('800x600')

        # Create a Figure instance
        self.fig = Figure(figsize=(5, 4), dpi=100)

        # Create an Axes instance, that will be used to plot
        self.ax = self.fig.add_subplot(111)

        # Define options for the dropdown menus
        self.models = ["SVM", "MLP", "CNN", "KNN", "RF"]
        self.sensors = ["accelerometer", "gyroscope", "IMU", "EMG", "all"]
        self.format = ["Raw", "Features"]
        self.window_size = ["0.05", "0.075", "0.1", "0.15", "0.2", "0.3", "0.5"]
        self.activity = ["Sit-to-Stand", "Stand-to-Sit"]
        self.circuit = [str(i) for i in range(1, 51)]

        # Load and resize images
        path = "/Users/theodorebedos/Documents/IMEE 2022/Semester2/FYP/FYPcode/GUI_img/"

        play_img = Image.open(path + 'play_icon.png')
        play_img = play_img.resize((25, 25), Image.ANTIALIAS)
        self.play_image = ImageTk.PhotoImage(play_img)

        pause_img = Image.open( path + 'pause_icon.png')
        pause_img = pause_img.resize((30, 30), Image.ANTIALIAS)
        self.pause_image = ImageTk.PhotoImage(pause_img)

        reset_img = Image.open(path + 'reset_icon.png')
        reset_img = reset_img.resize((40, 40), Image.ANTIALIAS)
        self.reset_image = ImageTk.PhotoImage(reset_img)




        # Use grid instead of pack, and configure the weights
        self.grid_rowconfigure(0, weight=1)  # make row 0 expandable
        self.grid_columnconfigure(0, weight=3)  # results_frame will take 75% width
        self.grid_columnconfigure(1, weight=1)  # control_frame will take 25% width

        # Create a frame to hold the results
        self.results_frame = tk.Frame(self, borderwidth=1, relief="solid")
        self.results_frame.grid(row=0, column=0, sticky='nsew')  # use grid instead of pack

        # Configure the weights inside results_frame
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)  # label_frame will take 25% height
        self.results_frame.grid_rowconfigure(1, weight=3)  # graph_frame will take 75% height

        # Split the result frame into label and graph frames
        self.label_frame = tk.Frame(self.results_frame, borderwidth=2, relief="solid")
        self.label_frame.grid(row=0, column=0, sticky='nsew')  # use grid instead of pack

        self.graph_frame = tk.Frame(self.results_frame, borderwidth=2, relief="solid")
        self.graph_frame.grid(row=1, column=0, sticky='nsew')  # use grid instead of pack

        
        # Configure the weights inside results_frame
        self.label_frame.grid_columnconfigure(0, weight=1)
        self.label_frame.grid_columnconfigure(1, weight=1)
        self.label_frame.grid_columnconfigure(2, weight=1)
        self.label_frame.grid_rowconfigure(0, weight=1)  


        # Split the label frame into true and predcit frames
        self.true_frame = tk.Frame(self.label_frame, borderwidth=2, relief="solid")
        self.true_frame.grid(row=0, column=0, sticky='nsew')  # use grid instead of pack
        self.true_frame.grid_propagate(False)

        self.true_frame.grid_columnconfigure(0, weight=1)
        self.true_frame.grid_rowconfigure(0, weight=1)  
        self.true_frame.grid_rowconfigure(1, weight=1)

        self.predict_frame = tk.Frame(self.label_frame, borderwidth=2, relief="solid")
        self.predict_frame.grid(row=0, column=1, sticky='nsew')  # use grid instead of pack
        self.predict_frame.grid_propagate(False)

        self.predict_frame.grid_columnconfigure(0, weight=1)
        self.predict_frame.grid_rowconfigure(0, weight=1)  
        self.predict_frame.grid_rowconfigure(1, weight=1)

        self.time_frame = tk.Frame(self.label_frame, borderwidth=2, relief="solid")
        self.time_frame.grid(row=0, column=2, sticky='nsew')  # use grid instead of pack
        self.time_frame.grid_propagate(False)

        self.predict_frame.grid_columnconfigure(0, weight=1)
        self.predict_frame.grid_columnconfigure(1, weight=1)
        self.predict_frame.grid_rowconfigure(0, weight=1)  
        self.predict_frame.grid_rowconfigure(1, weight=1)


        # Create a canvas to draw the graph on
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



        # Create control panel on the right
        self.control_frame = tk.Frame(self)
        self.control_frame.grid(row=0, column=1, sticky='nsew')  # use grid instead of pack

        # Configure the weights inside results_frame
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_rowconfigure(0, weight=1) 
        self.control_frame.grid_rowconfigure(1, weight=2) 


        # Split the control frame into top and bottom frames
        self.circuit_frame = tk.Frame(self.control_frame)
        self.circuit_frame.grid(row=0, column=0, sticky='nsew')

        self.settings_frame = tk.Frame(self.control_frame, borderwidth=1, relief="solid")
        self.settings_frame.grid(row=1, column=0, sticky='nsew')

        # Create a separator between top and bottom frames of the control frame
        #self.separator = ttk.Separator(self.control_frame, orient='horizontal')
        #self.separator.pack(fill='x')

        # Configure the weights inside results_frame
        self.circuit_frame.grid_columnconfigure(0, weight=1)
        self.circuit_frame.grid_rowconfigure(0, weight=2) 
        self.circuit_frame.grid_rowconfigure(1, weight=1) 

        # Split the circuit frame into top and bottom frames
        self.top_circuit_frame = tk.Frame(self.circuit_frame, borderwidth=1, relief="solid")
        self.top_circuit_frame.grid(row=0, column=0, sticky='nsew')


        self.bot_circuit_frame = tk.Frame(self.circuit_frame, borderwidth=1, relief="solid")
        self.bot_circuit_frame.grid(row=1, column=0, sticky='nsew')

        self.bot_circuit_frame.grid_columnconfigure(0, weight=1)
        self.bot_circuit_frame.grid_columnconfigure(1, weight=1)
        self.bot_circuit_frame.grid_columnconfigure(2, weight=1)
        self.bot_circuit_frame.grid_rowconfigure(0, weight=1) 


        # label box     
        self.true_label_text = tk.StringVar()
        self.true_label_text.set('True Label')

        self.true_label_box = tk.Label(self.true_frame, textvariable=self.true_label_text)
        self.true_label_box.grid(row=0, column=0, padx=10, pady=10)

        self.true_label2_text = tk.StringVar()
        self.true_label2_text.set('')

        self.true_label2_box = tk.Label(self.true_frame, textvariable=self.true_label2_text, fg ='white', bg='white', font=("Helvetica", 20))
        self.true_label2_box.grid(row=1, column=0, padx=10, pady=10)



        self.predict_label_text = tk.StringVar()
        self.predict_label_text.set('Predict Label')

        self.predict_label_box = tk.Label(self.predict_frame, textvariable=self.predict_label_text)
        self.predict_label_box.grid(row=0, column=0, padx=10, pady=10)

        self.predict_label2_text = tk.StringVar()
        self.predict_label2_text.set('')

        self.predict_label2_box = tk.Label(self.predict_frame, textvariable=self.predict_label2_text, fg ='white' ,bg='white', font=("Helvetica", 20))
        self.predict_label2_box.grid(row=1, column=0, padx=10, pady=10)


        self.time_label_text = tk.StringVar()
        self.time_label_text.set('Single window time (ms) ')

        self.time_label_box = tk.Label(self.time_frame, textvariable=self.time_label_text)
        self.time_label_box.grid(row=0, column=0, padx=10, pady=10)

        self.time_label2_text = tk.StringVar()
        self.time_label2_text.set('')

        self.time_label2_box = tk.Label(self.time_frame, textvariable=self.time_label2_text, bg='white')
        self.time_label2_box.grid(row=0, column=1, padx=10, pady=10)


        self.acuracy_label_text = tk.StringVar()
        self.acuracy_label_text.set('Average accuracy (%) ')

        self.acuracy_label_box = tk.Label(self.time_frame, textvariable=self.acuracy_label_text)
        self.acuracy_label_box.grid(row=1, column=0, padx=10, pady=10)

        self.acuracy_label2_text = tk.StringVar()
        self.acuracy_label2_text.set('')

        self.acuracy_label2_box = tk.Label(self.time_frame, textvariable=self.acuracy_label2_text, bg='white')
        self.acuracy_label2_box.grid(row=1, column=1, padx=10, pady=10)






        # Dropdown menus

        #settings dropdown
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self.settings_frame, values=self.models, state="readonly")
        self.model_dropdown.grid(row=0, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack
        self.model_dropdown.current(0)

        self.sensor_var = tk.StringVar()
        self.sensor_dropdown = ttk.Combobox(self.settings_frame, values=self.sensors, state="readonly")
        self.sensor_dropdown.grid(row=1, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack
        self.sensor_dropdown.current(0)

        self.format_var = tk.StringVar()
        self.format_dropdown = ttk.Combobox(self.settings_frame, values=self.format, state="readonly")
        self.format_dropdown.grid(row=2, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack
        self.format_dropdown.current(0)

        self.window_size_var = tk.StringVar()
        self.window_size_dropdown = ttk.Combobox(self.settings_frame, values=self.window_size, state="readonly")
        self.window_size_dropdown.grid(row=3, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack
        self.window_size_dropdown.current(0)

        #circuit dropdown
        self.activity_var = tk.StringVar()
        self.activity_dropdown = ttk.Combobox(self.top_circuit_frame, values=self.activity, state="readonly")
        self.activity_dropdown.grid(row=0, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack
        self.activity_dropdown.current(0)

        self.circuit_var = tk.StringVar()
        self.circuit_dropdown = ttk.Combobox(self.top_circuit_frame, values=self.circuit, state="readonly")
        self.circuit_dropdown.grid(row=1, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack
        self.circuit_dropdown.current(0)

        self.candidate_label_text = tk.StringVar()
        self.candidate_label_text.set('Candidate')

        self.candidate_label_box = tk.Label(self.top_circuit_frame, textvariable=self.candidate_label_text, bg='white')
        self.candidate_label_box.grid(row=2, column=0, padx=10, pady=10, sticky='ew')


        # Control buttons
        self.play_button = tk.Button(self.bot_circuit_frame, image=self.play_image, width=40, height=40, command=self.thread_start)
        self.play_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')  # use grid instead of pack

        self.pause_button = tk.Button(self.bot_circuit_frame, image=self.pause_image, width=40, height=40, command=self.pause)
        self.pause_button.grid(row=0, column=1, padx=10, pady=10, sticky='ew')  # use grid instead of pack

        self.reset_button = tk.Button(self.bot_circuit_frame, image=self.reset_image, width=40, height=40, command=self.reset)
        self.reset_button.grid(row=0, column=2, padx=10, pady=10, sticky='ew')  # use grid instead of pack


        self.load_model = None
        self.load_PCA= None
        self.load_norm = None
        self.best_idx = None

        self.stop = False
        self.stop_flag = False


        # Set dropdown menus to call update_graph on any change
        self.circuit_dropdown.bind('<<ComboboxSelected>>', self.update_model)
        self.activity_dropdown.bind('<<ComboboxSelected>>', self.update_model)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.update_model)
        self.sensor_dropdown.bind('<<ComboboxSelected>>', self.update_model)
        self.format_dropdown.bind('<<ComboboxSelected>>', self.update_model)
        self.window_size_dropdown.bind('<<ComboboxSelected>>', self.update_model)        

        # Display the default graph
        self.update_model()

    


    def update_model(self, event=None):
        # Get the selected value
        selected_model = self.model_dropdown.get()
        if selected_model == "SVM":
            self.window_size = ["0.1", "0.15", "0.2", "0.3", "0.5"]
            self.window_size_dropdown = ttk.Combobox(self.settings_frame, values=self.window_size, state="readonly")
            self.window_size_dropdown.current(0)
        else:
            self.window_size = ["0.05", "0.075", "0.1", "0.15", "0.2", "0.3", "0.5"]
            self.window_size_dropdown = ttk.Combobox(self.settings_frame, values=self.window_size, state="readonly")
            self.window_size_dropdown.current(0)

        selected_sensor = self.sensor_dropdown.get()
        selected_format = self.format_dropdown.get()
        selected_window_size = self.window_size_dropdown.get()
        selected_circuit = self.circuit_dropdown.get()
        selected_activity = self.activity_dropdown.get()

        # Set the StringVar to the selected value
        self.model_var.set(selected_model)
        self.sensor_var.set(selected_sensor)
        self.format_var.set(selected_format)
        self.window_size_var.set(selected_window_size)
        self.circuit_var.set(selected_circuit)
        self.activity_var.set(selected_activity)


        self.update_graph()
        



    def update_graph(self, event=None):
        circuit = self.circuit_var.get()
        activity = self.activity_var.get()

        model = self.model_var.get()
        sensor = self.sensor_var.get()
        format = self.format_var.get()
        window_size = self.window_size_var.get()


        exist_model = check_exist_model(activity, model, window_size, format, sensor)
        if exist_model == False:
            messagebox.showinfo("Error", "This model has not been trained")
        else:
            best_idx = find_best_index(model, activity, window_size, format, sensor)

            self.best_idx = best_idx

            self.load(activity,model, sensor, window_size, format)

            candidates = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]

            candidate = candidates[best_idx]

            self.candidate_label_text.set('Candidate ' + candidate)

            exist_circuit = check_exist_circuit(activity, circuit, best_idx)

            if exist_circuit == False:
                messagebox.showinfo("Error", "This circuit for this candidate does not exist")      
            else:
                self.display_graph(circuit, activity, best_idx)



    def load(self, activity,model, sensor, window_size, format):

        thisFolderParent = "/Users/theodorebedos/Documents/IMEE 2022/Semester2/FYP/FYPcode/"

        if activity == 'Sit-to-Stand':
            act = "SiSt"
        else:
            act = "StSi"

        if format == "Features":
            model_name = act + "_" + model + "_" +  sensor + "_13f_" +  str(window_size) + "s"
            format = "Features"
        else:
            model_name = act + "_" + model + "_" +  sensor + "_data_" +  str(window_size) + "s"
            format = "Data"

        window_folder = str(window_size).replace(".","_")

        pca_model_name = thisFolderParent + "/Models/" + model  + "/" + window_folder + "/" + format + "/PCA_" + model_name + ".joblib"
        norm_model_name = thisFolderParent + "/Models/" + model  + "/" + window_folder + "/" + format + "/Norm_" + model_name + ".joblib"
        model_save = thisFolderParent + "/Models/" + model + "/" + window_folder + "/" + format + "/" + model + "_" + model_name + "f_" + sensor

        self.load_PCA = load(pca_model_name)
        self.load_norm = load(norm_model_name)

        if model == "SVM" or model == "KNN" or model == "RF":
            # Save the model
            self.load_model = Load_Model(model_save)

        else:
            # load the model
            self.load_model = load_model(model_save + '.h5')



    def display_graph(self, circuit, activity, candidate_idx):
        # Clear the previous graph
        self.ax.clear()
        
        candidates = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]

        candidate = candidates[candidate_idx]

        if activity == "Sit-to-Stand":
            act = "SiSt"
        else:
            act = "StSi"


        hardrive_path = "/Volumes/UnionSine/IMEE 2022/Semester2/FYP/FYPcode"
        if int(circuit)<10:
            filename  = hardrive_path + "/New/AB" + candidate + "/AB" + candidate + "_Circuit_00" + circuit + "_" + act
        else:
            filename  = hardrive_path + "/New/AB" + candidate+ "/AB" + candidate+ "_Circuit_0" + circuit + "_" + act

        #filename  = thisFolderParent + "/Data/AB156/Processed/AB156_Circuit_002_post"

        if os.path.exists(filename + ".csv"):

            allData = readData(filename)

            for i in range(len(allData)):
                if allData[i][0] == "mode":
                    m = i 
                    break
            
            indexChange = []
            prev_label = int(allData[m][1])
            for i in range(2,len(allData[m])):
                current_label = int(allData[m][i])
                if current_label != prev_label:
                    indexChange.append(i)
                prev_label = current_label

            
            right_hip_angle = allData[m-3][1:]
            left_hip_angle = allData[m-2][1:]
            hip_tilt = allData[m-1][1:]
        
            freq = 500
            time_axis = [i/freq for i in range(len(allData[0][1:]))]

            # Plotting the data on the Axes instance
            self.ax.plot(time_axis, right_hip_angle, label="Right Hip Angle")
            self.ax.plot(time_axis, left_hip_angle, label="Left Hip Angle")
            self.ax.plot(time_axis, hip_tilt, label="Hip Tilt")

            for idx in indexChange:
                self.ax.axvline(x = idx/freq, color = 'b', linestyle = 'dashed')

            self.progress_line =self.ax.axvline(x=0, color='r')

            # Redraw the canvas
            self.canvas.draw()


    def change_label_color(self, true, predict):

        if true == predict:
            self.true_label2_box.configure(bg='green')
            self.predict_label2_box.configure(bg='green')
        else:
            self.true_label2_box.configure(bg='red')
            self.predict_label2_box.configure(bg='red') 


    def thread_start(self):
        self.stop = False
        self.stop_flag = False
        self.thread = threading.Thread(target=self.start)
        self.thread.start()
        

    def start(self):
        # TODO: Implement this

        self.play_button.config(command=do_nothing)


        circuit = self.circuit_var.get()
        activity = self.activity_var.get()

        model = self.model_var.get()
        sensor = self.sensor_var.get()
        format = self.format_var.get()
        window_size = self.window_size_var.get()

        c = self.best_idx

        candidate = ["156","185", "186", "188", "189", "190", "191", "192", "193", "194"]

        if activity == "Sit-to-Stand":
            activity = "SiSt"
        else:
            activity = "StSi"

        hardrive_path = "/Volumes/UnionSine/IMEE 2022/Semester2/FYP/FYPcode"
        if int(circuit)<10:
            filename  =  hardrive_path + "/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_00" + circuit + "_" + activity
        else:
            filename  =  hardrive_path + "/New/AB" + candidate[c] + "/AB" + candidate[c] + "_Circuit_0" + circuit + "_" + activity
  

        if os.path.exists(filename + ".csv"):

            allData = data_collection(filename)

            windows = featureExtraction(allData, float(window_size), 500, "train", sensor, activity)[0]


            if format == "Features":
                #Transform dictionnary in array
                features_only = []
                labels = []
                for i in range(len(windows)):
                    features_only.append([])
                    test = windows[i]
                    for j, header in enumerate(windows[i]):
                        if  header != "label":
                            for n, featur in enumerate(windows[i][header]):
                                if featur != "data":
                                    test2 = windows[i][header][featur]
                                    features_only[i].append(windows[i][header][featur])
                        else:
                            labels.append(windows[i]["label"])
                
                headers = []
                for j, header in enumerate(windows[0]):
                    if  header != "label":
                        for n, featur in enumerate(windows[i][header]):
                            if featur != "data":
                                test2 = windows[i][header][featur]
                                headers.append(header + "_" + featur)

                X_test = features_only


            else:
                #Transform dictionnary in array
                data_only = []
                labels = []
                for i in range(len(windows)):
                    data_only.append([])
                    test = windows[i]
                    for j, header in enumerate(windows[i]):
                        if  header != "label":
                            for n, featur in enumerate(windows[i][header]):
                                if featur == "data":
                                    test2 = windows[i][header][featur]
                                    for d in range(len(windows[i][header][featur])):
                                        data_only[i].append(windows[i][header][featur][d])
                        else:
                            labels.append(windows[i]["label"])

                headers = []
                for j, header in enumerate(windows[0]):
                    if  header != "label":
                        for h in range(int(float(window_size)*500)):
                            headers.append(header)
                
            X_test = data_only

            #Load PCA and normalization
            pca = self.load_PCA
            norm = self.load_norm
        
            all_predictions = []
            w = 0
            tot_time = 0
            window_time = time.time()
            predict_label = "None"
            for l in range(len(labels)):
                

                test = X_test[l]
                X_sample = np.array(X_test[l]).reshape(1, -1)
                Y_test = labels[l]

                true_label = nb_to_text(Y_test)
                while time.time() - window_time < float(window_size):
                        self.true_label2_text.set(true_label)
                if l < len(labels)-1:
                    true_label = nb_to_text(labels[l+1])
                    self.true_label2_text.set(true_label)
                    self.change_label_color(true_label, predict_label)

                w = w + 1
                self.progress_line.set_xdata(float(window_size) * w )  # Update the position of the vertical line
                self.canvas.draw()  # Redraw only the changes
                self.canvas.flush_events()  # Ensure the display is updated

                if self.stop:
                    while self.stop:
                        time.sleep(0.1)
                        if self.stop_flag == True:
                            self.stop = False
                            self.stop_flag = False
                            return
                        pass

                window_time = time.time()           # Start the timer

                #Apply PCA and normalization
                X_test_pca = pca.transform(X_sample)
                X_test_pca_norm = norm.transform(X_test_pca)
                Y_predict = self.load_model.predict(X_test_pca_norm)   # Predict classes
                end_time = time.time() - window_time          # Stop the timer 
            
                if model == "CNN" or model == "MLP":
                    # Convert the softmax probabilities into one-hot encoded vectors
                    y_pred = (Y_predict == Y_predict.max(axis=1)[:,None]).astype(int)
                    Y_predict = np.argmax(y_pred)
                    Y_test = np.argmax(Y_test)
                    predict_label = nb_to_text(Y_predict)
                    all_predictions.append(Y_predict)      
                else:
                    predict_label = nb_to_text(Y_predict[0])
                    all_predictions.append(Y_predict[0])      

                self.change_label_color(true_label, predict_label)
                self.predict_label2_text.set(predict_label)
                self.time_label2_text.set(np.round(end_time*1000,4))

                tot_time = tot_time + end_time*1000

                len_predict = len(all_predictions)
                accuracy = metrics.accuracy_score(labels[:len_predict], all_predictions)
                self.acuracy_label2_text.set(round(accuracy * 100,2))
                     

            self.time_label_text.set('Average window (ms) ')
            self.time_label2_text.set(np.round(tot_time/len_predict, 4))

        pass


    def pause(self):
        # TODO: Implement this
        if self.stop == False:
            self.stop = not self.stop
    
            self.play_button.config(command=self.pause2)


    def pause2(self):
        self.stop = not self.stop
        self.play_button.config(command=do_nothing)


    def reset(self):
        # TODO: Implement this
        self.progress_line.set_xdata(0)
        self.canvas.draw()  # Redraw only the changes
        self.canvas.flush_events()  # Ensure the display is updated
        self.play_button.config(command=self.thread_start)

        self.time_label_text.set('Single window time (ms) ')
        self.time_label2_text.set('')

        self.stop_flag = True
        pass




if __name__ == "__main__":
    window = MainWindow()
    window.mainloop()