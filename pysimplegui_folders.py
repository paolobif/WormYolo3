import PySimpleGUI as sg
import os
import time as t
import subprocess
import numpy as np
import signal

# Just count how many of the expected number of files there currently are

# Technically this isn't necessary, but it makes it more clear
global cur_prog
global max_prog
global bar_prog
global is_runniing
global out_directory
global in_directory
global window
global current_process

cur_prog=0
max_prog=0
total_prog = 0
number_of_functions = 0
is_running = False
current_process = None


cur_dir = os.path.split(__file__)[0]

def get_folder(in_path):
    all_files = []
    for file in os.listdir(in_path):
        file_path = in_path +"/"+file
        if os.isdir(file_path):
            all_files += get_folder(file_path)
        else:
            all_files.append(file)
    max_prog = len(all_files)
    return all_files

def updateProg():
    global cur_prog, total_prog, max_prog, in_directory, out_directory, current_process
    if current_process.poll()!=None:
        current_process = None
        print("Done!")
        total_prog = 0
        cur_prog = 0
        return
    vids = []
    # For every video we want to process, add it to our list
    for video in os.listdir(in_directory):
        vids.append(video.split(".")[0])
    folders=[]

    least_ids = None
    # For every folder we've created in our out_directory
    for folder in os.listdir(out_directory):
        ids = set()
        # Determine which video ids have been processed
        for file in os.listdir(os.path.join(out_directory,folder)):
            for video in vids:
                # If the video name (without .avi) is in the file, that video has been processed
                if video in file:
                    ids.add(video)
        folders.append(len(ids))
        if least_ids != None:
            if len(ids) < len(least_ids):
                least_ids = ids

        else:
            least_ids = ids
    # Completed processes have processed the same number of videos - somewhat deceptive, since we create the file before finishing

    # How many folders have the largest number of files

    #try:
    #print(folders)
    if len(folders)!= 0:
        folders = np.where(np.array(folders)==np.max(folders))
        cur_prog = len(folders[0])
    else:
        cur_prog = 0
    #except:
    #    cur_prog = 0

    if not least_ids is None:
        total_prog = len(least_ids)
    else:
        total_prog = 0
    max_prog = len(os.listdir(in_directory))

def start_running(do_downsample:bool,do_tod:bool,input_folder:str,output_folder:str,cfg_file:str,weight_file:str):
    if not (os.path.exists(input_folder) and os.path.exists(output_folder) and os.path.exists(cfg_file) and os.path.exists(weight_file)):
        return False
    window["-SELECT_GO-"].disabled = True
    do_downsample = str(do_downsample)
    do_tod = str(do_tod)
    run_proc = os.path.join(cur_dir,"run_all_processes.py")
    args = ["python",run_proc,do_downsample,do_tod,input_folder,output_folder,cfg_file,weight_file]
    print(" ".join(args))
    global current_process
    current_process = subprocess.Popen(" ".join(args))
    #print(current_process.communicate())

def cancel():
    os.kill(current_process.pid, signal.SIGINT)
    print(current_process.communicate())
    #current_process.terminate()

model_folder = cur_dir + "/cfg"
weights_folder = cur_dir + "/weights"

sg.theme("LightGreen5")
layout = [
    [
        sg.Text('Model', size=(4, 1)),
        sg.Input(key="model", default_text="C:/Users/cdkte/Downloads/Worm-Yolo3GUI/cfg/yolov3-spp-1cls.cfg", size=(64, 2)),
        sg.FileBrowse(key="-FILEBROWSE-",initial_folder=model_folder,),
    ],
    [
        sg.Text('Weights', size=(4, 1)),
        sg.Input(key="weights", default_text="C:/Users/cdkte/Downloads/Worm-Yolo3GUI/weights/416_1_4_full_best200ep.pt", size=(64, 2)),
        sg.FileBrowse(key="-WEIGHTSBROWSE-",initial_folder=weights_folder,),


    ],
    [sg.Text('Input', size=(4, 1)),
        sg.Input(key="-IN_FOLDER-", default_text="C:/Users/cdkte/Downloads/in_test", size=(64, 2)),
        sg.FolderBrowse(key="-IN_BROWSE-",initial_folder="C:/",)
    ],
    [sg.Text('Output', size=(4, 1)),
        sg.Input(key="-OUT_FOLDER-", default_text="C:/Users/cdkte/Downloads/out_test", size=(64, 2)),
        sg.FolderBrowse(key="-OUT_BROWSE-",initial_folder="C:/",)
    ],
    [ #TODO: Add checkboxes
     sg.Checkbox("Downsample",key="-CHECK_DOWNSAMPLE-"),sg.Checkbox("Time of Death",key="-CHECK_TOD-")
    ],
    [
        sg.Button("Go", key = "-SELECT_GO-"),
        sg.Button("Cancel", key="-SELECT_CANCEL-"),
    ],
    [
        sg.ProgressBar(100,orientation='h', size=(20,20),key="progbar",border_width=4,bar_color=['Red','Green'])
    ],
    [
        sg.ProgressBar(100,orientation='h', size = (20,20), key = "total_progbar",border_width=4,bar_color=['LightGreen','Green'])
    ]
]

window = sg.Window('test reset browse', layout)

start = t.time()
while True:
    event, values = window.read(timeout=100)
    cur = t.time()
    val = cur-start
    #window['progbar'].update_bar(val)
    if event == "-SELECT_CANCEL-":
        #window["file"].Update("cancel!")
        cancel()

    elif event == "-SELECT_FILE-":
        #print(values["file"])
        if not os.path.exists(values["file"]):
            print("Not a valid file")
    elif event == "-SELECT_FOLDER-":
        pass
        #print(values)
        #print(values["-FOLDER-"])

    elif event == sg.WIN_CLOSED:
        break
    elif event == "-SELECT_GO-":
        input_folder = values["-IN_FOLDER-"]
        output_folder = values["-OUT_FOLDER-"]
        cfg_file = values["-FILEBROWSE-"]
        weight_file = values["-WEIGHTSBROWSE-"]
        downsample = values["-CHECK_DOWNSAMPLE-"]
        tod = values["-CHECK_TOD-"]
        start_running(downsample,tod,input_folder,output_folder,cfg_file,weight_file)
        number_of_functions = 2
        if downsample:
            number_of_functions += 1
        if tod:
            number_of_functions += 1
        out_directory = output_folder
        in_directory = input_folder
    if not current_process is None:
        updateProg()
        #print(cur_prog,number_of_functions,total_prog,max_prog)
        window['progbar'].update_bar(int(cur_prog/number_of_functions*100))
        window["total_progbar"].update_bar(int(total_prog/max_prog*100))
