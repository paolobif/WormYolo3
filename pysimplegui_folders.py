import PySimpleGUI as sg
import os
import time as t
import subprocess

# Just count how many of the expected number of files there currently are
global cur_prog
global max_prog
global bar_prog
global is_runniing
cur_prog=0
max_prog=0
bar_prog=0
is_running = False
global current_process
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
def annotate_folder(in_path,output_path,model_path):
    bar_prog = cur_prog/max_prog


def start_running(do_downsample:bool,do_tod:bool,input_folder:str,output_folder:str,cfg_file:str,weight_file:str):
    if not (os.path.exists(input_folder) and os.path.exists(output_folder) and os.path.exists(cfg_file) and os.path.exists(weight_file)):
        return False
    do_downsample = str(do_downsample)
    do_tod = str(do_tod)
    run_proc = os.path.join(cur_dir,"run_all_processes.py")
    args = ["python",run_proc,do_downsample,do_tod,input_folder,output_folder,cfg_file,weight_file]
    print(" ".join(args))
    current_process = subprocess.Popen(" ".join(args))
    print(current_process.communicate())

def cancel():
    current_process.terminate()

model_folder = cur_dir + "/cfg"
weights_folder = cur_dir + "/weights"

sg.theme("LightGreen5")
layout = [
    [
        sg.Text('Model', size=(4, 1)),
        sg.Input(key="model", default_text="Select a Model", size=(64, 2)),
        sg.FileBrowse(key="-FILEBROWSE-",initial_folder=model_folder,),


    ],
    [
        sg.Text('Weights', size=(4, 1)),
        sg.Input(key="weights", default_text="Select a Weight", size=(64, 2)),
        sg.FileBrowse(key="-WEIGHTSBROWSE-",initial_folder=weights_folder,),


    ],
    [sg.Text('Input', size=(4, 1)),
        sg.Input(key="-IN_FOLDER-", default_text="Select a folder", size=(64, 2)),
        sg.FolderBrowse(key="-IN_BROWSE-",initial_folder="C:/",)
    ],
    [sg.Text('Output', size=(4, 1)),
        sg.Input(key="-OUT_FOLDER-", default_text="Select a folder", size=(64, 2)),
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
    window['progbar'].update_bar(val)
    if event == "-SELECT_CANCEL-":
        window["file"].Update("cancel!")
        # I want to reset also the FileBrowse initial path

        # Set initial folder here
        window["-FILEBROWSE-"].InitialFolder = r"C:\\"

    elif event == "-SELECT_FILE-":
        print(values["file"])
        if not os.path.exists(values["file"]):
            print("Not a valid file")
    elif event == "-SELECT_FOLDER-":
        print(values)
        print(values["-FOLDER-"])

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
