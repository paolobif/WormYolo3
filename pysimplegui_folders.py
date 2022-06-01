import PySimpleGUI as sg
import os
import time as t

# Just count how many of the expected number of files there currently are
global cur_prog
global max_prog
global bar_prog
cur_prog=0
max_prog=0
bar_prog=0

def run_yolo():
    # TODO: Add yolo file (Make yolo file)
    pass

def run_downsample():
    # TODO: Make downsample file and checkbox
    pass

def run_sort():
    # TODO: Make sort file and checkbox
    pass

def run_ToD():
    # TODO: Get time of death calls, add checkbox
    pass

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

cur_dir = "\\".join(__file__.split("\\")[0:-1])
model_folder = cur_dir
weights_folder = cur_dir

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
    ]
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

