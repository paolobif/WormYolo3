import PySimpleGUI as sg
import os
import time as t
import subprocess
import numpy as np
import signal
import shutil


from sklearn.model_selection import validation_curve

# Just count how many of the expected number of files there currently are

# Possible TODO: Just process YOLO output, rather than run YOLO, such as add checkbox for YOLO

# TODO: Move file when done processing (or check if already processed) and run YOLO->Sort,etc. sequentially

# TODO: Track most recent and delete it when interrupted
# TODO: Add circle exclusion (check Paolo-dev in procYOLOnu)

# TODO: At the end, add option to make annotated videos
    #  Options: Every video, every 10th video, every 25th video

# TODO: Add SFW easter eggs! :)

# TODO: change window label

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
max_prog=1
total_prog = 0
number_of_functions = 0
is_running = False
current_process = None


# TODO: Do a thing with this
INPUT_FOLDER = "/media/mlcomp/DrugAge/gui_test/in_test"
OUTPUT_FOLDER = "/media/mlcomp/DrugAge/gui_test/out_test"

file_path = os.path.abspath(__file__)
cur_dir = os.path.split(file_path)[0]
print('test',cur_dir,__file__)

py_path = shutil.which("python")
if py_path is None:
    py_path = shutil.which("python3")


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
        window["-SELECT_GO-"].update(disabled=False)
        window["-SELECT_CANCEL-"].update(disabled=True)

        log_file_path = os.path.join(cur_dir,"log.txt")
        log_file = open(log_file_path,"r")
        line_count = 0
        err_files = "Failed Files\n"
        size = 1
        for line in log_file:
            if line_count % 2 == 1:
                err_files += line
                size+=1
            line_count+=1
        if err_files!="Failed Files\n":
            window["Error_Message"].update(value=err_files,background_color="DarkGreen",visible=True)
            window["Error_Message"].set_size((50,size))
        window.finalize()
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

def start_running(do_downsample:bool,do_tod:bool,do_vids:bool,input_folder:str,output_folder:str,cfg_file:str,weight_file:str):
    print(cfg_file,weight_file)
    if not (os.path.exists(input_folder) and os.path.exists(output_folder) and os.path.exists(cfg_file) and os.path.exists(weight_file)):
        return False
    #window["-SELECT_GO-"].disabled = True
    window["-SELECT_GO-"].update(disabled = True)
    window["-SELECT_CANCEL-"].update(disabled=False)
    window.finalize()
    do_downsample = str(do_downsample)
    do_tod = str(do_tod)
    do_vids = str(do_vids)
    run_proc = os.path.join(cur_dir,"run_all_processes.py")

    # Get procYOLO args
    proc_yolo_args = [window["proc-thresh"].get(),window["proc-move"].get(),window["proc-overlap"].get()]

    args = [py_path,run_proc,do_downsample,do_tod,do_vids,input_folder,output_folder,cfg_file,weight_file]+proc_yolo_args
    print(" ".join(args))
    global current_process
    current_process = subprocess.Popen(" ".join(args),shell=True)
    #print(current_process.communicate())
    return True

def cancel():
    global current_process
    if current_process is None:
        window["Error_Message"].update(value="Not Currently Running",background_color="DarkRed",visible=True)
        return
    print(current_process.pid)

    #os.kill(current_process.pid+1, signal.SIGTERM)

    log_file_path = os.path.join(cur_dir,"log.txt")
    log_file = open(log_file_path,"r")
    line_count = 0
    err_files = "Failed Files\n"
    size = 1
    for line in log_file:
        if line_count == 0:
            os.kill(int(line),signal.SIGTERM)
        if line_count % 2 == 1:
            err_files += line
            size+=1
        line_count+=1

    #print(current_process.communicate())
    current_process.kill()
    current_process = None
    window["-SELECT_GO-"].update(disabled=False)
    window["-SELECT_CANCEL-"].update(disabled=True)
    if err_files!="Failed Files\n":
        window["Error_Message"].update(value=err_files,background_color="DarkGreen",visible=True)
        window["Error_Message"].set_size((50,size))

    window.Finalize()

print(cur_dir,os.path.join(cur_dir, "cfg"))
model_folder = os.path.join(cur_dir, "cfg")
weights_folder = os.path.join(cur_dir, "weights")
default_weight = os.path.join(model_folder,"yolov3-spp-1cls.cfg")

sg.theme("LightGreen5")

def make_window():
    global layout, window
    layout = [
        [
            sg.Text('Model', size=(6, 1),tooltip="The format of the model to be used in YOLO"),
            sg.Input(key="model", default_text=default_weight, size=(64, 2)),
            sg.FileBrowse(key="-FILEBROWSE-",initial_folder=model_folder,),
        ],
        [
            sg.Text('Weights', size=(6, 1),tooltip = "The trained weights to use in YOLO"),
            sg.Input(key="weights", default_text=os.path.join(weights_folder,"416_1_4_full_best200ep.pt"), size=(64, 2)),
            sg.FileBrowse(key="-WEIGHTSBROWSE-",initial_folder=weights_folder,),


        ],
        [sg.Text('Input', size=(6, 1),tooltip = "The folder of videos to be processed"),
            sg.Input(key="-IN_FOLDER-", default_text="/media/mlcomp/DrugAge/gui_test/in_test", size=(64, 2)),
            sg.FolderBrowse(key="-IN_BROWSE-",initial_folder=INPUT_FOLDER,)
        ],
        [sg.Text('Output', size=(6, 1),tooltip="The folder to output results"),
            sg.Input(key="-OUT_FOLDER-", default_text="/media/mlcomp/DrugAge/gui_test/out_test", size=(64, 2)),
            sg.FolderBrowse(key="-OUT_BROWSE-",initial_folder=OUTPUT_FOLDER,)
        ],
        [
        sg.Checkbox("Healthspan",key="-CHECK_DOWNSAMPLE-",tooltip="Reduces the number of frames to be observed. Should only be used on long videos"),
        sg.Checkbox("Timelapse Analysis",key="-CHECK_TOD-",tooltip = "Determine time of death or paralysis",default = True,enable_events=True),
        sg.Checkbox("Create Videos",key = "-CHECK_VIDS-",tooltip = "Create videos of each worm with bounding boxes marking time of death")
        ],
        [
            sg.Combo(["Lifespan","Paralysis","Custom"],default_value = "Paralysis",key="procYOLOoptions",tooltip = "Determines when a worm can be called dead",enable_events=True),
            #invis options
            sg.Text("Threshold",size=(7,1),tooltip="Number of frames a worm has to be tracked in order to be analyzed(Default 2/2)",key="thresh-label",visible=False),
            sg.Input(key="proc-thresh",default_text="2",size=(4,1),visible=False),
            sg.Text("Slow Move",size=(8,1),tooltip="Number of frames overlapping by 'delta_overlap' before being called dead or paralyzed (Default 15/5)",key="slow-move-label",visible=False),
            sg.Input(key="proc-move",default_text="15",size=(4,1),visible=False),
            sg.Text("Delta Overlap",size=(10,1),tooltip="Percent overlap tor be called motionless",key="delta-overlap-label (Default 0.95/0.8)",visible=False),
            sg.Input(key="proc-overlap",default_text="0.8",size=(4,1),visible=False)

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
        ],
        [
            sg.Text(text="Test Error",key="Error_Message",size=(50,1),text_color="Red",background_color="DarkRed", visible=False)
        ],
        [
            sg.Combo(["LightGreen","LightRed"],default_value = "LightGreen",key="theme-options",tooltip = "Changes the color of the window",enable_events=True),
        ]
    ]
    window = sg.Window('test reset browse', layout)
make_window()

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


    elif event == sg.WIN_CLOSED:
        cancel()
        break
    elif event == "procYOLOoptions":
        print(values["procYOLOoptions"])
        if values["procYOLOoptions"] == "Custom":
            window["thresh-label"].update(visible=True)
            window["proc-thresh"].update(visible=True)
            window["slow-move-label"].update(visible=True)
            window["proc-move"].update(visible=True)
            window["delta-overlap-label"].update(visible=True)
            window["proc-overlap"].update(visible=True)
        else:
            window["thresh-label"].update(visible=False)
            window["proc-thresh"].update(visible=False)
            window["slow-move-label"].update(visible=False)
            window["proc-move"].update(visible=False)
            window["delta-overlap-label"].update(visible=False)
            window["proc-overlap"].update(visible=False)

        if values["procYOLOoptions"] == "Lifespan":
            window["proc-thresh"].update(value="2")
            window["proc-move"].update(value="15")
            window["proc-overlap"].update(value="0.95")

        elif values["procYOLOoptions"] == "Paralysis":
            window["proc-thresh"].update(value="2")
            window["proc-move"].update(value="5")
            window["proc-overlap"].update(value="0.8")


    elif event == "-SELECT_GO-":
        input_folder = values["-IN_FOLDER-"]
        output_folder = values["-OUT_FOLDER-"]
        cfg_file = values["model"]
        weight_file = values["weights"]
        downsample = values["-CHECK_DOWNSAMPLE-"]
        tod = values["-CHECK_TOD-"]
        vids = values["-CHECK_VIDS-"]
        if not start_running(downsample,tod,vids,input_folder,output_folder,cfg_file,weight_file):
            window["Error_Message"].update(value="Invalid Files",background_color="DarkRed",visible=True)
        else:
            window["Error_Message"].update(visible=False)
        window.refresh()
        number_of_functions = 2
        if downsample:
            number_of_functions += 1
        if tod:
            number_of_functions += 1
        if vids:
            number_of_functions += 1
        out_directory = output_folder
        in_directory = input_folder
    elif event == "-CHECK_TOD-":
        tod = values["-CHECK_TOD-"]
        if not tod:
            window["-CHECK_VIDS-"].update(value=False)

    elif event == "theme-options":
        print(values["theme-options"])
        if values["theme-options"]=="LightGreen":
            sg.theme("LightGreen5")
        if values["theme-options"]=="LightRed":
            sg.theme("DarkRed1")

        window.close()
        make_window()

    if not current_process is None:
        updateProg()
        #print(cur_prog,number_of_functions,total_prog,max_prog)
        window['progbar'].update_bar(int(cur_prog/number_of_functions*100))
        window["total_progbar"].update_bar(int(total_prog/max_prog*100))