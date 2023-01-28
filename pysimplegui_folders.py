import PySimpleGUI as sg
import os
import time as t
import subprocess
import numpy as np
import signal
import shutil
import download_ip

# I hope nobody ever has to interpret this in the future. I'm sorry...
# This is a patchwork that we kept shoving more stuff into

from sklearn.model_selection import validation_curve

# Just count how many of the expected number of files there currently are

# TODO: Screenshots of GUI

# TODO: Add SFW easter eggs! :)

# TODO: Rename Healthspan to Daily Monitor

# TODO: Fix bottom bar

# TODO: Try to allow seperate parts
# Second tab that takes input folder from YOLO/input videos/ output folder
# All the little substeps
# Skips YOLO analysis

# TODO: Download Tab
# Option for IP, timelapse vs. days

# Make 4 or 5 download files keyed to worm-bots

# Technically this isn't necessary, but it makes it more clear
global cur_prog
global max_prog
global bar_prog
global is_runniing
global out_directory
global in_directory
global window
global current_process, download_proc
global stored_weights
global stored_model


cur_prog=0
max_prog=1
total_prog = 0
number_of_functions = 0
is_running = False
current_process = None
download_proc = None


# TODO: Do a thing with this
INPUT_FOLDER = "/media/mlcomp/DrugAge/gui_test/in_test"
OUTPUT_FOLDER = "/media/mlcomp/DrugAge/gui_test/out_test"

file_path = os.path.abspath(__file__)
cur_dir = os.path.split(file_path)[0]
print('test',cur_dir,__file__)

py_path = shutil.which("python3")
if py_path is None:
    py_path = shutil.which("python")


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

def download_run(values):

    api_endpoint = values["WORMBOT"]
    input_string = values["DLIST"]
    save_path = values["DOWNLOAD_OUT"]

    if not (os.path.exists(save_path)):
        return False

    window["DOWNLOAD_GO"].update(disabled = True)
    window["DOWNLOAD_CANCEL"].update(disabled=False)
    window.finalize()

    run_proc = os.path.join(cur_dir,"download_ip.py")



    download_days = str(values["Daily Monitor"])
    first_day = values["FIRST_DAY"]
    last_day = values["LAST_DAY"]
    args = [py_path, run_proc,api_endpoint,input_string,save_path,download_days,first_day,last_day]
    print(" ".join(args))
    global download_proc
    download_proc = subprocess.Popen(" ".join(args),shell=True)
    return True

def download_cancel():
    global download_proc
    if download_proc is None:
        window["DOWNLOAD_ERROR"].update(value="Not Currently Running",background_color="DarkRed",visible=True)
        return
    log_file_path = os.path.join(cur_dir,"download_log.txt")
    log_file = open(log_file_path,"r")
    line_count = 0
    err_files = ""
    size = 1
    for line in log_file:
        if line_count == 0:
            os.kill(int(line),signal.SIGTERM)
        else:
            err_files += line
            size+=1
        line_count+=1

    #print(current_process.communicate())
    download_proc.kill()
    download_proc = None
    window["DOWNLOAD_GO"].update(disabled=False)
    window["DOWNLOAD_CANCEL"].update(disabled=True)
    if err_files!="":
        window["DOWNLOAD_ERROR"].update(value=err_files,background_color="DarkGreen",visible=True)
        window["DOWNLOAD_ERROR"].set_size((70,size))

    window.Finalize()

def updateDownloadProg(values):
    first_day = values["FIRST_DAY"]
    last_day = values["LAST_DAY"]
    do_days = values["Daily Monitor"]

    vid_args = values["DLIST"]
    num_vids = len(download_ip.parse_vids_arg(vid_args))
    if do_days:
        num_vids *= (int(last_day)-int(first_day))

    cur_vids = len(os.listdir(values["DOWNLOAD_OUT"]))

    window['download_progbar'].update_bar(int(cur_vids/num_vids*100))

    if int(cur_vids/num_vids*100) >= 100:
        download_cancel()



def start_running(do_downsample:bool,do_tod:bool,do_vids:bool,input_folder:str,output_folder:str,cfg_file:str,weight_file:str,vid_count:str,start_running:bool,do_circles:bool,circle_val:str,yolo_folder:str):
    print(cfg_file,weight_file)
    if not (os.path.exists(input_folder) and os.path.exists(output_folder)):
        return False
    if (not (os.path.exists(cfg_file) and os.path.exists(weight_file))) and not (os.path.exists(yolo_folder)):
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

    more_args = [window["VidCount"].get(),str(start_running),str(do_circles),circle_val]

    do_yolo = str(not cfg_file == "NA"); yolo_folder = window["yolo_input"].get()

    rep_yolo_args = [do_yolo, yolo_folder]

    args = [py_path,run_proc,do_downsample,do_tod,do_vids,input_folder,output_folder,cfg_file,weight_file]+proc_yolo_args + more_args + rep_yolo_args
    args.append("NA")
    print(" ".join(args))
    print(args)
    #print(args[-1])
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
            sg.Combo(["YOLO Processing","Reprocessing","Download"],default_value = "YOLO Processing",key="layout-options",tooltip = "Changes the mode of the GUI",enable_events=True),
        ],
        [
            sg.Text('Model', size=(6, 1),tooltip="The format of the model to be used in YOLO",key="model_label"),
            sg.Input(key="model", default_text=default_weight, size=(64, 2)),
            sg.FileBrowse(key="-FILEBROWSE-",initial_folder=model_folder),
        ],
        [
            sg.Text('Weights', size=(6, 1),tooltip = "The trained weights to use in YOLO",key="weight_label"),
            sg.Input(key="weights", default_text=os.path.join(weights_folder,"416_1_4_full_best200ep.pt"), size=(64, 2)),
            sg.FileBrowse(key="-WEIGHTSBROWSE-",initial_folder=weights_folder),

            sg.Text('YOLO Input', size=(6, 1),tooltip = "The trained weights to use in YOLO",key="yolo_label",visible = False),
            sg.Input(key="yolo_input", default_text="NA", size=(64, 2), visible = False),
            sg.FolderBrowse(key="yolo_browse", visible = False),

        ],
        [sg.Text('Video Input', size=(6, 1),tooltip = "The folder of videos to be processed"),
            sg.Input(key="-IN_FOLDER-", default_text="/media/mlcomp/DrugAge/gui_test/in_test", size=(64, 2)),
            sg.FolderBrowse(key="-IN_BROWSE-",initial_folder=INPUT_FOLDER,)
        ],
        [sg.Text('Output', size=(6, 1),tooltip="The folder to output results"),
            sg.Input(key="-OUT_FOLDER-", default_text="/media/mlcomp/DrugAge/gui_test/out_test", size=(64, 2)),
            sg.FolderBrowse(key="-OUT_BROWSE-",initial_folder=OUTPUT_FOLDER,)
        ],
        [
        sg.Checkbox("Move Videos",key = "-CHECK_MOVE-",tooltip = "Move finished videos to the output folder to show they're done."),
        sg.Checkbox("Daily Monitor",key="-CHECK_DOWNSAMPLE-",tooltip="Reduces the number of frames to be observed. Should only be used on long videos"),
        sg.Checkbox("Circle Crop", key = "-CHECK_CIRCLES-",tooltip = "Crops the video to only include the plate while running YOLO. May take longer but provide more accurate results.",enable_events= True),
        sg.Checkbox("Timelapse Analysis",key="-CHECK_TOD-",tooltip = "Determine time of death or paralysis",default = True,enable_events=True),
        sg.Checkbox("Create Videos",key = "-CHECK_VIDS-",tooltip = "Create videos of each worm with bounding boxes marking time of death", visible = True, enable_events = True)
        ],
        [
            sg.Combo(["Lifespan","Paralysis","Custom"],default_value = "Paralysis",key="procYOLOoptions",tooltip = "Determines when a worm can be called dead",enable_events=True),
            #invis options
            sg.Text("Threshold",size=(7,1),tooltip="Number of frames a worm has to be tracked in order to be analyzed(Default 2/2)",key="thresh-label",visible=False),
            sg.Input(key="proc-thresh",default_text="2",size=(4,1),visible=False),
            sg.Text("Slow Move",size=(8,1),tooltip="Number of frames overlapping by 'delta_overlap' before being called paralyzed or dead (Default 15/5)",key="slow-move-label",visible=False),
            sg.Input(key="proc-move",default_text="15",size=(4,1),visible=False),
            sg.Text("Delta Overlap",size=(10,1),tooltip="Percent overlap tor be called motionless (Default 0.95/0.8)",key="delta-overlap-label",visible=False),
            sg.Input(key="proc-overlap",default_text="0.8",size=(4,1),visible=False)
        ],
        [
            sg.Text("Skip videos", key = "skip_vid_text",visible = False,tooltip="Create 1 labelled video for each X original videos"),
            sg.Combo(["1","5","10","25"],default_value = "1", key = "VidCount",tooltip="Create 1 labelled video for each X original videos",visible = False),
            sg.Text("Circle Crop Interval",key="circle_int_text",visible = False,tooltip="The number of frames between adjusting the center of the cropped circle (Default 200)"),
            sg.Input(key="crop-value",default_text = "200",size=(4,1),visible = False)
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
        ],
    ]

    download_layout = [
        [sg.Text('WormBot Address', size=(12, 1),tooltip="The IP of the wormbot"),
            sg.Input(key="WORMBOT", default_text="128.95.23.15", size=(58, 2)),
        ],
        [sg.Text('Output', size=(6, 1),tooltip="The folder to output results"),
            sg.Input(key="DOWNLOAD_OUT", default_text="/media/mlcomp/DrugAge/gui_test/out_test", size=(64, 2)),
            sg.FolderBrowse(key="-DOWNLOAD_OUT-",initial_folder=OUTPUT_FOLDER,)
        ],
        [
            sg.Checkbox("Daily Monitor",key="Daily Monitor",enable_events=True),
            sg.Text('First Day', size=(6, 1),tooltip="The day to start downloading", visible = False),
            sg.Input(key="FIRST_DAY", default_text="3", size=(2, 1),visible = False),
            sg.Text('Last Day', size=(6, 1),tooltip="The final day to download", visible = False),
            sg.Input(key="LAST_DAY", default_text="20", size=(2, 1), visible = False),
        ],
        [sg.Text("Download List: ",tooltip = "Which videos to download. In the form of 2:5 for videos 2-5 and 3,5,7 to only download those videos. Can be combined, i.e., 2:5,7:8 to download 2-8 but not 6"),
         sg.Input(key = "DLIST", default_text = "2:5,3:6", size = (58, 2))
        ],
        [
            sg.Button("Go", key = "DOWNLOAD_GO"),
            sg.Button("Cancel", key="DOWNLOAD_CANCEL"),
        ],
        [
            sg.ProgressBar(100,orientation='h', size=(20,20),key="download_progbar",border_width=4,bar_color=['Red','Green'])
        ],
        [
            sg.Text(text="Test Error",key="DOWNLOAD_ERROR",size=(70,1),text_color="Red",background_color="DarkRed", visible=False)
        ]
    ]

    tabs = [[sg.TabGroup([[
        sg.Tab("Processing",layout=layout),
        sg.Tab("Download",layout = download_layout)
    ]

    ])]]

    window = sg.Window("YOLO Manager", tabs)
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
        vid_count = values["VidCount"]
        move_vids = values["-CHECK_MOVE-"]
        circle_crop = values["-CHECK_CIRCLES-"]
        circle_val = values["crop-value"]
        yolo_path = values["yolo_input"]
        print(downsample, cfg_file)
        if not start_running(downsample,tod,vids,input_folder,output_folder,cfg_file,weight_file,vid_count,move_vids,circle_crop,circle_val,yolo_path):
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
            window["-CHECK_VIDS-"].update(visible=False)
        else:
            window["-CHECK_VIDS-"].update(visible=True)
    elif event == "-CHECK_VIDS-":
        vid = values["-CHECK_VIDS-"]
        if vid:
            window["skip_vid_text"].update(visible = True)
            window["VidCount"].update(visible = True)
        else:
            window["skip_vid_text"].update(visible = False)
            window["VidCount"].update(visible = False)
            window["VidCount"].update(value = "1")
    elif event == "-CHECK_CIRCLES-":
        circs = values["-CHECK_CIRCLES-"]
        if circs:
            window["circle_int_text"].update(visible = True)
            window["crop-value"].update(visible = True)
        else:
            window["circle_int_text"].update(visible = False)
            window["crop-value"].update(visible = False)
            window["crop-value"].update(value = "200")

    elif event == "theme-options":
        print(values["theme-options"])
        if values["theme-options"]=="LightGreen":
            sg.theme("LightGreen5")
        if values["theme-options"]=="LightRed":
            sg.theme("DarkRed1")

        window.close()
        make_window()
    elif event == "layout-options":
        if values["layout-options"] != "YOLO Processing":
            stored_models = values["model"]
            stored_weights = values["weights"]
            window["model"].update(value="NA")
            window["weights"].update(value="NA")
            window["model"].update(visible = False)
            window["weights"].update(visible = False)
            window["model_label"].update(visible = False)
            window["weight_label"].update(visible = False)
            window["-WEIGHTSBROWSE-"].update(visible = False)
            window["-FILEBROWSE-"].update(visible = False)
            window["yolo_label"].update(visible = True)
            window["yolo_input"].update(visible = True)
            window["yolo_browse"].update(visible = True)

        else:
            window["model"].update(value=stored_models)
            window["weights"].update(value=stored_weights)
            window["yolo_label"].update(visible = False)
            window["yolo_input"].update(visible = False)
            window["yolo_browse"].update(visible = False)
        if values["layout-options"] == "YOLO Processing":
            window["model_label"].update(visible = True)
            window["weight_label"].update(visible = True)
            window["model"].update(visible = True)
            window["weights"].update(visible = True)
            window["-WEIGHTSBROWSE-"].update(visible = True)
            window["-FILEBROWSE-"].update(visible = True)

    elif event == "Daily Monitor":
        if values["Daily Monitor"]:
            window["FIRST_DAY"].update(visible = True)
            window["LAST_DAY"].update(visible = True)
        else:
            window["FIRST_DAY"].update(visible = False)
            window["LAST_DAY"].update(visible = False)
    elif event == "DOWNLOAD_GO":
        download_run(values)
    elif event == "DOWNLOAD_CANCEL":
        download_cancel()

    if not current_process is None:
        updateProg()
        #print(cur_prog,number_of_functions,total_prog,max_prog)
        window['progbar'].update_bar(int(cur_prog/number_of_functions*100))
        window["total_progbar"].update_bar(int(total_prog/max_prog*100))
    if not download_proc is None:
        updateDownloadProg(values)
