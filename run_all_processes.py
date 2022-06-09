import os
import sys
from tabnanny import process_tokens
import time as t
import signal
import subprocess
import vid_annotater_bulk as vab
import procYOLOnu as pYn
import sort_forward_bulk as sfb
import cv2
import annotated_vid as avid

"""
Runs all the files
"""

global processes
processes = []

def on_kill(sig, frame):
  open("C:/Users/cdkte/Downloads/test.txt","w+")
  for process in processes:
    try:
      process.terminate()
    except Exception as E:
      print(E)
  raise



DEBUG=False

cur_dir = os.path.split(__file__)[0]

# TODO: Add try-catch so it continues if there is an error
# Tomorrow, Thursday, friday, sit down and where important files should be -
def run_sequential_files(do_downsample:bool, do_tod:bool, do_vid:bool, input_folder:str, output_folder:str,cfg_file:str,model_file:str,proc_threshold:int,proc_move:int,proc_overlap:float):
  try:
    count = 0
    for file in os.listdir(input_folder):
      count+=1
      cur_in = os.path.join(input_folder,file)
      file_id = os.path.split(cur_in)[-1][0:-4]

      if do_downsample:
        cur_out = os.path.join(output_folder,"downsample")
        pass
        cur_in = cur_out

      # Run YOLO
      cur_out = os.path.join(output_folder,"yolo")
      if not os.path.exists(cur_out):
        os.mkdir(cur_out)
      vab.runYOLOonOne(model_file,cfg_file,cur_in,cur_out)

      # Run Sort
      cur_in = cur_out
      cur_out = os.path.join(output_folder,"sort")
      if not os.path.exists(cur_out):
        os.mkdir(cur_out)
      cur_in_file = os.path.join(cur_in,file_id+".csv")
      cur_out_file = os.path.join(cur_out,file_id+".csv")
      sfb.sortOne(cur_in_file,cur_out_file)

      # procYOLOnu
      # Don't change cur_in! Still pulls from yolo data
      if do_tod:
        cur_in_file = os.path.join(cur_in,file_id+".csv")
        cur_out = os.path.join(output_folder,"tod")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        cur_out_file = os.path.join(cur_out,file_id+".csv")
        pYn.procYOLOonOne(cur_in_file,cur_out_file,proc_threshold,proc_move,proc_overlap)

      # TODO: Make this
      # Every bounding box is on frame, dead worms have bright bounding box and ID.
      # TODO: Possible issue YOLO output is xy-height while analyze sort may be x1,y1,x2,y2
      #if count%video_divide == 0:
      #  make_video()
      if do_vid:
        cur_out = os.path.join(output_folder,"vids")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)

        orig_video = os.path.join(input_folder,file)
        yolo_csv = os.path.join(os.path.join(output_folder,"yolo"),file_id+".csv")
        tod_csv = os.path.join(os.path.join(output_folder,"tod"),file_id+".csv")
        out_vid = os.path.join(cur_out,file_id+".avi")
        avid.makeVideo(orig_video,yolo_csv,tod_csv,out_vid)
  except Exception as E:
    print(file + "unsuccessful\n" + str(E))

def run_all_files(do_downsample:bool,do_tod:bool,input_folder:str,output_folder:str,cfg_file:str,model_file:str):
    cur_in = input_folder
    if do_downsample:
        cur_out = os.path.join(output_folder,"downsample")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        run_downsample(input_folder, cur_out)
        cur_in = cur_out

    cur_out = os.path.join(output_folder,"yolo")
    if not os.path.exists(cur_out):
          os.mkdir(cur_out)
    run_yolo(cur_in, cur_out, cfg_file,model_file)
    cur_in = cur_out

    cur_out = os.path.join(output_folder,"sort")
    if not os.path.exists(cur_out):
          os.mkdir(cur_out)
    run_sort(cur_in, cur_out)
    cur_in = cur_out

    if do_tod:
        cur_out = os.path.join(output_folder,"time_of_death")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        run_ToD(cur_in,cur_out)
        cur_in = cur_out


def run_yolo(input_folder, output_folder, cfg_file, model_file):
    # TODO: Add yolo file (Make yolo file)
    if not DEBUG:
      vid_bulk_file = os.path.join(cur_dir,"vid_annotater_bulk.py")
      args = ["python",vid_bulk_file,input_folder,output_folder,cfg_file,model_file]
      proc = subprocess.Popen(" ".join(args))
      proc.communicate()
      global processes
      processes.append(proc)
    else:
      for file in os.listdir(input_folder):
        open(os.path.join(output_folder,file),"w+")
        t.sleep(5)



def run_downsample(input_folder,output_folder):
    # TODO: Make downsample file and checkbox
    if not DEBUG:
      pass
    else:
      for file in os.listdir(input_folder):
        open(os.path.join(output_folder,file),"w+")
        t.sleep(3)

def run_sort(input_folder, output_folder):
    # TODO: Make sort file and checkbox
    if not DEBUG:
      vid_bulk_file = os.path.join(cur_dir,"sort_forward_bulk.py")
      args = ["python",vid_bulk_file,input_folder,output_folder]
      proc = subprocess.Popen(" ".join(args))
      proc.communicate()
      global processes
      processes.append(proc)
    else:
      for file in os.listdir(input_folder):
        open(os.path.join(output_folder,file),"w+")
        t.sleep(4)

def run_ToD(input_folder, output_folder):
    # TODO: Get time of death calls, add checkbox
    if not DEBUG:
      vid_bulk_file = os.path.join(cur_dir,"procYOLOnu.py")
      args = ["python",vid_bulk_file,input_folder,output_folder]
      proc = subprocess.Popen(" ".join(args))
      proc.communicate()
      global processes
      processes.append(proc)
    else:
      for file in os.listdir(input_folder):
        open(os.path.join(output_folder,file),"w+")
        t.sleep(3)


if __name__ =="__main__":
  #signal.signal(signal.SIGINT,on_kill)
  dwnsmpl = sys.argv[1] == "True"
  tod = sys.argv[2] == "True"
  vid = sys.argv[3] == "True"
  in_path = sys.argv[4]
  out = sys.argv[5]
  cfg_file = sys.argv[6]
  weight_file =sys.argv[7]
  proc_thresh = int(sys.argv[8])
  proc_move = int(sys.argv[9])
  proc_overlap = float(sys.argv[10])
  #run_all_files(dwnsmpl,tod,in_path,out,cfg_file,weight_file)
  run_sequential_files(dwnsmpl,tod,vid,in_path,out,cfg_file,weight_file,proc_thresh,proc_move,proc_overlap)