import os
import sys
import time as t
import signal
import subprocess
import vid_annotater_bulk as vab
import procYOLOnu as pYn
import sort_forward_bulk as sfb
import cv2
import annotated_vid as avid
import downsample_frames as dof
import shutil

"""
Runs all the files
"""

global processes
processes = []

def on_kill(sig, frame):
  global output_file
  output_file.close()
  for process in processes:
    try:
      process.terminate()
    except Exception as E:
      print(E)
  raise



DEBUG=False

cur_dir = os.path.split(__file__)[0]

def run_sequential_files(do_downsample:bool, do_tod:bool, do_vid:bool, input_folder:str, output_folder:str,cfg_file:str,model_file:str,proc_threshold:int,proc_move:int,proc_overlap:float,vid_count:int,move_vids:bool,do_circle:bool,circle_interval:int, do_yolo:bool, yolo_path):
  count = 0
  for file in os.listdir(input_folder):
    try:
      count+=1
      cur_in = input_folder
      file_id = file[0:-4]
      (file_id+"!")
      if do_downsample and do_yolo:
        cur_out = os.path.join(output_folder,"reduced-frame-healthspan")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        cur_out_file = os.path.join(cur_out,file_id+".avi")
        cur_in_file = os.path.join(cur_in,file_id+".avi")
        dof.downsample_vid(cur_in_file,cur_out_file)
        cur_in = cur_out

      # Run YOLO
      if do_yolo:
        cur_in_file = os.path.join(cur_in,file_id+".avi")
        print(cur_in_file)
        cur_out = os.path.join(output_folder,"yolo")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        vab.runYOLOonOne(model_file,cfg_file,cur_in_file,cur_out,do_circle,circle_interval)
      else:
        cur_out = yolo_path

      # Run Sort
      cur_in = cur_out
      cur_out = os.path.join(output_folder,"sorted-by-id")
      if not os.path.exists(cur_out):
        os.mkdir(cur_out)
      cur_in_file = os.path.join(cur_in,file_id+".csv")
      cur_out_file = os.path.join(cur_out,file_id+".csv")
      sfb.sortOne(cur_in_file,cur_out_file)

      # procYOLOnu
      # Don't change cur_in! Still pulls from yolo data
      if do_tod:
        cur_in_file = os.path.join(cur_in,file_id+".csv")
        cur_out = os.path.join(output_folder,"timelapse-analysis")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        cur_out_file = os.path.join(cur_out,file_id+".csv")
        pYn.procYOLOonOne(cur_in_file,cur_out_file,proc_threshold,proc_move,proc_overlap)


      if do_vid and do_tod and count % vid_count == 0:
        cur_out = os.path.join(output_folder,"vids")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)

        if do_downsample and do_yolo:
          orig_video = os.path.join(os.path.join(output_folder,"reduced-frame-healthspan"),file)
        else:
          orig_video = os.path.join(input_folder,file)
        yolo_csv = os.path.join(os.path.join(output_folder,"yolo"),file_id+".csv")
        tod_csv = os.path.join(os.path.join(output_folder,"timelapse-analysis"),file_id+".csv")
        out_vid = os.path.join(cur_out,file_id+".avi")
        avid.makeVideo(orig_video,yolo_csv,tod_csv,out_vid)
      if move_vids:
        cur_out = os.path.join(output_folder,"finished")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        in_file = os.path.join(input_folder,file)
        out_file = os.path.join(cur_out,file)
        shutil.move(in_file,out_file)
    except Exception as E:
      print(file + "unsuccessful\n" + str(E))
      output_file_path = os.path.join(cur_dir,"log.txt")
      output_file = open(output_file_path,"a")
      output_file.write(file+"\n")
      output_file.write(str(E)+"\n")
      output_file.close()

def run_all_files(do_downsample:bool,do_tod:bool,input_folder:str,output_folder:str,cfg_file:str,model_file:str):
    cur_in = input_folder
    if do_downsample:
        cur_out = os.path.join(output_folder,"reduced-frame-healthspan")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        run_downsample(input_folder, cur_out)
        cur_in = cur_out

    cur_out = os.path.join(output_folder,"yolo")
    if not os.path.exists(cur_out):
          os.mkdir(cur_out)
    run_yolo(cur_in, cur_out, cfg_file,model_file)
    cur_in = cur_out

    cur_out = os.path.join(output_folder,"sorted-by-id")
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
    if not DEBUG:
      pass
    else:
      for file in os.listdir(input_folder):
        open(os.path.join(output_folder,file),"w+")
        t.sleep(3)

def run_sort(input_folder, output_folder):
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
  output_file_path = os.path.join(cur_dir,"log.txt")
  output_file = open(output_file_path,"w+")
  output_file.write(str(os.getpid())+"\n")
  output_file.close()

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
  vid_count = int(sys.argv[11])
  move_vids = sys.argv[12] == "True"
  do_circle = sys.argv[13] == "True"
  circle_val = int(sys.argv[14])
  do_yolo = sys.argv[15] == "True"
  print(do_yolo)
  yolo_path = sys.argv[16]

  #run_all_files(dwnsmpl,tod,in_path,out,cfg_file,weight_file)
  run_sequential_files(dwnsmpl,tod,vid,in_path,out,cfg_file,weight_file,proc_thresh,proc_move,proc_overlap,vid_count,move_vids,do_circle,circle_val, do_yolo, yolo_path)