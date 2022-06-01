import os
import sys
"""
Runs all the files
"""


cur_dir = os.path.split(__file__)[0]

def run_all_files(do_downsample:bool,do_tod:bool,input_folder:str,output_folder:str,cfg_file:str,model_file:str):
    cur_in = input_folder
    if do_downsample:
        cur_out = os.path.join(output_folder,"downsample")
        if not os.path.exists(cur_out):
          os.mkdir(cur_out)
        run_downsample(input_folder, output_folder)
        cur_in = cur_out

    open(output_folder+"/test.txt","w+")
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
    vid_bulk_file = os.path.join(cur_dir,"vid_annotater_bulk.py")
    args = ["python",vid_bulk_file,input_folder,output_folder,cfg_file,model_file]
    os.system(" ".join(args))


def run_downsample(input_folder,output_folder):
    # TODO: Make downsample file and checkbox
    pass

def run_sort(input_folder, output_folder):
    # TODO: Make sort file and checkbox
    pass

def run_ToD(input_folder, output_folder):
    # TODO: Get time of death calls, add checkbox
    pass

if __name__ =="__main__":
  dwnsmpl = sys.argv[1] == "True"
  tod = sys.argv[2] == "True"
  in_path = sys.argv[3]
  out = sys.argv[4]
  cfg_file = sys.argv[5]
  weight_file =sys.argv[6]
  run_all_files(dwnsmpl,tod,in_path,out,cfg_file,weight_file)
