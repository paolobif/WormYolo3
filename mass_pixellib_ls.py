import yolo_to_pixellib as ytp
import sys
import os
#import tensorflow as tf
import time as t
import multiprocessing
#from numba import cuda
#import torch 
#import gc

if __name__ == "__main__":
  gen_folder_path = sys.argv[1]
  if not os.path.isdir(gen_folder_path+"/Processed"):
    os.mkdir(gen_folder_path+"/Processed")
  all_avi = os.listdir(gen_folder_path+"/Raw")
  all_csv = os.listdir(gen_folder_path+"/Sorted")
  for file in all_avi:
    #vid_id = file.split("_")[0]
    #day = file.split("_")[1]
    #for file2 in all_csv:
    file2 = file.replace(".avi",".csv")
    """
    multiprocessing.get_context('spawn')
    p = multiprocessing.Process(target = ytp.runAllPixellib, args = (gen_folder_path+"/Raw/"+file,gen_folder_path+"/Neural/"+file2,gen_folder_path+"/Processed/"+file_name))
    p.start()
    
    p.join()
    
    
    
    """
    #ytp.runAllPixellib(gen_folder_path+"/Raw/"+file,gen_folder_path+"/Neural/"+file2,gen_folder_path+"/Processed/"+file_name)
    os.system("cd "+os.path.dirname(os.path.abspath(__file__)))
    os.system("python3 yolo_to_pixellib.py "+gen_folder_path+"/Raw/"+file +" "+gen_folder_path+"/Sorted/"+file2+" "+gen_folder_path+"/Processed/"+file)
    
    #torch.cuda.empty_cache()
    #"""
