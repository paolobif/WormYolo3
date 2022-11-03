import os
import sys

DOWNSAMPLE = False
if __name__=="__main__":
    in_path = sys.argv[1]
    working_folders = []

    # Identify files to downsample & YOLO
    for file in os.listdir(in_path):
        if not os.path.isdir(in_path+"/"+file):
            continue
        if not len(file.split("_"))==1:
            continue
        if os.path.exists(in_path+"/"+file+"_downsample") and DOWNSAMPLE:
            continue
        if os.path.exists(in_path+"/"+file+"_yolo") and not DOWNSAMPLE:
            continue
        working_folders.append(file)

    for folder in working_folders:
        #print(folder)
        #"""
        fold_path = in_path+"/"+folder
        if DOWNSAMPLE:
            downsample_folder =in_path+"/"+folder+"_downsample"

            os.mkdir(downsample_folder)
            
            os.system("python3 downsample_30_frames.py "+fold_path+" "+downsample_folder)
            yolo_folder = fold_path+"_yolo_downsample"
            os.mkdir(yolo_folder)
            os.system("python3 vid_annotater_bulk.py "+downsample_folder+" "+yolo_folder)
        else:
            #os.system("python3 downsample_30_frames.py "+fold_path+" "+folder)
            yolo_folder = fold_path+"_yolo"
            os.mkdir(yolo_folder)
            os.system("python3 vid_annotater_bulk.py "+fold_path+" "+yolo_folder)
        #"""        

    # Run Sort
    working_folders = []
    for file in os.listdir(in_path):
        if not os.path.isdir(in_path+"/"+file):
            continue
        if len(file.split("sort"))!=1: # Don't repeat sort folders
            continue
        if len(file.split("yolo"))==1: # Only operate on yolo folders
            continue
        tod_path = "sort".join(file.split("yolo"))

        working_folders.append(in_path+"/"+file)
    print("Sorting Folders:")
    print(working_folders)
    for folder in working_folders:
        tod_path = "sort".join(folder.split("yolo")) #1020_yolo_downsample -> 1020_sort_downsample
        print(folder + " sorted to " + tod_path)
        try:
            os.mkdir(tod_path)
        except:
            continue
        os.system("python3 sort_forward_bulk.py "+ folder + " " +tod_path)

    # Run Pixellib
    working_folders = []
    for file in os.listdir(in_path):
        if not os.path.isdir(in_path+"/"+file):
            continue
        if len(file.split("sort"))==1:
            continue
        if len(file.split("pixellib"))!=1:
            continue
        pix_path = "pixellib".join(file.split("sort"))
        if os.path.exists(in_path+"/"+pix_path):
            continue
        working_folders.append(file)
    print("Annotating Folders:")
    print(working_folders)
    for folder in working_folders:
        pix_path = in_path+"/"+"pixellib".join(folder.split("sort"))
        yolo_path = in_path+"/"+"yolo".join(folder.split("sort"))
        sort_path = in_path+"/"+folder
        avi_path = in_path+"/"+"".join(folder.split("_sort"))
        try:
            os.mkdir(pix_path)
        except:
            continue
        print(pix_path)
        for file in os.listdir(avi_path):
            file2 = file.replace(".avi",".csv")
            video = avi_path+"/"+file
            csv = sort_path+"/"+file2
            out_path = pix_path+"/"+file
            print("Attempting: ",video, csv)
            os.system("python3 yolo_to_pixellib.py "+video+" "+csv+" "+out_path)

    #Run Post processing
    working_folders = []
    for file in os.listdir(in_path):
        if not os.path.isdir(in_path+"/"+file):
            continue
        if len(file.split("pixellib"))==1:
            continue
        if len(file.split("post"))!=1:
            continue
        post_path = "post".join(file.split("pixellib"))
        if os.path.exists(in_path+"/"+post_path):
            continue
        working_folders.append(file)
    print("Processing Folders")
    for folder in working_folders:
        pix_path = in_path+"/"+folder
        post_path = in_path+"/"+"post".join(folder.split("pixellib"))
        try:
            os.mkdir(post_path)
        except:
            continue
        print(post_path)
        for worm_folder in os.listdir(pix_path):
            
            fold_path = pix_path + "/" + worm_folder
            out_path = post_path+"/"+worm_folder+".csv"
            os.system("python3 postproc_summary.py "+fold_path+" "+out_path)


