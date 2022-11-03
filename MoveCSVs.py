import os
import shutil
src_dir = "/media/mlcomp/judybackup5-2-22"
dst_dir = "/media/mlcomp/judybackup5-2-22/descriptions/"
print("start")
for root, dirs, files in os.walk(src_dir):
	for f in files:		
		if f.endswith('description.txt'):
		    #print(f)
		    filename = (os.path.split(root)[-1])
		    nuname = f"{filename}_description.txt"
		    
		    new_name = os.path.join(dst_dir,nuname)
		    print(new_name)
		    print(os.path.join(root,f))
		    shutil.copy(os.path.join(root,f), os.path.join(dst_dir,nuname))
