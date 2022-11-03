import sys
import os
import cv2

print(os.listdir(sys.argv[1]))

for vid in os.listdir(sys.argv[1]):
	try:
		cap = cv2.VideoCapture(sys.argv[1]+"/"+vid)
		ret, frame = cap.read()
		print(sys.argv[1]+"/"+vid)
		h, w, _ = frame.shape
		frameTime = 100
	
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		fps = 1000 / frameTime
		writer = cv2.VideoWriter(sys.argv[2]+"/"+vid, fourcc, fps, (w, h))
	
		frame_count = 0
		while ret:
		    if frame_count%30==0:
			    writer.write(frame)
		    ret,	 frame = cap.read()
		    frame_count+=1
	
		writer.release()
		cap.release()
	except:
		pass
		#os.remove(sys.argv[1]+"/"+vid)
