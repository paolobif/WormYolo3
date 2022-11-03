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
		writer = cv2.VideoWriter("Pothole testing 2.mp4", fourcc, fps, (w, h))
	
		while ret:
		    writer.write(frame)
		    ret,	 frame = cap.read()
	
		writer.release()
		cap.release()
	except:
		os.remove(sys.argv[1]+"/"+vid)
