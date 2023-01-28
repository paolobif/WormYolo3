import cv2
import numpy as np
import pandas as pd

def makeVideo(orig_vid, bb_csv, tod_csv, out_video_path):
  vid = cv2.VideoCapture(orig_vid)
  df = pd.read_csv(bb_csv, usecols=[0, 1, 2, 3, 4],
                              names=["frame", "x1", "y1", "width", "height"])
  outputs = pd.read_csv(tod_csv)
  outputs = np.asarray(outputs)
  print(outputs)
  writer = None
  ret = True

  while (ret):
          ret, frame = vid.read()
          frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
          if frame_count == 1:
              height, width, channels = frame.shape
              #print(height, width)
              fourcc = cv2.VideoWriter_fourcc(*"MJPG")
              writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
          filtval = df['frame'] == int(frame_count)
          interim = df[filtval]
          interim = np.asarray(interim)
          #print(interim)
          for rows in interim:
              frameNA, x1, y1, w, h, *_ = rows
              x2=x1+w
              y2=y1+h
              frameNA = int(float(frameNA))
              #if frame_count > frameNA:
              x1=int(float(x1))
              x2=int(float(x2))
              y1=int(float(y1))
              y2=int(float(y2))
              cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
          for rows in outputs:
              frameNA, x1, y1, x2, y2,label,*_ = rows
              frameNA = int(float(frameNA))
              if frame_count > frameNA:
                  x1=int(float(x1))
                  x2=int(float(x2))
                  y1=int(float(y1))
                  y2=int(float(y2))
                  label=int(float(label))
                  cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)
                  cv2.putText(frame, str(label), (x1-5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
          writer.write(frame)
  writer.release()












