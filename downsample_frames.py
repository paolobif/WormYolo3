
import cv2

def downsample_vid(in_path, out_path, down_count = 30):
    cap = cv2.VideoCapture(in_path)
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frameTime = 100

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 1000 / frameTime
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    while ret:
        if frame_count%30==0:
                writer.write(frame)
        ret, frame = cap.read()
        frame_count+=1

    writer.release()
    cap.release()

