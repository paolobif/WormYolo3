import os
import sys

import cv2
import csv
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_bbs(vid_path, bb_path, death_path, out_path, update_freq):
    """
    Draw the bounding boxes defined in csv_path on the video defined in vid_path
    and output in out_path
    :param vid_path: the video
    :param bb_path: the bbs
    :param death_path: the death spots
    :param out_path: the output videos
    :param update_freq: per how many frames should we update graph
    :return: nothing
    """
    print("Processing " + vid_path)

    # Get video data through OpenCV
    vid = cv2.VideoCapture(vid_path)
    total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    # Get bb data in df
    df_bb = pd.read_csv(bb_path, names=('frame', 'x1', 'y1', 'w', 'h', 'label'))

    # Get death data in df
    df_death = pd.read_csv(death_path)
    total_alive = len(df_death) + 1
    # Set output path
    out_video_path = f"{out_path}/{os.path.basename(vid_path).strip('.avi')}_yolo_1.avi"

    # Set up writer object
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = float(vid.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height), True)

    # Matplotlib stuff
    PLT_DPI = 96.0      # Resolution of plot
    SCALE_DOWN = 5      # Scale of plot
    fig_width = width / (PLT_DPI * SCALE_DOWN)
    fig_height = height / (PLT_DPI * SCALE_DOWN)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=PLT_DPI)
    x1 = np.array([0.0])
    y1 = np.ones(x1.shape, dtype=float)
    plt.xlim((0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) + 20))
    plt.ylim((0.0, 1.2))
    plt.ylabel("Paralysis %")
    plt.xlabel("Frame")
    line1, = plt.plot(x1, y1, 'k-')  # so that we can update data later

    # Setup graph image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    df_death2=df_death
    #print(df_death2)
    #df_death2.columns = ['frame', 'x1', 'y1', 'x2', 'y2','label']
    df_death2 = df_death2.apply(pd.to_numeric)
    df_death2 = df_death2.apply(np.int64)
    while vid.isOpened():
        # Tuple returned, ret is a bool and frame is a frame
        ret, frame = vid.read()
        if not ret:
            print("Done")
            break

        frame_count = int(vid.get(cv2.CAP_PROP_POS_FRAMES))

        # Write bbs
        cur_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
        frame_bbs = df_bb.loc[df_bb['frame'] == cur_frame]
        for index, row in frame_bbs.iterrows():
            cv2.rectangle(frame, (row['x1'], row['y1']), (row['x1'] + row['w'], row['y1'] + row['h']), (255, 255, 0), 2)
        
        frame_dbbs = df_death2.loc[df_death2['frame'] <= cur_frame]
        for index, row in frame_dbbs.iterrows():
            cv2.rectangle(frame, (row['x1A'], row['y1A']), (row['x2A'], row['y2A']), (0, 255, 255), 2)

        if frame_count % update_freq == 0 or frame_count==(total_frame_count-1):
            # Update data

            # Calculate new paralysis %
            paralysed = 0
            for row in df_death[1:]['frame']:
                if float(row) <= frame_count:
                    paralysed += 1
            cur_paralysis = (total_alive - paralysed) / total_alive

            # Update plot
            y1 = np.append(y1, cur_paralysis)
            x1 = np.append(x1, frame_count)
            line1.set_xdata(x1)
            line1.set_ydata(y1)

            # Redraw the canvas
            fig.canvas.draw()

            # Convert canvas to image
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw on frame
        frame[0:int(fig_height * PLT_DPI), 0:int(fig_width * PLT_DPI)] = img

        # Write frame
        writer.write(frame)

    # Resource cleanup
    vid.release()
    writer.release()


if __name__ == "__main__":
    """
    Overlays the video with a graph in lower left corner (#TODO customise) with
    death rate of worms according to provided csv file
    Usage:
        python graph_overlay.py BB_CSV_FOLD_PATH DEATHS_CSV_FOLD_PATH VID_FOLD_PATH OUTPUT_FOLD_PATH
    where
        BB_CSV_FOLD_PATH: folder containing bounding box csvs for locations
        DEATHS_CSV_FOLD_PATH: folder containing csvs for death frames
        VID_FOLD_PATH: folder containing videos to annotate
        OUTPUT_FOLD_PATH: folder for output
    """

    BB_CSV_FOLD_PATH = sys.argv[1]
    DEATHS_CSV_FOLD_PATH = sys.argv[2]
    VID_FOLD_PATH = sys.argv[3]
    OUTPUT_FOLD_PATH = sys.argv[4]

    bbs_list = os.listdir(BB_CSV_FOLD_PATH)
    print("BBs: \n" + str(bbs_list))

    deaths_list = os.listdir(DEATHS_CSV_FOLD_PATH)
    print("Deaths: \n" + str(deaths_list))

    vid_list = os.listdir(VID_FOLD_PATH)
    print("Vids: \n" + str(vid_list))

    # Sanity check
    assert len(vid_list) == len(bbs_list)
    assert len(bbs_list) == len(deaths_list)

    for _ in tqdm.tqdm(range(len(bbs_list)), ncols=100):

        VID_PATH = os.path.join(VID_FOLD_PATH, vid_list[_])
        DEATH_PATH = os.path.join(DEATHS_CSV_FOLD_PATH, deaths_list[_])
        BB_PATH = os.path.join(BB_CSV_FOLD_PATH, bbs_list[_])
        OUT_PATH = OUTPUT_FOLD_PATH
        UPDATE_FREQ = 50

        draw_bbs(VID_PATH, BB_PATH, DEATH_PATH, OUT_PATH, 50)
