import cv2
import numpy as np
import pandas as pd


def read_lcs(file, video, obs_horizon, TTE):
    """Reads the lane_changes.txt file that stores all the information for a  detected lane change

    :param file: file_path: path to the lane change file
    :param video: caption: path to the video file
    :param obs_horizon: the amount of frames to be observed before a lane change event
    :param TTE: time to event, determines amount of frames before a lane change event
    :returns: all_frames, lane_changes
    """
    lcs = np.loadtxt(file)
    all_frames, lane_changes = [0, 0]
    # cap = cv2.VideoCapture(video)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # all_frames ={k: 0 for k in range(frame_count)}
    # lane_changes = {}
    # id_counter = 1
    # with open(file_path) as file:
    #     for line in file.readlines():
    #         line = line.strip('\n').split(' ')
    #         lc_class = int(float(line[2]))
    #         object = int(float(line[1]))
    #         start = int(float(line[3]))
    #         middle = int(float(line[4]))
    #         end = int(float(line[5]))
    #         blinker = int(float(line[6]))
    #         if start != -1:
    #             all_frames = fill_range_frames(all_frames ,obs_horizon ,tte ,start ,id_counter)
    #             lane_changes[id_counter] = create_a_lane(lc_class ,start ,middle ,end ,blinker ,object)
    #             id_counter += 1
    return all_frames, lane_changes


if __name__ == '__main__':
    test = np.loadtxt('lane_changes.txt')
    print(test)
    print('lol')
