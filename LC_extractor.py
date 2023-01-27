import cv2
import numpy as np
import pandas as pd
import os
import time

OUTPUT = 224  # output image size for videos


def read_lcs(file, video, detections, obs_horizon, TTE, ROIs):
    """
    Reads the lane_changes.txt file that stores all the information for a  detected lane change

    :param file: file_path: path to the lane change file
    :param video: caption: path to the video file
    :param obs_horizon: the amount of frames to be observed before a lane change event
    :param TTE: time to event, determines amount of frames before a lane change event
    """
    LCs = np.loadtxt(file)  # -, ID_object, LC_type, start, event, end, blinker
    tracker = detections_table(detections)

    src = cv2.VideoCapture(video)
    for LC in LCs:
        data = frames_in_horizon(tracker, LC[1], obs_horizon, LC[4]-TTE)
        if data.empty:  # no data could be found in detections tracker?
            continue
        for ROI in ROIs:
            imgs = get_ROI_frames(data, ROI, src)
            if len(imgs) == 0:  # failed to read the lane change?
                break
            # fill the video to have obs_horizon amount of frames
            while len(imgs) < obs_horizon:
                imgs.insert(0, imgs[0])

            if LC[2] == 3:
                path = f'./LC clips/TTE {TTE}/ROI {ROI}/unprocessed2/LLC'
            else:
                path = f'./LC clips/TTE {TTE}/ROI {ROI}/unprocessed2/RLC'
            # create folder if it does not exist
            if not os.path.isdir(path):
                os.makedirs(path)
            save_video(path, f'{int(LC[1])}-{int(LC[4])}_record{RECORD}_drive{DRIVE}_x{ROI}', imgs)
    src.release()
    return


def detections_table(file):
    """
    Creates a dataframe that contains all the needed information to track vehicles in a frame

    :param file: path to the detections file
    :returns: Dataframe with columns [Frame, Object, xy-coordinates]
    """
    data = np.zeros(6)
    with open(file) as f:
        for line in f.readlines():
            line = line.split(' ')
            if int(line[2]) < 3:  # not a vehicle?
                continue
            frame = int(line[0])
            ID = abs(int(line[1]))
            coords = line[3:]
            xs = [float(coords[i * 2]) for i in range(int(len(coords) / 2))]
            ys = [float(coords[i * 2 + 1]) for i in range(int(len(coords) / 2))]
            # remove obvious outliers, sometimes the detections_tracked.txt files have noisy points
            for i, (v, v2) in enumerate(zip(xs, ys)):
                if v < 1 or v > 1919 or v2 < 1 or v2 > 599:  # width of a frame is 1920, height 600
                    del xs[i]
                    del ys[i]
            data = np.vstack((data, [frame, ID, np.min(xs), np.max(xs), np.min(ys), np.max(ys)]))
    f.close()
    return pd.DataFrame({'Frame': data[1:, 0], 'Object': data[1:, 1], 'x min': data[1:, 2], 'x max': data[1:, 3],
                         'y min': data[1:, 4], 'y max': data[1:, 5]})


def frames_in_horizon(detections, ID, obs_horizon, point_event):
    """
    Extracts the frames for an object ID inside the observation horizon

    :param detections: dataframe containing the tracker information
    :param ID: ID of an object
    :param obs_horizon: observation horizon (in frames)
    :param point_event: the frame where the event happens
    :returns: Dataframe with columns [Frame, Object, xy-coordinates]
    """
    df = detections[detections['Object'].values == ID]
    start = point_event - obs_horizon
    if start < 0:
        start = 0
    end = point_event
    return df[df['Frame'].between(start+1, end)]


def get_ROI_frames(data, ROI, src):
    """
    Extracts the frames with a specified ROI around an object

    :param data: tracker dataframe containing the information of an object
    :param ROI: size of the region of interest
    :param src: a VideoCapture object used to read frames from a video
    :returns: a list of frames
    """
    frames = []
    src.set(cv2.CAP_PROP_POS_FRAMES, data['Frame'].values[0]-1)
    ret, frame = src.read()
    if not ret:
        print(f'Failed to extract LC {data["Object"].values[0]}-{data["Frame"].values[-1]}')
        return frames
    frames.append(format_frames(frame, data.iloc[0], ROI))

    for i, (_, row) in enumerate(data.iterrows()):
        if i == 0:
            continue
        ret, frame = src.read()
        if ret:
            frame = format_frames(frame, row, ROI)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {row["Frame"]} ({i+1}) for {row["Object"]}')
    return frames


def format_frames(img, row, ROI):
    """
    Formats the input image frame based on output size and ROI

    :param img: image to be formatted
    :param row: row of a tracker dataframe containing object information
    :param ROI: size of the region of interest
    :returns: a formatted image
    """
    # compute width and height of the object, use that information to create ROI box
    w = row['x max'] - row['x min']
    h = row['y max'] - row['y min']
    left = row['x min'] - (ROI * w) / 2
    right = row['x max'] + (ROI * w) / 2
    top = row['y min'] - (ROI * h) / 2
    bottom = row['y max'] + (ROI * h) / 2

    # Check for bounds, correct if needed
    if left < 0:
        left = 0
    if right > img.shape[1]:
        right = img.shape[1]
    if top < 0:
        top = 0
    if bottom > img.shape[0]:
        bottom = img.shape[0]

    # extract ROI and resize object to size (OUTPUT, OUTPUT), the vehicle will be centered
    new_img = img[int(np.round(top)):int(np.round(bottom)),
                  int(np.round(left)):int(np.round(right))]

    # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    old_size = new_img.shape[:2]  # old_size is in (height, width) format
    ratio = float(OUTPUT) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    img2 = cv2.resize(new_img, (new_size[1], new_size[0]))

    # zero pad the vehicle to middle of the frame
    delta_w = OUTPUT - new_size[1]
    delta_h = OUTPUT - new_size[0]

    # if delta_h >= OUTPUT//2:
    #     top, bottom = delta_h//2, delta_h-(delta_h//2)
    # else:
    #     top, bottom = delta_h, 0
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)  # usually 0
    color = [0, 0, 0]  # black
    new_img = cv2.copyMakeBorder(img2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img


def save_video(path, name, video, size=(OUTPUT, OUTPUT)):
    """
    Saves the video locally to specified path

    :param path: file path to store the video
    :param name: video name
    :param video: list of frames to be saved as a video
    :param size: output size of the video
    """
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # note the lower case
    v_out = cv2.VideoWriter()
    v_name = name + '.mp4'
    success = v_out.open(path + '/' + v_name, fourcc, fps, size, True)
    for frame in video:
        v_out.write(frame)
    v_out.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    start = time.time()
    RECORD = 4  # choose RECORD
    DRIVE = 1  # choose DRIVE
    TTE = 0  # choose TTE
    ROIs = [2, 3, 4]  # choose ROIs
    obs_horizon = 40  # choose observation horizon

    LC_file = f'./UAH PREVENTION/RECORD{RECORD}/DRIVE{DRIVE}/processed_data/detection_camera1/lane_changes.txt'
    detections = f'./UAH PREVENTION/RECORD{RECORD}/DRIVE{DRIVE}/processed_data/detection_camera1/detections_tracked.txt'
    video_file = f'./UAH PREVENTION/RECORD{RECORD}/DRIVE{DRIVE}/video_camera1.mp4'

    read_lcs(LC_file, video_file, detections, obs_horizon, TTE, ROIs)
    print(f'LC EXTRACTION DONE for record {RECORD} drive {DRIVE}')
    print(f'--  took {time.time()-start} seconds --')
