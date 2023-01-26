import cv2
import numpy as np
import pandas as pd
import os

RECORD = 4
DRIVE = 3
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
        print(len(data))
        for ROI in ROIs:
            imgs = get_ROI_frames(data, ROI, src)
            # fill the video to have obs_horizon amount of frames
            while len(imgs) < obs_horizon:
                imgs.insert(0, imgs[0])

            if LC[2] == 3:
                path = f'./LC clips/ROI {ROI}/LLC'
            else:
                path = f'./LC clips/ROI {ROI}/RLC'
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
    :return: Dataframe with columns [Frame, Object, xy-coordinates]
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
            # remove obvious outliers
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
    :return: Dataframe with columns [Frame, Object, xy-coordinates]
    """
    df = detections[detections['Object'].values == ID]
    start = point_event - obs_horizon
    if start < 0:
        start = 0
    end = point_event
    return df[df['Frame'].between(start+1, end)]


def get_ROI_frames(data, ROI, src):
    frames = []
    src.set(cv2.CAP_PROP_POS_FRAMES, data['Frame'].values[0]-1)
    ret, frame = src.read()
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
    w = row['x max'] - row['x min']
    h = row['y max'] - row['y min']
    # center = (h // 2, w // 2)  # middle of vehicle
    left = row['x min'] - (ROI * w) / 2
    right = row['x max'] + (ROI * w) / 2
    top = row['y min'] - (ROI * h) / 2
    bottom = row['y max'] + (ROI * h) / 2

    # Check for bounds
    if left < 0:
        left = 0
    if right > img.shape[1]:
        right = img.shape[1]
    if top < 0:
        top = 0
    if bottom > img.shape[0]:
        bottom = img.shape[0]

    new_img = img[int(np.round(top)):int(np.round(bottom)),
                  int(np.round(left)):int(np.round(right))]

    old_size = new_img.shape[:2]  # old_size is in (height, width) format
    ratio = float(OUTPUT) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(new_img, (new_size[1], new_size[0]))

    # zero pad the vehicle to middle of the frame
    delta_w = OUTPUT - new_size[1]
    delta_h = OUTPUT - new_size[0]
    if delta_h > 112:
        top, bottom = delta_h, 0
    else:
        top, bottom = delta_h//2, delta_h//2
    left, right = delta_w//2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


def save_video(path, name, video, size=(OUTPUT, OUTPUT)):
    """
    Saves the video locally
    """
    fps = 10
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
    v_out = cv2.VideoWriter()
    v_name = name + '.mp4'
    success = v_out.open(path + '/' + v_name, fourcc, fps, size, True)
    for frame in video:
        v_out.write(frame)
    v_out.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    TTE = 0  # choose TTE
    ROIs = [2, 3, 4]  # choose ROIs
    obs_horizon = 40  # choose observation horizon

    LC_file = f'./UAH PREVENTION/RECORD{RECORD}/DRIVE{DRIVE}/processed_data/detection_camera1/lane_changes.txt'
    detections = f'./UAH PREVENTION/RECORD{RECORD}/DRIVE{DRIVE}/processed_data/detection_camera1/detections_tracked.txt'
    video_file = f'./UAH PREVENTION/RECORD{RECORD}/DRIVE{DRIVE}/video_camera1.mp4'

    read_lcs(LC_file, video_file, detections, obs_horizon, TTE, ROIs)
    print(f'DONE for record {RECORD} drive {DRIVE}')
