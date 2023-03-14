import cv2
import numpy as np
import os
import imutils


def horizontal_flip(video):
    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)
    print('frames', frame_count)
    ret, frame = src.read()
    if not ret:
        print(f'Failed to perform horizontal flipping')
        return frames
    frames.append(np.flip(frame, 1))

    for i in range(1, int(frame_count)):
        ret, frame = src.read()
        if ret:
            frame = np.flip(frame, 1)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {i} for flipping of in {os.path.basename(video)}')
    src.release()
    return frames


def random_rotate(video, degree_bound):
    ub = np.abs(degree_bound)
    lb = -np.abs(degree_bound)
    angle = np.random.uniform(low=lb, high=ub)
    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)
    print('frames', frame_count)
    ret, frame = src.read()
    if not ret:
        print(f'Failed to perform horizontal flipping')
        return frames
    frames.append(imutils.rotate(frame, angle))

    for i in range(1, int(frame_count)):
        ret, frame = src.read()
        if ret:
            frame = imutils.rotate(frame, angle)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {i} for flipping of in {os.path.basename(video)}')
    src.release()
    return frames


def save_video(path, name, video, size=(224, 224)):
    """
    Saves the video locally to specified path

    :param path: file path to store the video
    :param name: video name
    :param video: list of frames to be saved as a video
    :param size: output size of the video
    """
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # note the lower case
    v_name = name + '.mp4'
    v_out = cv2.VideoWriter()
    ret = v_out.open(path + '/' + v_name, fourcc, fps, size, True)
    for frame in video:
        v_out.write(frame)
    v_out.release()
    cv2.destroyAllWindows()
    return


v = './LC clips/TTE 0/ROI 3/processed/LLC/2042-1908_record4_drive3_x3.mp4'
new_v = random_rotate(v, 30)
save_video('..', 'test flip2', new_v)

