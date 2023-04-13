import cv2
import numpy as np
import os
import imutils
from numpy.fft import fft2, ifft2
from PIL import ImageEnhance, Image
from torchvision import transforms
import torch
import random


# this file is based on:
# https://towardsdatascience.com/augmenting-images-for-deep-learning-3f1ea92a891c
def horizontal_flip(video):
    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)

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


def random_rotate(video, degree_bound=25):
    ub = np.abs(degree_bound)
    lb = -np.abs(degree_bound)
    angle = np.random.uniform(low=lb, high=ub)

    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, frame = src.read()
    if not ret:
        print(f'Failed to perform random rotation')
        return frames
    frames.append(imutils.rotate(frame, angle))

    for i in range(1, int(frame_count)):
        ret, frame = src.read()
        if ret:
            frame = imutils.rotate(frame, angle)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {i} for rotation of in {os.path.basename(video)}')
    src.release()
    return frames


def random_brightness(video, lower_bound=0.45, upper_bound=0.6):
    ub = np.abs(upper_bound)
    lb = -np.abs(lower_bound)
    amount = 1 + np.random.uniform(low=lb, high=ub)

    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, frame = src.read()
    if not ret:
        print(f'Failed to perform random brightness')
        return frames
    frames.append(adjust_brightness(frame, amount))

    for i in range(1, int(frame_count)):
        ret, frame = src.read()
        if ret:
            frame = adjust_brightness(frame, amount)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {i} for random brightness of in {os.path.basename(video)}')
    src.release()
    return frames


def random_noise(video):
    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, frame = src.read()
    if not ret:
        print(f'Failed to perform noise')
        return frames
    frames.append((gaussian_noise(frame) * 255).astype(np.uint8))

    for i in range(1, int(frame_count)):
        ret, frame = src.read()
        if ret:
            frame = gaussian_noise(frame)
            frames.append((frame * 255).astype(np.uint8))
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {i} for noise of in {os.path.basename(video)}')
    src.release()
    return frames


def apply_jitter(video):
    seed = np.random.randint(2147483647)
    torch.manual_seed(seed)

    frames = []
    src = cv2.VideoCapture(video)
    frame_count = src.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, frame = src.read()
    if not ret:
        print(f'Failed to perform jitter')
        return frames
    frames.append(jitter(frame))

    for i in range(1, int(frame_count)):
        ret, frame = src.read()
        if ret:
            torch.manual_seed(seed)  # keep same jitter for all frames sin the video
            frame = jitter(frame)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[0]))
            print(f'Failed to add frame {i} for jitter of in {os.path.basename(video)}')
    src.release()
    return frames


def random_augment(video):
    # gaussian noise
    noise = lambda vid: random_noise(vid)

    # jitter
    jitter = lambda vid: apply_jitter(vid)

    # adjusted brightness
    brightness = lambda vid: random_brightness(vid)

    # random rotate
    rotate = lambda vid: random_rotate(vid)

    # make a random selection, rotated videos are less likely to appear
    action = random.choices([noise, jitter, brightness, rotate], weights=(27.5, 27.5, 27.5, 17.5), k=1)[0]
    return action(video)


def gaussian_noise(img):
    u = 0.005
    sd = 0.03
    gaussian = np.random.normal(u, sd, img.shape)
    F = fft2(img / 255)
    N = fft2(gaussian)
    G = F + N
    return np.abs(ifft2(G))


def adjust_brightness(img, amount):
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    filter = ImageEnhance.Brightness(im_pil)
    im_pil = filter.enhance(amount)

    # For reversing the operation:
    im_np = np.array(im_pil)
    im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
    return im_np


def jitter(img, b=0.2, c=0.25, s=0.25, h=0.1):
    """
    Randomly alter brightness, contrast, saturation, hue within given range
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transform = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

    # apply transform
    img = transform(img)
    # For reversing the operation:
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


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

# v = '../datasets/train/Recognition/ROI 3/NLC/53-39_record5_drive3_x3.mp4'
# new_v = horizontal_flip(v)
# save_video('..', 'test random', new_v)
