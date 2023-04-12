import numpy as np
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Default plot configurations
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 150
sns.set()


def plot_training_path(data, model):
    # loss and accuracy plot
    fig, subplots = plt.subplots(1, 2, figsize=(14, 6))
    subplots[0].plot(data['epoch'], data['train_loss'], label='train')
    subplots[0].plot(data['epoch'], data['test_loss'], label='test')
    subplots[0].set_title(f'{model} Loss')
    subplots[0].set_ylabel('Loss')
    subplots[0].set_xlabel('Epoch')
    subplots[0].legend()

    subplots[1].plot(data['epoch'], data['train_acc'], label='train')
    subplots[1].plot(data['epoch'], data['test_acc'], label='test')
    subplots[1].set_title(f'{model} Accuracy')
    subplots[1].set_ylabel('Accuracy')
    subplots[1].set_xlabel('Epoch')
    subplots[1].legend()
    plt.show()
    return


data = pd.read_csv('training_results.txt')

plot_training_path(data, 'test 1')

from torchvision import transforms as t
from models.helper_functions import VideoFolderCustom

# setup transformer for the clips
transforms = [t.Resize((112, 112))]
transformer = t.Compose(transforms)

# setup videofolder to download the data
train_dir = './datasets/train/Recognition/ROI 2'
train_data = VideoFolderCustom(targ_dir=train_dir, transform=transformer,
                               permute=True, augmented=True)

test_dir = './datasets/test/Recognition/ROI 2'
test_data = VideoFolderCustom(targ_dir=test_dir, transform=transformer,
                              permute=True, augmented=True)

print('Training data:', len(train_data))
print('Test data:', len(test_data))

# desired_size = 224
# im_pth = "inp_TB&T.png"
#
# im = cv2.imread(im_pth)
# old_size = im.shape[:2]  # old_size is in (height, width) format
#
# ratio = float(desired_size)/max(old_size)
# new_size = tuple([int(x*ratio) for x in old_size])
# print(ratio)
#
# # new_size should be in (width, height) format
#
# im = cv2.resize(im, (new_size[1], new_size[0]))
#
# delta_w = desired_size - new_size[1]
# delta_h = desired_size - new_size[0]
# top, bottom = delta_h, 0
# left, right = delta_w//2, delta_w-(delta_w//2)
#
# color = [0, 0, 0]
# new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#
# cv2.imshow("image", new_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# a = np.array([[2,3,4,5], [2,3,4,5], [2,3,4,5]])
# b = np.array([[6,7,8,8], [6,7,8,8], [6,7,8,8]])
# c = np.stack((a,b), axis=0)
# print(c[0])
# print(c.shape)
# sys.exit()
# RECORD = 4
# DRIVE = 3

# if row['Object'] == 2581:
#     print(new_size)
#     print('delta h', delta_h, 'delta w', delta_w)

# vals= [1,24,5,6,2,55,2]
# i = 53
# i = np.where(np.array(vals) > i)[0]
# print(i)
# df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
# print(type(df['col1'].values))
# print(len(df.axes[0]))
# LCs = np.loadtxt('lane_changes.txt')  # -, ID_object, LC_type, start, event, end, blinker
# ID_LCs = LCs[:, 1].T
# print(ID_LCs)
# print(LCs[:, 1])
# print(np.where(LCs == 882)[0])
# sys.exit()

# video = f'./LC clips/ROI 3/RLC/2581-2310_record4_drive3_x3.mp4'
# print(video[:-7])
# cap = cv2.VideoCapture(video)
# frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print('NUM OF FRAMES', frameCount)
# frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# cap.release()
# sys.exit()
# buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
#
# fc = 0
# ret = True
#
# while (fc < frameCount  and ret):
#     ret, buf[fc] = cap.read()
#     fc += 1
#
# cap.release()
#
# cv2.namedWindow('frame 10')
# cv2.imshow('frame 10', buf[0])
# print(buf[0].shape)
# print(frameCount)

# import sys
#
#
# def root_dir() -> str:
#     return sys.path[1]
# print(root_dir())
# cv2.waitKey(0)
# cap = cv2.VideoCapture(video)
# ret, img = cap.read()
# cap.release()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# with open(detections) as f:
#     f.readline()
#     line = f.readline()
# f.close()
#
# data = line.split(' ')
# print(data)
# coor = data[3:]
# print(coor)
# x = [float(coor[i*2]) for i in range(int(len(coor)/2))]
# y = [float(coor[i*2+1]) for i in range(int(len(coor)/2))]
# for i, v in enumerate(x):
#     if v < 1:
#         del x[i]
#         del y[i]
#
# plt.imshow(img)
# # put a red dot, size 40, at 2 locations:
# plt.scatter(x=x, y=y, c='r')
#
# plt.show()