import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing.data_augmentation import *
import config
import random
import glob

root_path = config.root_dir()
SPLIT_DATA = False
AUGMENTATION = True


def copy_files(path, ROI, df, data_type: str):
    """
    Copies video clips from a dataset folder (LC clips and NLC clips) to another folder

    :param path: path to copy the files into
    :param ROI: Region of interest size
    :param df: dataframe containing the clips to be handled
    :param data_type: Recognition or Prediction
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    os.makedirs(path+f'/ROI {ROI}/NLC')
    os.makedirs(path + f'/ROI {ROI}/LLC')
    os.makedirs(path + f'/ROI {ROI}/RLC')

    for _, row in df.iterrows():
        if row['class'] == 'NLC':
            file = f"{root_path}/datasets/NLC clips/ROI {ROI}/processed/{row['clip']}"
            target = f"{path}/ROI {ROI}/NLC/{row['clip']}"
            shutil.copy(file, target)
        else:
            file = f"{root_path}/datasets/LC clips/{data_type}/ROI {ROI}/processed/{row['clip']}"
            target = f"{path}/ROI {ROI}/{row['clip']}"
            shutil.copy(file, target)
    return


def get_full_df(ROI, path_LC, path_NLC):
    """
    Generates a full dataframe that contains the information for all the clips in folder LC clips and NLC clips
    for a specified ROI.

    :param ROI: size of the region of interest
    :param path_LC: path to LC clips folder
    :param path_NLC: path NLC clips folder
    :return: a dataframe with the data details concatenated together
    """

    LC_clip_store = f'{path_LC}/ROI {ROI}/clip_store.csv'
    NLC_clip_store = f'{path_NLC}/ROI {ROI}/clip_store.csv'

    LC_data = pd.read_csv(LC_clip_store)
    NLC_data = pd.read_csv(NLC_clip_store)

    max_amount = LC_data['class'].value_counts().max()
    NLC_data = NLC_data.sample(n=max_amount, random_state=0)  # we want same split for all ROIs
    return pd.concat([LC_data, NLC_data])


def augmentation(path, LLC_clips, NLC_clips, RLC_clips, max_amount, aug_function, suffix):
    """
    Performs data augmentation on videos according to the input aug_function.

    :param LLC_clips: list of LLC clips
    :param NLC_clips: list of NLC clips
    :param RLC_clips: list of RLC clips
    :param max_amount: maximum amount of clips to be generated
    :param aug_function: the function used to perform augmentation
    :param suffix: the suffix used for renaming the videos
    """

    random_NLCs = random.sample(NLC_clips, max_amount)
    random_RLCs = random.sample(RLC_clips, max_amount)
    random_LLCs = random.sample(LLC_clips, max_amount)

    for (clip_llc, clip_nlc, clip_rlc) in zip(random_LLCs, random_NLCs, random_RLCs):
        # LLC
        name = os.path.basename(clip_llc).split('.')[0]
        new_clip = aug_function(clip_llc)
        save_video(f'{path}/LLC', f'{name}_{suffix}', new_clip)
        # NLC
        name = os.path.basename(clip_nlc).split('.')[0]
        new_clip = aug_function(clip_nlc)
        save_video(f'{path}/NLC', f'{name}_{suffix}', new_clip)
        # RLC
        name = os.path.basename(clip_rlc).split('.')[0]
        new_clip = aug_function(clip_rlc)
        save_video(f'{path}/RLC', f'{name}_{suffix}', new_clip)
    return


if __name__ == '__main__':

    # split the dataset to test-train?
    if SPLIT_DATA:
        main_lcs = f'{root_path}/datasets/LC clips/Recognition'
        data_type = 'Recognition'
        # main_lcs = f'{root_path}/datasets/LC clips/Prediction'
        # data_type = 'Prediction'
        nlcs = f'{root_path}/datasets/NLC clips'
        test = f'{root_path}/datasets/test'
        train = f'{root_path}/datasets/train'

        # ROI 2
        ROI = 2
        full_data = get_full_df(ROI=ROI, path_LC=main_lcs, path_NLC=nlcs)

        # 75, 25 split, random state = 35 produces a fairly equal split
        itrain, itest = train_test_split(range(full_data.shape[0]), random_state=35, test_size=0.25)

        X_train = full_data.iloc[itrain, :]
        X_test = full_data.iloc[itest, :]

        print('Training data size:', len(itrain))
        print('Test data size:', len(itest))

        print('Training data split:')
        print(X_train['class'].value_counts())
        print('Test data split:')
        print(X_test['class'].value_counts())

        copy_files(train + '/' + data_type, ROI, X_train, data_type)
        copy_files(test + '/' + data_type, ROI, X_test, data_type)

        # ROI 3
        ROI = 3
        full_data = get_full_df(ROI=ROI, path_LC=main_lcs, path_NLC=nlcs)

        # use same split
        X_train = full_data.iloc[itrain, :]
        X_test = full_data.iloc[itest, :]

        copy_files(train + '/' + data_type, ROI, X_train, data_type)
        copy_files(test + '/' + data_type, ROI, X_test, data_type)

        # ROI 4
        ROI = 4
        full_data = get_full_df(ROI=ROI, path_LC=main_lcs, path_NLC=nlcs)

        # use same split
        X_train = full_data.iloc[itrain, :]
        X_test = full_data.iloc[itest, :]

        copy_files(train + '/' + data_type, ROI, X_train, data_type)
        copy_files(test + '/' + data_type, ROI, X_test, data_type)

    # perform data augmentation on training set with horizontal flip, jitter, gaussian noise, random rotation
    # and adjusting brightness? the data will be balanced. Augmentation done for one ROI at a time
    if AUGMENTATION:
        # -- RECOGNITION --
        ROI = 2
        train = f'{root_path}/datasets/train/Recognition/ROI {ROI}'
        # train = f'{root_path}/datasets/train/Prediction/ROI {ROI}'

        LLC_clips = glob.glob(f"{train}/LLC/*.mp4")
        RLC_clips = glob.glob(f"{train}/RLC/*.mp4")
        NLC_clips = glob.glob(f"{train}/NLC/*.mp4")

        max_amount = min([len(LLC_clips), len(RLC_clips), len(NLC_clips)])  # length of LLC

        # horizontal flips
        for clip in LLC_clips:
            name = os.path.basename(clip).split('.')[0]
            new_clip = horizontal_flip(clip)
            save_video(f'{train}/RLC', f'{name}_flipped', new_clip)

        for clip in RLC_clips:
            name = os.path.basename(clip).split('.')[0]
            new_clip = horizontal_flip(clip)
            save_video(f'{train}/LLC', f'{name}_flipped', new_clip)

        random_NLCs = random.sample(NLC_clips, max_amount)
        for clip in random_NLCs:
            name = os.path.basename(clip).split('.')[0]
            new_clip = horizontal_flip(clip)
            save_video(f'{train}/NLC', f'{name}_flipped', new_clip)

        # gaussian noise
        noise = lambda vid: random_noise(vid)
        augmentation(train, LLC_clips, NLC_clips, RLC_clips, max_amount, noise, 'noise')

        # jitter
        jitter = lambda vid: apply_jitter(vid)
        augmentation(train, LLC_clips, NLC_clips, RLC_clips, max_amount, jitter, 'jitter')

        # adjusted brightness
        brightness = lambda vid: random_brightness(vid)
        augmentation(train, LLC_clips, NLC_clips, RLC_clips, max_amount, brightness, 'brightness')

        # random rotate
        rotate = lambda vid: random_rotate(vid)
        augmentation(train, LLC_clips, NLC_clips, RLC_clips, max_amount, rotate, 'rotate')

        # random augment
        random_aug = lambda vid: random_augment(vid)
        augmentation(train, LLC_clips, NLC_clips, RLC_clips, 95, random_aug, 'random')  # 95 to get to 1700 clips per class

        print('Data augmentation done for ROI', ROI)
