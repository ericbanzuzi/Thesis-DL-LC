import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_DATA = True
AUGMENTATION = False


def copy_files(path, ROI, df, data_type: str):

    if not os.path.isdir(path):
        os.makedirs(path)

    os.makedirs(path+f'/ROI {ROI}/NLC')
    os.makedirs(path + f'/ROI {ROI}/LLC')
    os.makedirs(path + f'/ROI {ROI}/RLC')

    for _, row in df.iterrows():
        if row['class'] == 'NLC':
            file = f"./datasets/NLC clips/ROI {ROI}/processed/{row['clip']}"
            target = f"{path}/ROI {ROI}/NLC/{row['clip']}"
            shutil.copy(file, target)
        else:
            file = f"./datasets/LC clips/{data_type}/ROI {ROI}/processed/{row['clip']}"
            target = f"{path}/ROI {ROI}/{row['clip']}"
            shutil.copy(file, target)
    return


def get_full_df(ROI, path_LC, path_NLC):

    LC_clip_store = f'{path_LC}/ROI {ROI}/clip_store.csv'
    NLC_clip_store = f'{path_NLC}/ROI {ROI}/clip_store.csv'

    LC_data = pd.read_csv(LC_clip_store)
    NLC_data = pd.read_csv(NLC_clip_store)

    max_amount = LC_data['class'].value_counts().max()
    NLC_data = NLC_data.sample(n=max_amount)

    return pd.concat([LC_data, NLC_data])


if __name__ == '__main__':

    if SPLIT_DATA:
        recognition_lcs = './datasets/LC clips/Recognition'
        nlcs = './datasets/NLC clips'
        test = './datasets/test'
        train = './datasets/train'

        # ROI 2
        ROI = 2
        path_to_files = f'{recognition_lcs}/ROI {ROI}/processed/'
        full_data = get_full_df(ROI=ROI, path_LC=recognition_lcs, path_NLC=nlcs)

        # 80, 20 split, random state = 35 produces a fairly equal split
        itrain, itest = train_test_split(range(full_data.shape[0]), random_state=35, test_size=0.25)

        X_train = full_data.iloc[itrain, :]
        X_test = full_data.iloc[itest, :]

        print('Training data size:', len(itrain))
        print('Test data size:', len(itest))

        print('Training data split:\n', X_train['class'].value_counts())
        print('Test data split:\n', X_test['class'].value_counts())

        copy_files(train+'/Recognition', ROI, X_train, 'Recognition')
        copy_files(test+'/Recognition', ROI, X_test, 'Recognition')

        # ROI 3
        ROI = 3
        path_to_files = f'{recognition_lcs}/ROI {ROI}/processed/'
        full_data = get_full_df(ROI=ROI, path_LC=recognition_lcs, path_NLC=nlcs)

        # use same split
        X_train = full_data.iloc[itrain, :]
        X_test = full_data.iloc[itest, :]

        copy_files(train+'/Recognition', ROI, X_train, 'Recognition')
        copy_files(test+'/Recognition', ROI, X_test, 'Recognition')

        # ROI 4
        ROI = 4
        path_to_files = f'{recognition_lcs}/ROI {ROI}/processed/'
        full_data = get_full_df(ROI=ROI, path_LC=recognition_lcs, path_NLC=nlcs)

        # use same split
        X_train = full_data.iloc[itrain, :]
        X_test = full_data.iloc[itest, :]

        copy_files(train+'/Recognition', ROI, X_train, 'Recognition')
        copy_files(test+'/Recognition', ROI, X_test, 'Recognition')

