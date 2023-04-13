import os
import glob
import pandas as pd
import config

columns = ['clip', 'class']
root_path = config.root_dir()


def LC_clip_store(ROI, data_type):
    """
    Generate clip store csv for doing train test splits, and to keep track of the clips stored and processed.

    :param ROI: size of region of interest
    :param data_type: type of LC to be stored, Recognition or Prediction
    """
    name = f'{root_path}/datasets/LC clips/{data_type}/ROI {ROI}/clip_store.csv'

    path = f'{root_path}/datasets/LC clips/{data_type}/ROI {ROI}/processed/LLC'
    files = glob.glob(path + '/*.mp4')

    data = []
    for file in files:
        clip = 'RLC/'+os.path.basename(file)
        data.append([clip, 'LLC'])
    df = pd.DataFrame(data=data, columns=columns)
    if os.path.isfile(name):
        df.to_csv(name, mode='a', index=False, header=False)
    else:
        df.to_csv(name, index=False)

    path = f'{root_path}/datasets/LC clips/{data_type}/ROI {ROI}/processed/RLC'
    files = glob.glob(path+'/*.mp4')

    data = []
    for file in files:
        clip = 'RLC/'+os.path.basename(file)
        data.append([clip, 'RLC'])
    df = pd.DataFrame(data=data, columns=columns)

    if os.path.isfile(name):
        df.to_csv(name, mode='a', index=False, header=False)
    else:
        df.to_csv(name, index=False)
    return


def NLC_clip_store(ROI):
    """
    Generate clip store csv for doing train test splits, and to keep track of the clips stored and processed.

    :param ROI: size of region of interest
    """
    name = f'../datasets/NLC clips/ROI {ROI}/clip_store.csv'

    path = f'../datasets/NLC clips/ROI {ROI}/processed'
    files = glob.glob(path + '/*.mp4')

    data = []
    for file in files:
        clip = os.path.basename(file)
        data.append([clip, 'NLC'])

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(name, index=False)
    return


if __name__=='__main__':
    NLC_clip_store(3)
    NLC_clip_store(4)
