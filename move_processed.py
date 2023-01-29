import os
import shutil
import glob

# directory to processed UAH PREVENTION files to be moved, I used ROI 2 for manually checking files.
# all qualified videos in "unprocessed" folder had been moved to a folder "passed" in their class folder
# i.e. "./datasets/LC clips/TTE 0/ROI 2/unprocessed/LLC/passed"

LC = True  # move LC clips
TTE, ROI = 0, 2  # choose TTE and ROI that was checked, this only works with ROIs: ROI, ROI+1, ROI+2

if LC:  # lane change class
    # -------------- ROI 2 -------------
    path_LLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed/LLC/passed'
    files_LLC = glob.glob(path_LLC + '/*.mp4')

    path_RLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed/RLC/passed'
    files_RLC = glob.glob(path_RLC + '/*.mp4')

    # target directories for ROI 2
    target_LLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed/LLC'
    if not os.path.isdir(target_LLC):
        os.makedirs(target_LLC)

    target_RLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed/RLC'
    if not os.path.isdir(target_RLC):
        os.makedirs(target_RLC)

    # move files to processed files
    for file in files_LLC:
        name = os.path.basename(file)
        shutil.move(file, target_LLC + '/' + name)

    for file in files_RLC:
        name = os.path.basename(file)
        shutil.move(file, target_RLC + '/' + name)

    # -------------- ROI 3 -------------
    ROI = ROI+1
    path_LLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed/LLC'
    files_LLC2 = glob.glob(path_LLC + '/*.mp4')

    path_RLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed/RLC'
    files_RLC2 = glob.glob(path_RLC + '/*.mp4')

    # target directories for ROI 3
    target_LLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed/LLC'
    if not os.path.isdir(target_LLC):
        os.makedirs(target_LLC)

    target_RLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed/RLC'
    if not os.path.isdir(target_RLC):
        os.makedirs(target_RLC)

    # find correct files and store them into a new folder
    for file in files_LLC:
        clip = os.path.basename(file)[:-7]
        for file2 in files_LLC2:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target_LLC + '/' + name)
                break  # move to next file instead of looping through the remainder of the list

    for file in files_RLC:
        clip = os.path.basename(file)[:-7]
        for file2 in files_RLC2:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target_RLC + '/' + name)
                break  # move to next file instead of looping through the remainder of the list

    # -------------- ROI 4 -------------
    ROI = ROI+1
    path_LLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed/LLC'
    files_LLC3 = glob.glob(path_LLC + '/*.mp4')

    path_RLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed/RLC'
    files_RLC3 = glob.glob(path_RLC + '/*.mp4')

    # target directories for ROI 4
    target_LLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed/LLC'
    if not os.path.isdir(target_LLC):
        os.makedirs(target_LLC)

    target_RLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed/RLC'
    if not os.path.isdir(target_RLC):
        os.makedirs(target_RLC)

    # find correct files and store them into a new folder
    for file in files_LLC:
        clip = os.path.basename(file)[:-7]
        for file2 in files_LLC3:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target_LLC + '/' + name)
                break  # move to next file instead of looping through the remainder of the list

    for file in files_RLC:
        clip = os.path.basename(file)[:-7]
        for file2 in files_RLC3:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target_RLC + '/' + name)
                break  # move to next file instead of looping through the remainder of the list

else:  # no lane change class
    # -------------- ROI 2 -------------
    path_NLC = f'./datasets/NLC clips/TTE {TTE}/ROI {ROI}/unprocessed/passed'
    files_NLC = glob.glob(path_NLC + '/*.mp4')

    target_NLC = f'./datasets/NLC clips/TTE {TTE}/ROI {ROI}/processed'
    if not os.path.isdir(target_NLC):
        os.makedirs(target_NLC)

    for file in files_NLC:
        name = os.path.basename(file)
        shutil.move(file, target_NLC + '/' + name)

    # -------------- ROI 3 -------------
    ROI = ROI+1
    path_NLC = f'./datasets/NLC clips/TTE {TTE}/ROI {ROI}/unprocessed/passed'
    files_NLC2 = glob.glob(path_NLC + '/*.mp4')

    # target directories for ROI 3
    target_NLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed'
    if not os.path.isdir(target_NLC):
        os.makedirs(target_NLC)

    # find correct files and store them into a new folder
    for file in files_NLC:
        clip = os.path.basename(file)[:-7]
        for file2 in files_NLC2:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target_NLC + '/' + name)
                break  # move to next file instead of looping through the remainder of the list

    # -------------- ROI 4 -------------
    ROI = ROI+1
    path_NLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/unprocessed'
    files_NLC3 = glob.glob(path_NLC + '/*.mp4')

    # target directories for ROI 4
    target_NLC = f'./datasets/LC clips/TTE {TTE}/ROI {ROI}/processed'
    if not os.path.isdir(target_NLC):
        os.makedirs(target_NLC)

    # find correct files and store them into a new folder
    for file in files_NLC:
        clip = os.path.basename(file)[:-7]
        for file2 in files_NLC3:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target_NLC + '/' + name)
                break  # move to next file instead of looping through the remainder of the list

print('Manually checked files have been moved to processed files')
