import os
import shutil
import glob
import config

# script to move files after checking them manually and removing the outlier data

# check that the directory to processed UAH PREVENTION files to be is correct, I used ROI 2 for manually checking files.
# all qualified videos in "unprocessed" folder had been moved to a folder "passed" in their class folder
# i.e. "../datasets/LC clips/TTE 0/ROI 2/unprocessed/LLC/passed"

root_path = config.root_dir()
LC = True  # move LC clips
TTE, ROI = 0, 2  # choose TTE and ROI that was checked, this only works with ROIs: ROI, ROI+1, ROI+2


def move_files(processed_files, unprocessed_files, target):

    # find correct files and store them into a new folder
    for file in processed_files:
        clip = os.path.basename(file)[:-7]
        for file2 in unprocessed_files:
            name = os.path.basename(file2)
            if clip == name[:-7]:
                shutil.move(file2, target + '/' + name)
                break  # move to next file instead of looping through the remainder of the list


if __name__ == '__main__':
    if LC:  # lane change class
        if TTE == 0:  # recognition videos
            # -------------- ROI 2 -------------
            path_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/unprocessed/LLC/passed'
            files_LLC = glob.glob(path_LLC + '/*.mp4')

            path_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/unprocessed/RLC/passed'
            files_RLC = glob.glob(path_RLC + '/*.mp4')

            # target directories for ROI 2
            target_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/processed/LLC'
            if not os.path.isdir(target_LLC):
                os.makedirs(target_LLC)

            target_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/processed/RLC'
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
            path_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/unprocessed/LLC'
            files_LLC2 = glob.glob(path_LLC + '/*.mp4')

            path_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/unprocessed/RLC'
            files_RLC2 = glob.glob(path_RLC + '/*.mp4')

            # target directories for ROI 3
            target_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/processed/LLC'
            if not os.path.isdir(target_LLC):
                os.makedirs(target_LLC)

            target_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/processed/RLC'
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
            path_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/unprocessed/LLC'
            files_LLC3 = glob.glob(path_LLC + '/*.mp4')

            path_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/unprocessed/RLC'
            files_RLC3 = glob.glob(path_RLC + '/*.mp4')

            # target directories for ROI 4
            target_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/processed/LLC'
            if not os.path.isdir(target_LLC):
                os.makedirs(target_LLC)

            target_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {ROI}/processed/RLC'
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

        else:  # prediction videos based on recognition clips, make sure recognition dataset is ready first

            # -------------- ROI 2 -------------
            path_LLC = f'{root_path}/datasets/LC clips/Recognition/ROI {2}/processed/LLC'
            files_LLC = glob.glob(path_LLC + '/*.mp4')

            path_RLC = f'{root_path}/datasets/LC clips/Recognition/ROI {2}/processed/RLC'
            files_RLC = glob.glob(path_RLC + '/*.mp4')

            # target directories for ROI 2
            target_LLC = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/processed/LLC'
            if not os.path.isdir(target_LLC):
                os.makedirs(target_LLC)

            target_RLC = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/processed/RLC'
            if not os.path.isdir(target_RLC):
                os.makedirs(target_RLC)

            # unprocessed prediction lane change clips
            path_LLC_move = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/unprocessed/LLC'
            files_LLC2 = glob.glob(path_LLC_move + '/*.mp4')

            path_RLC_move = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/unprocessed/RLC'
            files_RLC2 = glob.glob(path_RLC_move + '/*.mp4')

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

            # -------------- ROI 3 -------------
            ROI = ROI + 1
            # unprocessed prediction lane change clips
            path_LLC_move = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/unprocessed/LLC'
            files_LLC3 = glob.glob(path_LLC_move + '/*.mp4')

            path_RLC_move = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/unprocessed/RLC'
            files_RLC3 = glob.glob(path_RLC_move + '/*.mp4')

            # target directories for ROI 3
            target_LLC = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/processed/LLC'
            if not os.path.isdir(target_LLC):
                os.makedirs(target_LLC)

            target_RLC = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/processed/RLC'
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

            # -------------- ROI 4 -------------
            ROI = ROI + 1
            # unprocessed prediction lane change clips
            path_LLC_move = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/unprocessed/LLC'
            files_LLC4 = glob.glob(path_LLC_move + '/*.mp4')

            path_RLC_move = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/unprocessed/RLC'
            files_RLC4 = glob.glob(path_RLC_move + '/*.mp4')

            # target directories for ROI 4
            target_LLC = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/processed/LLC'
            if not os.path.isdir(target_LLC):
                os.makedirs(target_LLC)

            target_RLC = f'{root_path}/datasets/LC clips/Prediction/ROI {ROI}/processed/RLC'
            if not os.path.isdir(target_RLC):
                os.makedirs(target_RLC)

            # find correct files and store them into a new folder
            for file in files_LLC:
                clip = os.path.basename(file)[:-7]
                for file2 in files_LLC4:
                    name = os.path.basename(file2)
                    if clip == name[:-7]:
                        shutil.move(file2, target_LLC + '/' + name)
                        break  # move to next file instead of looping through the remainder of the list

            for file in files_RLC:
                clip = os.path.basename(file)[:-7]
                for file2 in files_RLC4:
                    name = os.path.basename(file2)
                    if clip == name[:-7]:
                        shutil.move(file2, target_RLC + '/' + name)
                        break  # move to next file instead of looping through the remainder of the list

    else:  # no lane change class

        # -------------- ROI 2 -------------
        path_NLC = f'{root_path}/datasets/NLC clips/ROI {ROI}/unprocessed'
        files_NLC = glob.glob(path_NLC + '/*.mp4')

        target_NLC = f'{root_path}/datasets/NLC clips/ROI {ROI}/processed'
        if not os.path.isdir(target_NLC):
            os.makedirs(target_NLC)

        for file in files_NLC:
            name = os.path.basename(file)
            shutil.move(file, target_NLC + '/' + name)

        # -------------- ROI 3 -------------
        ROI = ROI+1
        path_NLC = f'{root_path}/datasets/NLC clips/ROI {ROI}/unprocessed'
        files_NLC2 = glob.glob(path_NLC + '/*.mp4')

        # target directories for ROI 3
        target_NLC = f'{root_path}/datasets/NLC clips/ROI {ROI}/processed'
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
        path_NLC = f'{root_path}/datasets/NLC clips/ROI {ROI}/unprocessed'
        files_NLC3 = glob.glob(path_NLC + '/*.mp4')

        # target directories for ROI 4
        target_NLC = f'{root_path}/datasets/NLC clips/ROI {ROI}/processed'
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
