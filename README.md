# BSc Thesis: Deep Learning - Lane Change Recognition/Prediction

Using computer vision based deep learning approaches on the PREVENTION dataset (https://prevention-dataset.uah.es/) for the recognition and prediction of
lane changing behaviour of surrounding vehicles in highways.

### Downloading the dataset
To download the PREVENTION dataset, run the following command in the same directory as the file *data_downloader.sh*:
```
bash data_downloader.sh RECORD DRIVE
```
where **RECORD** is the number of the RECORD in the dataset and **DRIVE** is the number of the DRIVE of the record, both input as integers.

### Lane Change / No Lane Change clip extraction
The dataset is obtained from preprocessing the 5 records that are in the PREVENTION dataset. From these records,
small clips have been extracted that were used to train and test the implemented models.
All the preprocessing related scripts can be found in the `data_preprocessing` folder.

Work flow for clip extraction with the scripts:
1. Use `LC_exractor.py` or `NLC_extractor.py` to extract the clips from the records.
2. After manually checking and removing outliers from the dataset, use `move_processed.py` to move the files
to another folder and label them as processed.
3. After you have extracted and processed all the needed clips, use `clip_store.py` to generate clip store csvs. 
The `clip_store.csv` file can be used to track the amount of clips found, and is used to split the data to test-train.

### Test-train split with data augmentation
The test-train split with data augmentation can be done with `data_splitter.py`. It uses 5 different data augmentation
techniques on the original training data: horizontal flipping, gaussian noise, color jitter, random rotation, and random brightness increase/decrease.

### Models

### Experiments
All the done experiments and their results with the implemented models can be found in `notebooks`. 

