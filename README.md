# BSc Thesis: Deep Learning - Lane Change Recognition & Prediction

Using computer vision based deep learning approaches on the PREVENTION dataset (https://prevention-dataset.uah.es/) for the recognition 
and prediction of lane changing behaviour of surrounding vehicles in highways.

### Report
For a detailed understanding of the research and methodologies employed in this project, you can refer to the [report](https://drive.google.com/file/d/1KBE7uTcpW-UwwpLsnkbMafijDF6mizen/view?usp=sharing).

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
1. Use `LC_exractor.py` to extract the lane change clips or `NLC_extractor.py` to extract the no lane change clips from the records.
2. After manually checking and removing outliers from the dataset, use `move_processed.py` to move the files
to another folder and label them as processed. the script assumes that all the same lane change clips are manually labeled as passed 
and put into a `passed` subfolder or all files are moved by putting variable `move_all_files = True`.
3. After you have extracted and processed all (or some of) the needed clips, use `clip_store.py` to generate clip store csv files. 
The `clip_store.csv` file can be used to track the amount of clips extracted, and is used to split the data to test-train.

NOTE: both *LC_extractor.py* and *NLC_extractor.py* contain some variables that can be used to tune the extraction
specifications.

### Test-train split with data augmentation
The test-train split with data augmentation can be done with `data_splitter.py`. It uses 5 different data augmentation
techniques on the original training data: horizontal flipping, gaussian noise, color jitter, random rotation, and 
random brightness increase/decrease.
When splitting the data, the scripts assumes that all regions of interest (ROIs) have equal amount of data to be splitted so
it can produce same splits for each ROI size.

### Models
The details about the implemented models can be found in `models.py` file inside the ``models`` directory. The directory also
contains a file `helper_functions.py` with useful functions used for the training and testing of the models. 
Four different models were implemented: R(2+1)D [2], MC4 [2], S3D[3] and ViViT [1].

The trained model weights and full dataset can be accessed through `weights/model_weights.md`.

### Experiments
All the done experiments and their results with the implemented models can be found in `notebooks`.

### References
[1] Anurag Arnab et al. "ViViT: A Video Vision Transformer". 2021. arXiv: 2103.15691.

[2] Du Tran et al. “A Closer Look at Spatiotemporal Convolutions for Action Recognition”. In: June 2018, pp. 6450–6459. DOI: 10.1109/CVPR.2018.00675.

[3] Saining Xie et al. "Rethinking Spatiotemporal Feature Learning:
Speed-Accuracy Trade-offs in Video Classification". 2018. arXiv: 1712.04851.
