#!/bin/bash
# type in the correct integer for drive and record as input args
RECORD=$1
DRIVE=$2

OUT_DIR="./UAH PREVENTION/RECORD$RECORD/DRIVE$DRIVE"

# creating download folder to OUT_DIR
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# load the data from the web
wget "https://prevention-dataset.uah.es/static/RECORD$RECORD/DRIVE$DRIVE/video_camera1.mp4" --no-check-certificate
wget "https://prevention-dataset.uah.es/static/RECORD$RECORD/DRIVE$DRIVE/processed_data.zip" --no-check-certificate

# unzip final data
unzip "./processed_data.zip"

# remove zip file and the unnecessary folders to save disc memory
rm "./processed_data.zip"
rm -rf "./processed_data/detection_camera2"
rm -rf "./processed_data/logs"
rm -rf "./processed_data/detection_cloud"
rm -rf "./processed_data/detection_radar"
