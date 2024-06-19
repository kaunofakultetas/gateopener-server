#!/bin/bash


# Neural model downloader
download_file_if_not_exists() {
    local FILE_PATH="$1"
    local DOWNLOAD_URL="$2"
    local TEMP_FILE_PATH="${FILE_PATH}.TMP"

    rm -f "$TEMP_FILE_PATH"

    # Check if the file exists
    if [ -f "$FILE_PATH" ]; then
        echo "File already exists: $FILE_PATH"
    else
        echo "File does not exist. Downloading: $FILE_PATH"
        wget -O "$TEMP_FILE_PATH" "$DOWNLOAD_URL"
        
        # Check if the download was successful
        if [ $? -eq 0 ]; then
            mv "$TEMP_FILE_PATH" "$FILE_PATH"
            echo "File downloaded successfully: $FILE_PATH"
        else
            echo "Failed to download the file: $FILE_PATH"
            rm -f "$TEMP_FILE_PATH"
        fi
    fi
}


# Download models if they do not exist in our directory.
mkdir -p yolov7-np-location
download_file_if_not_exists "./yolov7-np-location/LP_detect_yolov7_500img.pt" "https://github.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN/releases/download/Model/LP_detect_yolov7_500img.pt"

mkdir -p yolov7-objects
download_file_if_not_exists "./yolov7-objects/yolov7.pt" "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
download_file_if_not_exists "./yolov7-objects/yolov7-tiny.pt" "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
download_file_if_not_exists "./yolov7-objects/yolov7-e6e.pt" "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt"

sudo docker-compose down
sudo docker-compose up -d --build --force-recreate
