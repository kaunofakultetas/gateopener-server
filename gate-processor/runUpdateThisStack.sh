#!/bin/bash

mkdir -p mysql
mkdir -p ./gate-incoming/saved_videos
mkdir -p ./gate-incoming/saved_numberplates
mkdir -p ./gate-exiting/saved_videos

sudo docker-compose down
sudo docker-compose up -d --build --force-recreate
