#!/bin/bash

mkdir -p mysql
mkdir -p ./gate-incoming/saved_videos
mkdir -p ./gate-incoming/saved_numberplates
mkdir -p ./gate-exiting/saved_videos

sudo docker-compose down
sudo docker network create --subnet=172.18.0.0/24 external
sudo docker-compose up -d --build --force-recreate
