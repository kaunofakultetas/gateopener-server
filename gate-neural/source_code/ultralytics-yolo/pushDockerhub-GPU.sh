#!/bin/bash
set -e

CONTAINER_NAME=gateopener-neural-gpu

TODAY=$(date +%Y%m%d)
sudo docker login -u admin@knf.vu.lt
sudo docker build -t $CONTAINER_NAME -f Dockerfile.gpu .
sudo docker tag $CONTAINER_NAME vuknf/$CONTAINER_NAME:latest
sudo docker tag $CONTAINER_NAME vuknf/$CONTAINER_NAME:$TODAY
sudo docker push vuknf/$CONTAINER_NAME:latest
sudo docker push vuknf/$CONTAINER_NAME:$TODAY