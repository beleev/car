#!/bin/bash

id=$1

mkdir -p ./data/$id
cd ./data/$id
wget http://roadhack-sources.bj.bcebos.com/train/image/"$id".zip -O image.zip &
wget http://roadhack-sources.bj.bcebos.com/train/attr/"$id".zip -O attr.zip &
wait
mkdir attr image
unzip -d image image.zip &
unzip -d attr attr.zip &
wait
