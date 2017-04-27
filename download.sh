#!/bin/bash

id=$1

rm -rf /mnt/data/$id
mkdir -p /mnt/data/$id
cd /mnt/data/$id
wget http://roadhack-sources.bj.bcebos.com/train/image/"$id".zip -O image.zip &
wget http://roadhack-sources.bj.bcebos.com/train/attr/"$id".zip -O attr.zip &
wait
unzip image.zip
mv "$id".h5 image.h5
rm -f "$id".h5
unzip attr.zip
mv "$id".h5 attr.h5
rm -f "$id".h5
rm -f *.zip
echo "GET #$id data DONE!"
