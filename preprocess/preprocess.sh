#!/bin/sh

id=$1
rm -rf /mnt/out
mkdir /mnt/out
cd /mnt/out
wget http://roadhack-sources.bj.bcebos.com/train/image/"$id".zip -O image.zip &
wget http://roadhack-sources.bj.bcebos.com/train/attr/"$id".zip -O attr.zip &
wait
mkdir attr image
unzip -d image image.zip &
unzip -d attr attr.zip &
wait
python /root/workspace/preprocess/preprocess.py -p /mnt/out/ -i $id

for i in `ls -al | grep '\.[0-9]' | awk '{print $9}' | sort | xargs`
do
    echo "/mnt/out/$i" >> ../files_list
done
