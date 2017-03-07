#!/bin/sh

id=$1
hostnum=$2

rm -rf /mnt/out* /mnt/temp /mnt/dir_sp_file*
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
#rm -rf image.zip attr.zip image attr

for i in `ls -al | grep '\.[0-9]' | awk '{print $9}' | sort | xargs`
do
    echo "/mnt/out/$i" >> files_list
done

if [ $hostnum -gt 1 ]
then
    mv files_list ..
    cd ..
    mv out temp
    numall=`ls -al temp | grep '\.[0-9]' | wc -l`
    num=`echo $numall/$hostnum | bc`
    split -l $num files_list sp_file
    for i in `ls sp_file*`
    do
        mkdir dir_$i
        mv $i dir_$i/files_list
        for j in `cat dir_$i/files_list | awk -F\/ '{print $4}' | xargs`
        do
            mv /mnt/temp/$j dir_$i/
        done
        tar czf "$i".tgz $i
    done
fi
