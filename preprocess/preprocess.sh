#!/bin/bash

id=$1
hostnum=$2

rm -rf /mnt/out* /mnt/temp /mnt/*sp_file* /mnt/files_list

sh ./download.sh $id

python /root/workspace/preprocess/preprocess.py -p /mnt/download/ -i $id
#rm -rf image.zip attr.zip image attr

for i in `ls -al | grep '\.[0-9]' | awk '{print $9}' | sort | xargs`
do
    echo "/mnt/download/$i" >> files_list
done

if [ $hostnum -gt 1 ]
then
    mv files_list ..
    cd ..
    mv download temp
    numall=`ls -al temp | grep '\.[0-9]' | wc -l`
    num=`echo $numall/$hostnum | bc`
    split -l $num files_list sp_file
    line=1
    for i in `ls sp_file*`
    do
        mkdir dir_$i
        mv $i dir_$i/files_list
        for j in `cat dir_$i/files_list | awk -F\/ '{print $4}' | xargs`
        do
            mv /mnt/temp/$j dir_$i/
        done
        tar czf "$i".tgz dir_$i
    done
 
    #for i in `ls sp_file*`
    #do
    #    host=`sed -n "$line"p /root/workspace/preprocess/hosts`
    #    ssh -p 2222 $host "rm -rf /mnt/out /mnt/*.tgz"
    #    scp -P 2222 "$i".tgz $host:/mnt
    #    ssh -p 2222 $host "cd /mnt; tar zxf /mnt/dir_$i; mv dir_$i out"
    #    ((line = $line + 1))
    #done
else
    mv /mnt/download /mnt/out
fi
