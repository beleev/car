for i in `cat down_id`
do
    echo $i
    sh download.sh $i
done
