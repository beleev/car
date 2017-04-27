for i in `cat ids`
do
    echo $i
    sh download.sh $i
done
