rm -f train.log
for i in `cat ids`
do
    python demo.py --data_id=$i --model_path=./model/model --save_path=./model/model >> train.log 2>&1
    wait
done
