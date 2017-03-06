#!/bin/sh

#python paddle.py \
#  --job_workspace="/root/demo/recommendation" \
#  --dot_period=10 \
#  --ports_num_for_sparse=2 \
#  --log_period=50 \
#  --num_passes=10 \
#  --trainer_count=4 \
#  --saving_period=1 \
#  --local=0 \
#  --config=./trainer_config.py \
#  --save_dir=./output \
#  --use_gpu=0

python paddle.py \
  --job_dispatch_package="/root/workspace" \
  --dot_period=10 \
  --log_period=50 \
  --num_passes=50 \
  --trainer_count=6 \
  --saving_period=1 \
  --local=0 \
  --config=./trainer.py \
  --save_dir=./output \
  --use_gpu=0 \
  --show_layer_stat=1
