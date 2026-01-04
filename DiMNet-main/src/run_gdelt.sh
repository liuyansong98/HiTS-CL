#!/bin/bash

## train
#python main.py -d GDELT --history_len 5 --num_head 1 --num_ly 3 --topk 50 --decay 1e-4 --gpu 2 --test 0

# test
#python main.py -d GDELT --history_len 5 --num_head 1 --num_ly 3 --topk 50 --decay 1e-4 --gpu 2 --test 1

# online train valid
python main.py -d GDELT --history_len 5 --num_head 1 --num_ly 3 --topk 50 --decay 1e-4 --gpu 1 --test 2 \
--temperature 2 --distill_weight 1 --con_description clod_10.10

# online train test
python main.py -d GDELT --history_len 5 --num_head 1 --num_ly 3 --topk 50 --decay 1e-4 --gpu 1 --test 3 \
--temperature 2 --distill_weight 1 --con_description clod_10.10
