#!/bin/bash

# train
#python main.py -d ICEWS18 --history_len 4 --num_head 4 --num_ly 3 --topk 30 --decay 1e-4 --gpu 2 --test 0

## test
#python main.py -d ICEWS18 --history_len 4 --num_head 4 --num_ly 3 --topk 30 --decay 1e-4 --gpu 7 --test 1

# online train valid
python main.py -d ICEWS18 --history_len 4 --num_head 4 --num_ly 3 --topk 30 --decay 1e-4 --gpu 7 --test 2 \
--temperature 2 --distill_weight 1 --con_description clod_10.10

# online train test
python main.py -d ICEWS18 --history_len 4 --num_head 4 --num_ly 3 --topk 30 --decay 1e-4 --gpu 7 --test 3 \
--temperature 2 --distill_weight 1 --con_description clod_10.10
