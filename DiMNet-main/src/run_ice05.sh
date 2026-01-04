#!/bin/bash

## train
#python main.py -d ICEWS05-15 --history_len 2 --num_head 1 --num_ly 1 --topk 50 --decay 1e-4 --gpu 1 --test 0

# test
#python main.py -d ICEWS05-15 --history_len 2 --num_head 1 --num_ly 1 --topk 50 --decay 1e-4 --gpu 3 --test 1

# online train valid
python main.py -d ICEWS05-15 --history_len 2 --num_head 1 --num_ly 1 --topk 50 --decay 1e-4 --gpu 3 --test 2 \
--temperature 2 --distill_weight 1 --con_description clod_10.10

# online train test
python main.py -d ICEWS05-15 --history_len 2 --num_head 1 --num_ly 1 --topk 50 --decay 1e-4 --gpu 3 --test 3 \
--temperature 2 --distill_weight 1 --con_description clod_10.10
