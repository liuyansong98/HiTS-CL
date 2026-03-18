#!/bin/bash

# train
python main.py -d ICEWS14s --history_len 10 --num_head 4 --num_ly 3 --topk 50 --decay 1e-4 --gpu 7 --test 0

# test
python main.py -d ICEWS14s --history_len 10 --num_head 4 --num_ly 3 --topk 50 --decay 1e-4 --gpu 6 --test 1

# online train valid
python main.py -d ICEWS14s --history_len 10 --num_head 4 --num_ly 3 --topk 50 --decay 1e-4 --gpu 6 --test 2 \
--temperature 2 --distill_weight 1 --con_description continual_learning --base_capacity 3 --flexible_capacity 0.3

# online train test
python main.py -d ICEWS14s --history_len 10 --num_head 4 --num_ly 3 --topk 50 --decay 1e-4 --gpu 6 --test 3 \
--temperature 2 --distill_weight 1 --con_description continual_learning --base_capacity 3 --flexible_capacity 0.3
