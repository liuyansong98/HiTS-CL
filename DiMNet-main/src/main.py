import sys
import torch
from tqdm import tqdm
import random
import numpy as np
import os
import pickle

sys.path.append("..")
import utils
from torch import nn
import json
from config import args
from model import DiMNet
from datetime import datetime
from torch.utils import data as torch_data
from torch import distributed as dist
from typing import Dict
import wandb
from collections import defaultdict, OrderedDict
import scipy.sparse as sp
import torch.nn.functional as F
import copy
from collections import deque
import time

class LRUCache1:
    def __init__(self):
        self.cache = OrderedDict()

    def put(self, key: int, value: int) -> None:
        freq = 1
        if key in self.cache:
            # 如果 key 已经存在，更新其值并将其移到末尾
            _, freq = self.cache[key]
            freq += 1
            self.cache.move_to_end(key)
        # 更新频率
        self.cache[key] = (value, freq)

    def pop(self):
        return self.cache.popitem(last=False)

    def remove(self, key):
        del self.cache[key]


class LFUCache1:
    def __init__(self):
        self.cache = {}  # key -> (value, freq)
        self.freq = defaultdict(OrderedDict)  # freq -> OrderedDict of keys preserving LRU order
        self.min_freq = 0

    def update_fre(self, key: int) -> int:
        if key not in self.cache:
            return -1
        value, freq = self.cache[key]
        # 将该键从当前频率列表中移除
        del self.freq[freq][key]
        # 空频率列表且频率正好是当前最小频率时，min_freq 加 1
        if not self.freq[freq]:
            del self.freq[freq]
            if freq == self.min_freq:
                self.min_freq += 1
        # 更新频率并加入新的 freq 列表
        self.freq[freq + 1][key] = value
        self.cache[key] = (value, freq + 1)
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 更新已有条目内容并递增频率
            _, freq = self.cache[key]
            self.cache[key] = (value, freq)
            self.update_fre(key)  # 更新频率
            return

        # 插入新键，频率初始化为 1
        self.cache[key] = (value, 1)
        self.freq[1][key] = value
        self.min_freq = 1

    def pop(self):
        evict_key, _ = self.freq[self.min_freq].popitem(last=False)
        # 空频率列表且频率正好是当前最小频率时，min_freq 加 1
        if not self.freq[self.min_freq]:
            del self.freq[self.min_freq]
            if self.freq:
                self.min_freq = min(self.freq.keys())
            else:
                self.min_freq = 0
        del self.cache[evict_key]
        return evict_key


class ARCCache1:
    def __init__(self, capacity):
        self.capacity = capacity
        self.p = 0  # T1 的目标大小
        self.cache = {}  # key -> value
        self.t1 = LRUCache1()  # 最近访问但只访问一次的缓存页（T1）, LRU
        self.b1 = deque()  # T1 的 ghost 列表（只保存 key）
        self.t2 = LFUCache1()  # 访问频繁的缓存页（T2）, LFU
        self.b2 = deque()  # T2 的 ghost 列表

    def set_capccity(self, capacity):
        self.capacity = capacity

    def replace(self, key):
        # 按 ARC 论文策略决定从 T1 还是 T2 驱逐
        if self.t1.cache and ((key in self.b2 and len(self.t1.cache) == self.p) or (len(self.t1.cache) > self.p)):
            old, _ = self.t1.pop()
            self.b1.appendleft(old)
        else:
            old = self.t2.pop()
            self.b2.appendleft(old)
        del self.cache[old]

    def put(self, key, value):
        if key in self.cache:
            # 命中：若在 T1 移入 T2，否则保持在 T2
            if key in self.t1.cache:
                self.t1.remove(key)
                self.t2.put(key, value)
                freq = 2
            else:
                _, freq = self.cache[key]
                self.t2.put(key, value)
                freq += 1
            # 写入缓存
            self.cache[key] = (value, freq)
            # a = set(self.cache.keys())
            # b = set(self.t1.cache.keys()).union(set(self.t2.cache.keys()))
            # if a != b:
            #     print("cache集合不相等", )
            return

        # 未命中：先加载
        # 如果在 B1，说明曾经近期被驱逐
        if key in self.b1:
            self.p = min(self.capacity, self.p + max(len(self.b2) // len(self.b1), 1))
            self.replace(key)
            self.b1.remove(key)
            self.t2.put(key, value)
            freq = 2
        # 如果在 B2，说明曾经频繁访问过但被驱逐
        elif key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) // len(self.b2), 1))
            self.replace(key)
            self.b2.remove(key)
            self.t2.put(key, value)
            freq = 2
        elif len(self.t1.cache) + len(self.b1) == self.capacity:
            # 新 key
            if len(self.t1.cache) < self.capacity:
                self.b1.pop()
                self.replace(key)
            else:
                old, _ = self.t1.pop()
                del self.cache[old]
            self.t1.put(key, value)
            freq = 1
        else:
            total = len(self.t1.cache) + len(self.b1) + len(self.t2.cache) + len(self.b2)
            if total >= self.capacity:
                if total == 2 * self.capacity:
                    self.b2.pop()
                self.replace(key)
            self.t1.put(key, value)
            freq = 1

        # 最后写入缓存
        self.cache[key] = (value, freq)
        # a = set(self.cache.keys())
        # b = set(self.t1.cache.keys()).union(set(self.t2.cache.keys()))
        # if a != b:
        #     print("cache集合不相等", )
        return


def cal_entropy1(result, temperature):
    result = F.softmax(result, dim=-1)
    probs = torch.sigmoid(-torch.sum(result * torch.log(result + 1e-12), dim=-1)) * temperature
    return probs.unsqueeze(dim=-1)


def boosting_cal_entropy(scores, target):
    return F.cross_entropy(scores, target, reduction='none')


def train_and_validate(args, model, train_list, valid_list, test_list, num_nodes, num_rels, model_state_file):
    print(
        "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart training\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    best_mrr = 0
    patience = args.patience
    start_train_time = time.perf_counter()
    for epoch in range(args.n_epoch):
        if patience == 0:
            print("Early stopping at epoch: {}".format(epoch))
            break
        # print("\nepoch:"+str(epoch)+ ' Time: ' + datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'))
        model.train()
        losses = list()
        cllosses = list()
        predlosses = list()

        idx = [_ for _ in range(len(train_list))]  # timestamps index [0,1,2,3,...,n]
        random.shuffle(idx)
        idx_proc = tqdm(idx, ncols=100, desc='Epoch %i' % epoch)
        for future_sample_id in idx_proc:
            # for future_sample_id in idx:
            if future_sample_id == 0: continue
            # future_sample as the future graph
            futrue_graph = train_list[future_sample_id]
            # future_triple : [num_edges, 3] (format: h,t,r)
            # Note that we also add reverse edges in 'future_triple' as query query_triple
            future_triple = torch.cat((futrue_graph.edge_index, futrue_graph.edge_type.unsqueeze(0))).t()  # 包含逆三元组
            # x = futrue_graph.target_triplets  # 不包含逆三元组
            # get history graph list
            if future_sample_id - args.history_len < 0:
                history_list = train_list[0: future_sample_id]
            else:
                history_list = train_list[future_sample_id - args.history_len:
                                          future_sample_id]

            batch = future_triple  # all future tirples is an only batch

            pred, cl_loss = model(history_list, batch)
            pred_loss = model.get_loss(pred, batch[:, 1])
            loss = pred_loss + cl_loss

            predlosses.append(pred_loss.item())
            cllosses.append(cl_loss.item() if cl_loss != 0 else 0.0)
            losses.append(loss.item())

            # idx_proc.set_postfix(loss='%f'%(sum(losses) / len(losses)))
            idx_proc.set_postfix(pred_loss='%f' % (sum(predlosses) / len(predlosses)),
                                 cl_loss='%f' % (sum(cllosses) / len(cllosses)))
            wandb.log({"pred_loss": sum(predlosses) / len(predlosses),
                       "cl_loss": sum(cllosses) / len(cllosses),
                       "loss": sum(losses) / len(losses)})
            # print({"pred_loss": sum(predlosses) / len(predlosses),
            #        "cl_loss": sum(cllosses) / len(cllosses),
            #        "loss": sum(losses) / len(losses)})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        # evaluation
        print("valid dataset eval:", end='\t')
        metrics_valid = test(model, valid_list, num_rels, num_nodes)
        wandb.log({"MRR": metrics_valid['mrr'],
                   "H1": metrics_valid['hits@1'],
                   "H3": metrics_valid['hits@3'],
                   "H10": metrics_valid['hits@10'],
                   "epoch": epoch})
        # print({"MRR": metrics_valid['mrr'],
        #            "H1": metrics_valid['hits@1'],
        #            "H3": metrics_valid['hits@3'],
        #            "H10": metrics_valid['hits@10'],
        #            "epoch": epoch})

        if metrics_valid['mrr'] >= best_mrr:
            best_mrr = metrics_valid['mrr']
            patience = args.patience
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'args': args}, model_state_file)
            print("--------best_mrr updated!--------")

        else:
            patience -= 1
            print("---------------------------------")
    end_train_time = time.perf_counter()
    print(f"DiMNet训练时间(80% train): {end_train_time - start_train_time:.4f} 秒")
    # testing

    print("\nFinal eval test dataset with best model:...")
    metrics_test = test(model, test_list, num_rels, num_nodes, mode="test", model_name=model_state_file)
    wandb.log({"MRR-test": metrics_test['mrr'],
               "H1-test": metrics_test['hits@1'],
               "H3-test": metrics_test['hits@3'],
               "H10-test": metrics_test['hits@10']})
    # print({"MRR-test": metrics_test['mrr'],
    #            "H1-test": metrics_test['hits@1'],
    #            "H3-test": metrics_test['hits@3'],
    #            "H10-test": metrics_test['hits@10']})

    return best_mrr

def temporal_regularization(params1, params2):
    regular = 0
    for (param1, param2) in zip(params1, params2):
        regular += torch.norm(param1 - param2, p=2)
    # print(regular)
    return regular


def continuous_test(args, model, history_graph_list, test_graph_list, history_data_list, test_data_list, train_data,
                    num_nodes, num_rels, mode="valid", model_name=None):
    total_train_time = 0
    print(
        f"\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\ncontinuous learning on {mode}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # step 1: 初始化变量，加载模型，准备数据
    all_graph_list = history_graph_list + test_graph_list
    all_data_list = history_data_list + test_data_list
    # 计算memory unit容量
    history_flex_cap_no_Rep = [0 for _ in range(num_nodes * 2 * num_rels)]
    train_triple_list = []
    train_src_l = train_data[:, 0]
    train_e_idx_l = train_data[:, 1]
    train_dst_l = train_data[:, 2]
    train_ts_l = train_data[:, 3]
    for i in range(len(train_src_l)):
        train_triple_list.append((train_src_l[i], train_e_idx_l[i], train_dst_l[i]))  # 三元组
    train_triple_set = list(set(train_triple_list))
    for i in range(len(train_triple_set)):  # 不计算重复出现的三元组
        history_flex_cap_no_Rep[train_triple_set[i][0] * 2 * num_rels + train_triple_set[i][1]] += 1
        history_flex_cap_no_Rep[train_triple_set[i][2] * 2 * num_rels + num_rels + train_triple_set[i][1]] += 1

    history_cap = np.floor(np.array(history_flex_cap_no_Rep) * args.flexible_capacity + args.base_capacity)

    # 初始化一个字典，保存已经出现的实体id和频率
    count_dict_o = {}
    count_dict_so = {}
    ranks_raw_cold_o, ranks_filter_cold_o, ranks_raw_cold_so, ranks_filter_cold_so = [], [], [], []

    # 初始化历史记录memory
    valid_history_entity_dic: Dict[(int, int), ARCCache1] = {}  # 定义一个空字典(s,r)->LRUcache
    train_history_entity_dic: Dict[(int, int), ARCCache1] = {}  # 定义一个空字典(s,r)->LRUcache
    test_history_entity_dic: Dict[(int, int), ARCCache1] = {}  # 定义一个空字典(s,r)->LRUcache
    for i in range(num_nodes):
        for j in range(2 * num_rels):
            if history_cap[i * 2 * num_rels + j] == 0:
                continue
            else:
                # valid_history_entity_dic[(i, j)] = ARCCache1(history_cap[i * 2 * num_rels + j])
                # train_history_entity_dic[(i, j)] = ARCCache1(history_cap[i * 2 * num_rels + j])
                test_history_entity_dic[(i, j)] = ARCCache1(history_cap[i * 2 * num_rels + j])
    # all_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_nodes * num_rels, num_nodes))
    # 初始化历史记录 和历史实体指示矩阵
    for snap in history_data_list[:-2]:
        tmp_src_l = snap[:, 0]
        tmp_e_idx_l = snap[:, 1]
        tmp_dst_l = snap[:, 2]
        tmp_ts_l = snap[:, 3]
        for i in range(len(tmp_src_l)):
            test_history_entity_dic[tmp_src_l[i], tmp_e_idx_l[i]].put(tmp_dst_l[i], tmp_ts_l[i])
        for i in range(len(tmp_src_l)):
            test_history_entity_dic[tmp_dst_l[i], tmp_e_idx_l[i] + num_rels].put(tmp_src_l[i], tmp_ts_l[i])
        for element in tmp_dst_l:
            count_dict_o[element] = count_dict_o.get(element, 0) + 1
            count_dict_so[element] = count_dict_so.get(element, 0) + 1
        for element in tmp_src_l:
            count_dict_so[element] = count_dict_so.get(element, 0) + 1
        # row = tmp_src_l * num_rels + tmp_e_idx_l
        # col = tmp_dst_l
        # d1 = np.ones(len(row))
        # tmp_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_nodes * num_rels, num_nodes))
        # all_tail_seq = all_tail_seq + tmp_tail_seq
    tmp_src_l = history_data_list[-2][:, 0]
    tmp_e_idx_l = history_data_list[-2][:, 1]
    tmp_dst_l = history_data_list[-2][:, 2]
    tmp_ts_l = history_data_list[-2][:, 3]
    for i in range(len(tmp_src_l)):
        test_history_entity_dic[tmp_src_l[i], tmp_e_idx_l[i]].put(tmp_dst_l[i], tmp_ts_l[i])
    for i in range(len(tmp_src_l)):
        test_history_entity_dic[tmp_dst_l[i], tmp_e_idx_l[i] + num_rels].put(tmp_src_l[i], tmp_ts_l[i])
    for element in tmp_dst_l:
        count_dict_o[element] = count_dict_o.get(element, 0) + 1
        count_dict_so[element] = count_dict_so.get(element, 0) + 1
    for element in tmp_src_l:
        count_dict_so[element] = count_dict_so.get(element, 0) + 1

    tmp_src_l = history_data_list[-1][:, 0]
    tmp_e_idx_l = history_data_list[-1][:, 1]
    tmp_dst_l = history_data_list[-1][:, 2]
    tmp_ts_l = history_data_list[-1][:, 3]
    for i in range(len(tmp_src_l)):
        test_history_entity_dic[tmp_src_l[i], tmp_e_idx_l[i]].put(tmp_dst_l[i], tmp_ts_l[i])
    for i in range(len(tmp_src_l)):
        test_history_entity_dic[tmp_dst_l[i], tmp_e_idx_l[i] + num_rels].put(tmp_src_l[i], tmp_ts_l[i])
    for element in tmp_dst_l:
        count_dict_o[element] = count_dict_o.get(element, 0) + 1
        count_dict_so[element] = count_dict_so.get(element, 0) + 1
    for element in tmp_src_l:
        count_dict_so[element] = count_dict_so.get(element, 0) + 1

    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_init, ranks_filter_init, mrr_raw_list_init, mrr_filter_list_init = [], [], [], []
    ranks_raw_enh, ranks_filter_enh, mrr_raw_list_enh, mrr_filter_list_enh = [], [], [], []
    init_sups_num, conti_sups_num = [], []
    distill_enh_sups_both, distill_enh_sups_init, distill_enh_sups_conti, distill_enh_lag_both = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_raw1, ranks_filter1 = [], []
    rank_diff_raw_list, rank_diff_time_list = [], []
    diff_raw_entity_list, diff_time_entity_list = [], []
    diff_raw_entity_list1, diff_time_entity_list1 = [], []
    instance_num_list = []
    ranks_raw_distill, ranks_filter_distill = [], []
    mrr_raw_list_distill, mrr_filter_list_distill = [], []
    ranks_raw_distill_enh, ranks_filter_distill_enh, mrr_raw_list_distill_enh, mrr_filter_list_distill_enh = [], [], [], []

    start_idx = len(history_data_list)
    # load pretrained model which valid in the whole valid data
    if mode == "valid":
        save_state_file = model_name + args.con_description + "_con-valid"
        model_state_file_distill = model_name + args.con_description + "_con-valid-distill"
    else:
        save_state_file = model_name + args.con_description + "_con-test"
        model_state_file_distill = model_name + args.con_description + "_con-test-distill"

    if not os.path.exists(model_name):
        print("Pretrain the model first before continual learning...")
        sys.exit()
    else:
        init_checkpoint = torch.load(model_name, map_location='cpu')
        if mode == "test":
            checkpoint = torch.load(model_name + args.con_description + "_con-valid", map_location='cpu')
        else:
            checkpoint = torch.load(model_name, map_location='cpu')
        print("Load pretrain model: {}.".format(model_name))  # use best stat checkpoint
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # save an init model for analysis
        # model_initial = copy.deepcopy(model)
        # model_distill = copy.deepcopy(model)
        # model_initial.load_state_dict(init_checkpoint['state_dict'], strict=False)
        # model_initial.to(device)
        # model_distill.to(device)
        # model_initial.eval()
        # parameter for the temporal normalize at the first timestamp
        previous_param = [param.detach().clone() for param in model.parameters()]
        model.to(device)
        model.eval()
        # epoch = checkpoint['epoch']
        # 保存一个在线训练的模型权重
        torch.save({'state_dict': model.state_dict(), 'epoch': -1}, save_state_file)
        torch.save({'state_dict': model.state_dict(), 'epoch': -1}, model_state_file_distill)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=args.decay)
    valid_history_glist = history_graph_list[-args.history_len - 2: -2]
    valid_snap = history_graph_list[-2]
    valid_tensor = torch.LongTensor(history_data_list[-2]).to(device)
    ft_history_glist = history_graph_list[-args.history_len - 1: -1]
    ft_snap = history_graph_list[-1]
    ft_tensor = torch.LongTensor(history_data_list[-1]).to(device)
    test_history_glist = history_graph_list[-args.history_len:]

    for time_idx, test_snap in enumerate(test_graph_list):
        test_tensor = torch.LongTensor(test_data_list[time_idx]).to(device)
        tc = start_idx + time_idx
        print(f"----------------{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [tc: {tc}]--------------", flush=True)
        print(f"DiMNet训练时间: {total_train_time:.4f} 秒")
        # result of the pre-trained model on validation set (tc-1)。上一阶段持续训练模型在验证集结果
        # valid_triple_all = torch.cat((valid_snap.edge_index, valid_snap.edge_type.unsqueeze(0))).t()  # 包含逆四元组

        num_int = random.randint(args.history_len, tc - 1)
        replay_history_glist = [snap for snap in all_graph_list[num_int - args.history_len:num_int]]
        replay_snap = all_graph_list[num_int]
        replay_tensor = torch.LongTensor(all_data_list[num_int]).to(device)

        # result of the pre-trained model on test set (tc)。预训练模型在测试集结果
        test_triple = test_snap.target_triplets  # 不含逆四元组
        # 初始模型
        model.load_state_dict(init_checkpoint['state_dict'], strict=False)
        model.eval()
        with torch.no_grad():
            test_pred, _ = model(test_history_glist, test_triple)
            mrr_filter_test_snap_init, mrr_test_snap_init, rank_test_raw_init, rank_test_filter_init = utils.get_total_rank(
                test_pred, test_triple,
                num_nodes)
            print("Pretrained Model : test mrr ", mrr_filter_test_snap_init)

            valid_triple = valid_snap.target_triplets  # 不含逆四元组
            val_final_score_init, _ = model(valid_history_glist, valid_triple)
            mrr_filter_valid_snap_init, mrr_valid_snap_init, val_rank_raw_init, val_rank_filter_init = utils.get_total_rank(
                val_final_score_init, valid_triple,
                num_nodes)
            ft_triple = ft_snap.target_triplets  # 不含逆四元组
            ft_final_score_init, _ = model(ft_history_glist, ft_triple)
            mrr_filter_ft_snap_init, mrr_ft_snap_init, ft_rank_raw_init, ft_rank_filter_init = utils.get_total_rank(
                ft_final_score_init, ft_triple,
                num_nodes)

        tmp_checkpoint = torch.load(save_state_file, map_location='cpu')
        model.load_state_dict(tmp_checkpoint['state_dict'], strict=False)
        with torch.no_grad():
            # result of the last step fine-tuned model on test set (tc) # first iteration: mrr on first test timestep。微调模型测试集测试
            test_triple = test_snap.target_triplets  # 不含逆四元组
            test_pred, _ = model(test_history_glist, test_triple)
            mrr_filter_test_snap, mrr_test_snap, rank_raw, rank_filter = utils.get_total_rank(
                test_pred, test_triple,
                num_nodes)
            print("Continual Model : test mrr before ft ", mrr_filter_test_snap)
            valid_pred, _ = model(valid_history_glist, valid_triple)
            mrr_filter_valid_snap, mrr_valid_snap, rank_raw, rank_filter = utils.get_total_rank(valid_pred,
                                                                                                valid_triple,
                                                                                                num_nodes)

        # init mrr for validation
        best_mrr = mrr_filter_valid_snap
        ft_epoch, losses = 0, []

        # tmp_checkpoint = torch.load(model_state_file_distill, map_location='cpu')
        # model.load_state_dict(tmp_checkpoint['state_dict'], strict=False)
        # step 3: 微调训练
        start_ft_time = time.perf_counter()
        while ft_epoch < args.ft_epochs:
            model.train()
            # future_triple : [num_edges, 3] (format: h,t,r)
            # Note that we also add reverse edges in 'future_triple' as query query_triple
            ft_triple_all = torch.cat((ft_snap.edge_index, ft_snap.edge_type.unsqueeze(0))).t()  # 包含逆四元组
            ft_pred, cl_loss = model(ft_history_glist, ft_triple_all)
            pred_loss = model.get_loss(ft_pred, ft_triple_all[:, 1])

            # regularization
            # loss_norm = temporal_regularization(model.parameters(), previous_param)
            # loss = pred_loss + cl_loss + 0.1 * loss_norm

            # replay
            # replay_triple_all = torch.cat((replay_snap.edge_index, replay_snap.edge_type.unsqueeze(0))).t()  # 包含逆四元组
            # replay_pred, replay_cl_loss = model(replay_history_glist, replay_triple_all)
            # replay_pred_loss = model.get_loss(replay_pred, replay_triple_all[:, 1])
            # loss = pred_loss + cl_loss + replay_pred_loss + replay_cl_loss

            loss = pred_loss + cl_loss
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

            # evaluation
            print("valid dataset eval:", end='\t')
            model.eval()
            valid_triple = valid_snap.target_triplets  # 不含逆四元组
            valid_pred, _ = model(valid_history_glist, valid_triple)
            mrr_filter_valid_snap, mrr_valid_snap, rank_raw, rank_filter = utils.get_total_rank(valid_pred,
                                                                                                valid_triple,
                                                                                                num_nodes)
            print(f"epoch:{ft_epoch}, mrr_filter_valid_snap:{mrr_filter_valid_snap}")

            ft_epoch += 1
            if mrr_filter_valid_snap >= best_mrr:
                best_mrr = mrr_filter_valid_snap
                torch.save({'state_dict': model.state_dict(), 'epoch': ft_epoch, 'args': args}, save_state_file)
            else:
                if ft_epoch > 3:
                    break

        # save the best parameter in model-tc
        previous_param = [param.detach().clone() for param in model.parameters()]
        end_ft_time = time.perf_counter()
        total_train_time = total_train_time + (end_ft_time-start_ft_time)
        # step 4: 加载当前模型并测试
        # load current model
        tmp_checkpoint = torch.load(save_state_file, map_location='cpu')
        model.load_state_dict(tmp_checkpoint['state_dict'], strict=False)
        model.to(device)
        model.eval()
        with torch.no_grad():
            test_triple = test_snap.target_triplets  # 不含逆四元组
            final_score, _ = model(test_history_glist, test_triple)
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
                final_score, test_triple,
                num_nodes)
            print("Continual Model : ***test mrr*** ", mrr_filter_snap)

            torch.cuda.empty_cache()
            # 历史记忆和历史频率强化
            test_triples_input = test_tensor.cpu().numpy()
            src_idx_l = test_triples_input[:, 0]
            e_idx_l = test_triples_input[:, 1]
            cut_time_l = test_triples_input[:, 3]
            encoded_mask_his_mem = torch.zeros_like(test_pred).to(device)
            encoded_mask_fre = torch.zeros_like(test_pred).to(device)
            for j in range(len(src_idx_l)):
                if len(test_history_entity_dic[src_idx_l[j], e_idx_l[j]].cache) == 0:
                    continue
                else:
                    key_o = list(test_history_entity_dic[src_idx_l[j], e_idx_l[j]].cache.keys())
                    value_freq = list(test_history_entity_dic[src_idx_l[j], e_idx_l[j]].cache.values())
                    value_t, freq_t = zip(*value_freq)
                    value_t, tail_fre = torch.tensor(value_t).to(device), torch.tensor(freq_t).to(device)
                    key_o = torch.tensor(key_o).to(device)
                    value_delta_t = (cut_time_l[j] - value_t).to(device)
                    # if value_delta_t.max() > 0:
                    encoded_mask_his_mem[j][key_o] = (torch.exp(1 - value_delta_t / value_delta_t.max()) - 0.9) * 3
                    encoded_mask_fre[j][key_o] = (torch.log(tail_fre + 1) * 0.5).to(device)
                    # encoded_mask_his_mem[i][key_o] = 10

            final_score_enh = final_score + encoded_mask_his_mem + encoded_mask_fre
            mrr_filter_snap_enh, mrr_snap_enh, rank_raw_enh, rank_filter_enh = utils.get_total_rank(
                final_score_enh, test_triple,
                num_nodes)
            print("Continual Model (history enhance) : ***test mrr*** ", mrr_filter_snap_enh)

        # step 5: 蒸馏学习
        # 数据准备
        model.eval()
        # model_initial.eval()
        start_distill_time1 = time.perf_counter()
        with torch.no_grad():
            valid_triple = valid_snap.target_triplets  # 不含逆四元组
            val_final_score_c, _ = model(valid_history_glist, valid_triple)
            mrr_filter_valid_snap_c, mrr_valid_snap_c, val_rank_raw_c, val_rank_filter_c = utils.get_total_rank(
                val_final_score_c, valid_triple,
                num_nodes)

            ft_triple = ft_snap.target_triplets  # 不含逆四元组
            ft_final_score_c, _ = model(ft_history_glist, ft_triple)
            mrr_filter_ft_snap_c, mrr_ft_snap_c, ft_rank_raw_c, ft_rank_filter_c = utils.get_total_rank(
                ft_final_score_c, ft_triple,
                num_nodes)
        end_distill_time1 = time.perf_counter()
        total_train_time = total_train_time + (end_distill_time1 - start_distill_time1)
        # 计算预训练模型与持续训练模型在测试集的推理差异
        mrr_diff_raw = (1 / rank_test_raw_init) - (1 / rank_raw)
        mrr_diff_time = (1 / rank_test_filter_init) - (1 / rank_filter)
        diff_mask_raw = mrr_diff_raw > 0
        diff_mask_time = mrr_diff_time > 0

        # 计算预训练模型与持续训练模型在验证集的推理差异
        valid_rank_diff_time = -(
                boosting_cal_entropy(val_final_score_init, valid_tensor[:, 2]) - boosting_cal_entropy(
            val_final_score_c, valid_tensor[:, 2]))
        # valid_rank_diff_time = (1 / val_rank_filter_init) - (1 / val_rank_filter_c)
        valid_diff_mask_time_i = valid_rank_diff_time > 0  # 初始模型表现更优的样本
        valid_diff_mask_time_c = valid_rank_diff_time < 0  # 持续训练模型表现更优的样本
        # 计算预训练模型与持续训练模型在微调数据集的推理差异
        ft_rank_diff_time = -(boosting_cal_entropy(ft_final_score_init, ft_tensor[:, 2]) - boosting_cal_entropy(
            ft_final_score_c, ft_tensor[:, 2]))
        # ft_rank_diff_time = (1 / ft_rank_filter_init) - (1 / ft_rank_filter_c)
        ft_diff_mask_time_i = ft_rank_diff_time > 0  # 初始模型表现更优的样本
        ft_diff_mask_time_c = ft_rank_diff_time < 0  # 持续训练模型表现更优的样本

        # 计算蒸馏权重
        init_weight_val = mrr_filter_valid_snap_init / (mrr_filter_valid_snap_init + mrr_filter_valid_snap_c)
        continue_weight_val = mrr_filter_valid_snap_c / (mrr_filter_valid_snap_init + mrr_filter_valid_snap_c)
        init_weight_ft = mrr_filter_ft_snap_init / (mrr_filter_ft_snap_init + mrr_filter_ft_snap_c)
        continue_weight_ft = mrr_filter_ft_snap_c / (mrr_filter_ft_snap_init + mrr_filter_ft_snap_c)
        print(f"init_weight_val: {init_weight_val}, continue_weight_val: {continue_weight_val}")
        print(f"init_weight_ft: {init_weight_ft}, continue_weight_ft: {continue_weight_ft}")

        # 根据权重计算蒸馏模型的初始参数
        sdA = copy.deepcopy(model.state_dict())
        model.load_state_dict(init_checkpoint['state_dict'], strict=False)
        sdB = copy.deepcopy(model.state_dict())
        sd_avg = model.state_dict()  # 可直接复用结构
        # 参数加权平均
        for k in sd_avg.keys():
            sd_avg[k] = continue_weight_val * sdA[k] + init_weight_val * sdB[k]
            # ablation w/o adaptive weight
            # sd_avg[k] = (sdA[k] + sdB[k]) / 2.0
        # 加载新参数
        model.load_state_dict(sd_avg)

        # tmp_checkpoint = torch.load(model_state_file_distill, map_location='cpu')
        # model.load_state_dict(tmp_checkpoint['state_dict'], strict=False)
        optimizer_distill = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=0)

        # 蒸馏训练
        best_mrr_distill = 0
        best_loss = 1000
        count = 0
        distill_epoch, total_losses = 0, []
        start_distill_time2 = time.perf_counter()
        while distill_epoch < args.ft_epochs:
            model.train()
            valid_triple = valid_snap.target_triplets  # 不含逆四元组
            val_final_score_distill, _ = model(valid_history_glist, valid_triple)
            ft_triple = ft_snap.target_triplets  # 不含逆四元组
            ft_final_score_distill, cl_loss = model(ft_history_glist, ft_triple)
            pred_loss = model.get_loss(ft_final_score_distill, ft_triple[:, 1])
            loss_ft = pred_loss + cl_loss
            distill_loss1_val = torch.zeros(1).cuda().to(device)
            distill_loss2_val = torch.zeros(1).cuda().to(device)
            distill_loss1_ft = torch.zeros(1).cuda().to(device)
            distill_loss2_ft = torch.zeros(1).cuda().to(device)
            # ablation w/o adaptive temperature
            # T_tch1_val = 1
            # T_tch2_val = 1
            # T_tch1_ft = 1
            # T_tch2_ft = 1
            T = 1
            T_tch1_val = cal_entropy1(val_final_score_init, args.temperature)[valid_diff_mask_time_i]
            T_tch2_val = cal_entropy1(val_final_score_c, args.temperature)[valid_diff_mask_time_c]
            T_tch1_ft = cal_entropy1(ft_final_score_init, args.temperature)[ft_diff_mask_time_i]
            T_tch2_ft = cal_entropy1(ft_final_score_c, args.temperature)[ft_diff_mask_time_c]

            # ablation w/o adaptive sample
            # T_tch1_val = cal_entropy1(val_final_score_init, args.temperature)
            # T_tch2_val = cal_entropy1(val_final_score_c, args.temperature)
            # T_tch1_ft = cal_entropy1(ft_final_score_init, args.temperature)
            # T_tch2_ft = cal_entropy1(ft_final_score_c, args.temperature)

            if torch.sum(valid_diff_mask_time_i).item() > 0:
                soft_teacher1_val = F.softmax(val_final_score_init[valid_diff_mask_time_i].detach() / T_tch1_val,
                                              dim=1)
                soft_student1_val = F.log_softmax(val_final_score_distill[valid_diff_mask_time_i] / T_tch1_val, dim=1)
                distill_loss1_val = F.kl_div(soft_student1_val, soft_teacher1_val, reduction='batchmean')
            if torch.sum(valid_diff_mask_time_c).item() > 0:
                soft_teacher2_val = F.softmax(val_final_score_c[valid_diff_mask_time_c].detach() / T_tch2_val, dim=1)
                soft_student2_val = F.log_softmax(val_final_score_distill[valid_diff_mask_time_c] / T_tch2_val, dim=1)
                distill_loss2_val = F.kl_div(soft_student2_val, soft_teacher2_val, reduction='batchmean')

            if torch.sum(ft_diff_mask_time_i).item() > 0:
                soft_teacher1_ft = F.softmax(ft_final_score_init[ft_diff_mask_time_i].detach() / T_tch1_ft, dim=1)
                soft_student1_ft = F.log_softmax(ft_final_score_distill[ft_diff_mask_time_i] / T_tch1_ft, dim=1)
                distill_loss1_ft = F.kl_div(soft_student1_ft, soft_teacher1_ft, reduction='batchmean')
            if torch.sum(ft_diff_mask_time_c).item() > 0:
                soft_teacher2_ft = F.softmax(ft_final_score_c[ft_diff_mask_time_c].detach() / T_tch2_ft, dim=1)
                soft_student2_ft = F.log_softmax(ft_final_score_distill[ft_diff_mask_time_c] / T_tch2_ft, dim=1)
                distill_loss2_ft = F.kl_div(soft_student2_ft, soft_teacher2_ft, reduction='batchmean')
            if torch.sum(valid_diff_mask_time_i).item() <= 0 and torch.sum(
                    valid_diff_mask_time_c).item() <= 0 and torch.sum(ft_diff_mask_time_i).item() <= 0 and torch.sum(
                ft_diff_mask_time_c).item() <= 0:
                break

            # ablation w/o adaptive sample
            # soft_teacher1_val = F.softmax(val_final_score_init.detach() / T_tch1_val,
            #                               dim=1)
            # soft_student1_val = F.log_softmax(val_final_score_distill / T_tch1_val, dim=1)
            # distill_loss1_val = F.kl_div(soft_student1_val, soft_teacher1_val, reduction='batchmean')
            #
            # soft_teacher2_val = F.softmax(val_final_score_c.detach() / T_tch2_val, dim=1)
            # soft_student2_val = F.log_softmax(val_final_score_distill / T_tch2_val, dim=1)
            # distill_loss2_val = F.kl_div(soft_student2_val, soft_teacher2_val, reduction='batchmean')
            #
            # soft_teacher1_ft = F.softmax(ft_final_score_init.detach() / T_tch1_ft, dim=1)
            # soft_student1_ft = F.log_softmax(ft_final_score_distill / T_tch1_ft, dim=1)
            # distill_loss1_ft = F.kl_div(soft_student1_ft, soft_teacher1_ft, reduction='batchmean')
            #
            # soft_teacher2_ft = F.softmax(ft_final_score_c.detach() / T_tch2_ft, dim=1)
            # soft_student2_ft = F.log_softmax(ft_final_score_distill / T_tch2_ft, dim=1)
            # distill_loss2_ft = F.kl_div(soft_student2_ft, soft_teacher2_ft, reduction='batchmean')


            distill_loss = (init_weight_val * distill_loss1_val + continue_weight_val * distill_loss2_val +
                            init_weight_ft * distill_loss1_ft + continue_weight_ft * distill_loss2_ft)
            # distill_loss = distill_loss1_val + distill_loss1_ft
            total_loss = distill_loss * args.distill_weight + loss_ft
            total_losses.append(total_loss.item())
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer_distill.step()
            optimizer_distill.zero_grad()

            model.eval()
            valid_score_distill, _ = model(valid_history_glist, valid_triple)
            mrr_filter_valid_snap, mrr_valid_snap, rank_valid_raw, rank_valid_filter = utils.get_total_rank(
                valid_score_distill, valid_triple,
                num_nodes)
            print(f"distill_epoch:{distill_epoch}, mrr_filter_valid_snap:{mrr_filter_valid_snap}, "
                  f"total_loss:{total_loss.item():.3f}, "
                  f"distill_loss:{distill_loss.item():.3f}, distill_loss1:{distill_loss1_val.item():.3f},"
                  f"distill_loss2:{distill_loss2_val.item():.3f}")
            # update best_mrr_distill
            distill_epoch += 1
            if mrr_filter_valid_snap <= best_mrr_distill:
                count += 1
                if count > 2:
                    break
            else:
                count = 0
                best_mrr_distill = mrr_filter_valid_snap
                torch.save({'state_dict': model.state_dict(), 'epoch': -1}, model_state_file_distill)
        end_distill_time2 = time.perf_counter()
        total_train_time = total_train_time + (end_distill_time2-start_distill_time2)
        torch.cuda.empty_cache()
        tmp_checkpoint = torch.load(model_state_file_distill, map_location='cpu')
        model.load_state_dict(tmp_checkpoint['state_dict'], strict=False)
        model.eval()
        # 开始测试蒸馏模型
        with torch.no_grad():
            test_triple = test_snap.target_triplets  # 不含逆四元组
            test_score_distill, _ = model(test_history_glist, test_triple)
            test_dsitill_score_enh = test_score_distill + encoded_mask_his_mem + encoded_mask_fre

            mrr_filter_distill_snap, mrr_distill_snap, rank_distill_raw, rank_distill_filter = utils.get_total_rank(
                test_score_distill, test_triple,
                num_nodes)
            mrr_filter_distill_snap_enh, mrr_distill_snap_enh, rank_distill_raw_enh, rank_distill_filter_enh = utils.get_total_rank(
                test_dsitill_score_enh, test_triple,
                num_nodes)
            print("Continual Model (after distill) : ***test mrr*** ", mrr_filter_distill_snap)
            print("Continual Model (after distill enhance) : ***test mrr*** ", mrr_filter_distill_snap_enh)

        # 更新历史记录
        ft_triples_input = ft_tensor.cpu().numpy()
        src_l_cut = ft_triples_input[:, 0]
        e_l_cut = ft_triples_input[:, 1]
        dst_l_cut = ft_triples_input[:, 2]
        ts_l_cut = ft_triples_input[:, 3]
        valid_triples_input = valid_tensor.cpu().numpy()
        src_l_cut_val = valid_triples_input[:, 0]
        e_l_cut_val = valid_triples_input[:, 1]
        dst_l_cut_val = valid_triples_input[:, 2]
        ts_l_cut_val = valid_triples_input[:, 3]
        test_triples_input = test_tensor.cpu().numpy()
        src_l_cut_test = test_triples_input[:, 0]
        e_l_cut_test = test_triples_input[:, 1]
        dst_l_cut_test = test_triples_input[:, 2]
        ts_l_cut_test = test_triples_input[:, 3]
        # 更新历史记录
        for i in range(len(src_l_cut_test)):
            test_history_entity_dic[(src_l_cut_test[i], e_l_cut_test[i])].put(dst_l_cut_test[i], ts_l_cut_test[i])
        for i in range(len(src_l_cut_test)):
            test_history_entity_dic[(dst_l_cut_test[i], e_l_cut_test[i] + num_rels)].put(src_l_cut_test[i],
                                                                                         ts_l_cut_test[i])
        for element in dst_l_cut_test:
            count_dict_o[element] = count_dict_o.get(element, 0) + 1
            count_dict_so[element] = count_dict_so.get(element, 0) + 1
        for element in src_l_cut_test:
            count_dict_so[element] = count_dict_so.get(element, 0) + 1

        # step 7: update history glist and prepare inputs
        ft_history_glist.pop(0)
        ft_history_glist.append(ft_snap)
        valid_history_glist.pop(0)
        valid_history_glist.append(valid_snap)
        test_history_glist.pop(0)
        test_history_glist.append(test_snap)
        valid_snap = ft_snap
        ft_snap = test_snap
        valid_tensor = ft_tensor
        ft_tensor = test_tensor

        # step 8: save results
        # used to global statistic
        ranks_raw_init.append(rank_test_raw_init)
        ranks_filter_init.append(rank_test_filter_init)
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_enh.append(rank_raw_enh)
        ranks_filter_enh.append(rank_filter_enh)
        ranks_raw_distill.append(rank_distill_raw)
        ranks_filter_distill.append(rank_distill_filter)
        ranks_raw_distill_enh.append(rank_distill_raw_enh)
        ranks_filter_distill_enh.append(rank_distill_filter_enh)
        test_src, test_dst, test_r = test_triple.t().cpu().numpy()
        for k in range(len(test_dst)):
            if count_dict_o.get(test_dst[k], 0) < 10:
                ranks_raw_cold_o.append(rank_distill_raw_enh[k])
                ranks_filter_cold_o.append(rank_distill_filter_enh[k])
            if count_dict_so.get(test_dst[k], 0) < 10:
                ranks_raw_cold_so.append(rank_distill_raw_enh[k])
                ranks_filter_cold_so.append(rank_distill_filter_enh[k])

        # used to show slide results
        mrr_raw_list_init.append(mrr_test_snap_init)
        mrr_filter_list_init.append(mrr_filter_test_snap_init)
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        mrr_raw_list_enh.append(mrr_snap_enh)
        mrr_filter_list_enh.append(mrr_filter_snap_enh)
        mrr_raw_list_distill.append(mrr_distill_snap)
        mrr_filter_list_distill.append(mrr_filter_distill_snap)
        mrr_raw_list_distill_enh.append(mrr_distill_snap_enh)
        mrr_filter_list_distill_enh.append(mrr_filter_distill_snap_enh)

        # 计算预训练模型与持续训练模型的推理差异
        rand_diff_raw = rank_test_raw_init - rank_raw
        rand_diff_time = rank_test_filter_init - rank_filter
        diff_mask_raw = rand_diff_raw < 0
        diff_mask_time = rand_diff_time < 0
        diff_mask_raw1 = (rand_diff_raw < -10) & (rank_test_raw_init < 10)
        diff_mask_time1 = (rand_diff_time < -10) & (rank_test_filter_init < 10)
        total_rank_diff_raw = torch.sum(diff_mask_raw * rand_diff_raw).item()
        total_rank_diff_time = torch.sum(diff_mask_time * rand_diff_time).item()
        rank_diff_raw_list.append(total_rank_diff_raw)
        rank_diff_time_list.append(total_rank_diff_time)
        diff_raw_entity_list.append(torch.sum(diff_mask_raw).item())
        diff_time_entity_list.append(torch.sum(diff_mask_time).item())
        diff_raw_entity_list1.append(torch.sum(diff_mask_raw1).item())
        diff_time_entity_list1.append(torch.sum(diff_mask_time1).item())
        instance_num_list.append(len(test_tensor))

        init_sups_num.append(torch.sum(mrr_diff_time >= 0).item())
        conti_sups_num.append(torch.sum(mrr_diff_time < 0).item())
        distill_init_diff_time = (1 / rank_distill_filter_enh) - (1 / rank_test_filter_init)
        distill_conti_diff_time = (1 / rank_distill_filter_enh) - (1 / rank_filter)
        distill_enh_sups_both.append(torch.sum((distill_init_diff_time >= 0) & (distill_conti_diff_time >= 0)).item())
        distill_enh_sups_init.append(torch.sum(distill_init_diff_time >= 0).item())
        distill_enh_sups_conti.append(torch.sum(distill_conti_diff_time >= 0).item())
        distill_enh_lag_both.append(torch.sum((distill_init_diff_time < 0) & (distill_conti_diff_time < 0)).item())
        torch.cuda.empty_cache()

    print("\ninit model")
    mrr_raw, hit_raw = utils.stat_ranks(ranks_raw_init, "raw", "test")
    mrr_filter, hit_filter = utils.stat_ranks(ranks_filter_init, "filter", "test")
    print("\ncontinuous model")
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent", "test")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent", "test")
    print("\ncontinuous model (history enhance)")
    mrr_raw, hit_raw = utils.stat_ranks(ranks_raw_enh, "raw", "test")
    mrr_filter, hit_filter = utils.stat_ranks(ranks_filter_enh, "filter", "test")
    print("\ndistillation model")
    mrr_raw = utils.stat_ranks(ranks_raw_distill, "raw", "test")
    mrr_filter = utils.stat_ranks(ranks_filter_distill, "filter", "test")
    print("\ndistillation_enh model  (history enhance)")
    mrr_raw = utils.stat_ranks(ranks_raw_distill_enh, "raw", "test")
    mrr_filter = utils.stat_ranks(ranks_filter_distill_enh, "filter", "test")
    print(f"DiMNet训练时间: {total_train_time:.4f} 秒")


    print("\ncold start o")
    if len(ranks_raw_cold_o) > 0:
        mrr_raw = utils.stat_ranks_clod(ranks_raw_cold_o, "raw", "test")
        mrr_filter = utils.stat_ranks_clod(ranks_filter_cold_o, "filter", "test")
    print("\ncold start so")
    if len(ranks_raw_cold_so) > 0:
        mrr_raw = utils.stat_ranks_clod(ranks_raw_cold_so, "raw", "test")
        mrr_filter = utils.stat_ranks_clod(ranks_filter_cold_so, "filter", "test")

    if mode == "test":
        print("\noffline mrr raw list")
        for i in range(len(mrr_filter_list_init)):
            print(mrr_filter_list_init[i])

        print("\ncontinuous model mrr time list")
        for i in range(len(mrr_filter_list)):
            print(mrr_filter_list[i])

        print("\ncontinuous model mrr time list (distillation_enh)")
        for i in range(len(mrr_filter_list_distill_enh)):
            print(mrr_filter_list_distill_enh[i])

        mrr_list_id = np.arange(len(mrr_raw_list_init))
        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_raw_list_init))
        decay = np.sum((np.array(mrr_raw_list_init) - y_bar) * (mrr_list_id - x_bar)) / np.sum((mrr_list_id - x_bar) ** 2)
        print(f"\n\noffline raw mrr decay  (continuous): {decay}")
        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_filter_list_init))
        decay = np.sum((np.array(mrr_filter_list_init) - y_bar) * (mrr_list_id - x_bar)) / np.sum((mrr_list_id - x_bar) ** 2)
        print(f"offline time mrr decay  (continuous): {decay}")

        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_raw_list))
        decay = np.sum((np.array(mrr_raw_list) - y_bar) * (mrr_list_id - x_bar)) / np.sum((mrr_list_id - x_bar) ** 2)
        print(f"\nonline raw mrr decay  (continuous): {decay}")
        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_filter_list))
        decay = np.sum((np.array(mrr_filter_list) - y_bar) * (mrr_list_id - x_bar)) / np.sum((mrr_list_id - x_bar) ** 2)
        print(f"online time mrr decay  (continuous): {decay}")

        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_raw_list_enh))
        decay = np.sum((np.array(mrr_raw_list_enh) - y_bar) * (mrr_list_id - x_bar)) / np.sum((mrr_list_id - x_bar) ** 2)
        print(f"\nonline raw mrr decay (continuous enhance): {decay}")
        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_filter_list_enh))
        decay = np.sum((np.array(mrr_filter_list_enh) - y_bar) * (mrr_list_id - x_bar)) / np.sum((mrr_list_id - x_bar) ** 2)
        print(f"online time mrr decay (continuous enhance): {decay}")

        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_raw_list_distill))
        decay = np.sum((np.array(mrr_raw_list_distill) - y_bar) * (mrr_list_id - x_bar)) / np.sum(
            (mrr_list_id - x_bar) ** 2)
        print(f"\nonline raw mrr decay  (distillation): {decay}")
        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_filter_list_distill))
        decay = np.sum((np.array(mrr_filter_list_distill) - y_bar) * (mrr_list_id - x_bar)) / np.sum(
            (mrr_list_id - x_bar) ** 2)
        print(f"online time mrr decay (distillation): {decay}")

        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_raw_list_distill_enh))
        decay = np.sum((np.array(mrr_raw_list_distill_enh) - y_bar) * (mrr_list_id - x_bar)) / np.sum(
            (mrr_list_id - x_bar) ** 2)
        print(f"\nonline raw mrr decay  (distillation_enh): {decay}")
        x_bar = np.mean(mrr_list_id)
        y_bar = np.mean(np.array(mrr_filter_list_distill_enh))
        decay = np.sum((np.array(mrr_filter_list_distill_enh) - y_bar) * (mrr_list_id - x_bar)) / np.sum(
            (mrr_list_id - x_bar) ** 2)
        print(f"online time mrr decay (distillation_enh): {decay}")

        print("\nrank_diff_raw_list. total ranks of init_ranks - conti_ranks")
        for j in range(len(rank_diff_raw_list)):
            print(rank_diff_raw_list[j])
        print("\nrank_diff_time_list. total ranks of init_ranks - conti_ranks")
        for j in range(len(rank_diff_time_list)):
            print(rank_diff_time_list[j])
        print("\ndiff_raw_entity_list. the number of init_rank - conti_rank < 0")
        for j in range(len(diff_raw_entity_list)):
            print(diff_raw_entity_list[j], instance_num_list[j])
        print("\ndiff_time_entity_list. the number of init_rank - conti_rank < 0")
        for j in range(len(diff_time_entity_list)):
            print(diff_time_entity_list[j], instance_num_list[j])
        print("\ndiff_raw_entity_list1. the number of init_rank - conti_rank < -10 & init_rank < 10 and sample number")
        for j in range(len(diff_raw_entity_list1)):
            print(diff_raw_entity_list1[j], instance_num_list[j])
        print("\ndiff_time_entity_list1. the number of init_rank - conti_rank < -10 & init_rank < 10 and sample number")
        for j in range(len(diff_time_entity_list1)):
            print(diff_time_entity_list1[j], instance_num_list[j])

        dump_file_name = 'saved_data' + f"{args.temperature}-{args.distill_weight}"
        obj = {"init_sups_num": init_sups_num, "conti_sups_num": conti_sups_num,
               "distill_enh_sups_both": distill_enh_sups_both, "distill_enh_lag_both": distill_enh_lag_both,
               "distill_enh_sups_init": distill_enh_sups_init, "distill_enh_sups_conti": distill_enh_sups_conti,
               "mrr_filter_list_init": mrr_filter_list_init, "mrr_filter_list": mrr_filter_list,
               "mrr_filter_list_enh": mrr_filter_list_enh, "mrr_filter_list_distill": mrr_filter_list_distill,
               "mrr_filter_list_distill_enh": mrr_filter_list_distill_enh,
               "ranks_filter_init": ranks_filter_init, "ranks_filter": ranks_filter,
               "ranks_filter_enh": ranks_filter_enh, "ranks_filter_distill_enh": ranks_filter_distill_enh,
               "instance_num_list": instance_num_list}
        # with open(dump_file_name, "wb") as f:
        #     pickle.dump(obj, f)

    return mrr_raw, mrr_filter


@torch.no_grad()
def test(model, test_list, num_rels, num_nodes, mode="train", model_name=None):
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        print("Load Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(model_name, checkpoint['epoch'],
                                                                                checkpoint[
                                                                                    'args']))  # use best stat checkpoint
        print(
            "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart test\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    idx = [_ for _ in range(len(test_list))]  # timestamps index [0,1,2,3,...,n]

    model.eval()
    rankings = []
    raw_rankings = []

    for future_sample_id in idx:
        if future_sample_id < args.history_len: continue
        # future_sample as the future graph
        future_graph = test_list[future_sample_id]
        # future_triple : [num_edges, 3] (format: h,t,r)
        # Note that we are not add reverse edges in 'future_triple' as query query_triple in test phase
        future_triple = future_graph.target_triplets
        future_triple_reverse = future_triple[:, [1, 0, 2]]
        future_triple_reverse[:, 2] += num_rels

        # get history graph list
        history_list = test_list[future_sample_id - args.history_len: future_sample_id]

        # time_filter data only contains the future triple without reverse edges for mask generation
        time_filter_data = {
            'num_nodes': num_nodes,
            'edge_index': torch.stack([future_triple[:, 0], future_triple[:, 1]]),
            'edge_type': future_triple[:, 2]
        }

        batch = future_triple  # all future tirples is an only batch

        triple = torch.cat([future_triple, future_triple_reverse])
        # triple = future_triple
        pred, _ = model(history_list, triple)

        t_pred, h_pred = torch.chunk(pred, 2, dim=0)
        # t_pred = pred
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        # time_filter Rank
        timef_t_mask, timef_h_mask = utils.strict_negative_mask(time_filter_data, batch)
        t_ranking, timef_t_ranking = utils.compute_ranking(t_pred, pos_t_index, timef_t_mask)
        h_ranking, timef_h_ranking = utils.compute_ranking(h_pred, pos_h_index, timef_h_mask)
        rankings += [timef_t_ranking, timef_h_ranking]
        # rankings += timef_t_ranking
        rankings.append(timef_t_ranking)
        # raw_rankings.append(t_ranking)

        # This is the end of prediction at 'future_sample_id' time
    # This is the end of prediction at test_set
    # mrr, mrr_raw_list = utils.stat_ranks(raw_rankings, "raw", "test")
    mrr, mrr_filter_list = utils.stat_ranks(rankings, "filter", "test")
    # if mode == "test":
    #     print(f"验证集第一个snapshot,MRR raw: {mrr_raw_list[0]}, MRR filter: {mrr_filter_list[0]}")
    #     print("mrr_filter_list: ")
    #     for i in range(len(mrr_filter_list)):
    #         print(mrr_filter_list[i])
    all_ranking = torch.cat(rankings)

    metrics_dict = dict()
    for metric in args.metric:
        if metric == "mr":
            score = all_ranking.float().mean()
        elif metric == "mrr":
            score = (1 / all_ranking.float()).mean()
        elif metric.startswith("hits@"):
            values = metric[5:].split("_")
            threshold = int(values[0])
            score = (all_ranking <= threshold).float().mean()
        metrics_dict[metric] = score.item() * 100
    metrics_dict['time'] = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    # print(json.dumps(metrics_dict, indent=4))
    print("MRR:{:.5f}\tH1:{:.5f}\tH3:{:.5f}\tH10:{:.5f}"
          .format(metrics_dict['mrr'], metrics_dict['hits@1'], metrics_dict['hits@3'], metrics_dict['hits@10']))

    # mrr = (1 / all_ranking.float()).mean()

    return metrics_dict


if __name__ == '__main__':
    current_timestamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args, current_timestamp)
    device = utils.get_device(args)

    model_name = "len:{}-dim:{}-ly:{}-head:{}-topk:{}" \
        .format(args.history_len, args.input_dim, args.num_ly, args.num_head,
                args.topk)
    # model_name = "len_{}-dim_{}-ly_{}-head_{}-topk_{}" \
    #     .format(args.history_len, args.input_dim, args.num_ly, args.num_head,
    #             args.topk)

    model_state_file = model_name + ".pth"

    # load datasets
    data = utils.load_data(args.dataset)
    num_nodes = data.num_nodes
    num_rels = data.num_rels  # not include reverse edge type

    print("# Model ID: {}".format(model_state_file))
    print("# Sanity Check: entities: {}".format(data.num_nodes))
    print("# Sanity Check: relations: {}".format(data.num_rels))
    print("# Sanity Check: edges: {}".format(len(data.train)))

    train_list_sp, valid_list_sp, test_list_sp, train_graph_list, valid_graph_list, test_graph_list = \
        utils.generate_graph_data(data, num_nodes, num_rels, False, device)
    # Each item in the graph list is a snapshot of the graph
    # edge_index: [2, num_edges], which has added reverse edges
    # edge_type: [num_edges], which has added reverse edges
    # num_nodes: int
    # target_triplets: [num_edges, 3] (format: h,t,r)

    train_g_list = train_graph_list
    valid_g_list = train_g_list[-args.history_len:] + valid_graph_list
    test_g_list = valid_g_list[-args.history_len:] + test_graph_list

    model = DiMNet(
        dim=args.input_dim,
        num_layer=args.num_ly,
        num_relation=num_rels,
        num_node=num_nodes,
        message_func=args.message_func,
        aggregate_func=args.aggregate_func,
        short_cut=args.short_cut,
        layer_norm=args.layer_norm,
        activation="rrelu",
        history_len=args.history_len,
        topk=args.topk,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        num_head=args.num_head
    )
    model = model.to(device)
    wandb.watch(model, log='all', log_freq=500)
    print(args)
    print("datetime" + datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f'))

    # 参数总数
    total_params = sum(p.numel() for p in model.parameters())

    # 模型参数大小（字节）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 ** 2

    print(f"模型参数总数: {total_params:,}")
    print(f"模型参数占用内存: {model_size_mb:.2f} MB")

    if args.test == 1:
        test(model, test_g_list, num_rels, num_nodes, mode="test", model_name=model_state_file)
    elif args.test == 2:
        continuous_test(args, model, train_graph_list + [valid_graph_list[0]], valid_graph_list[1:], train_list_sp + [valid_list_sp[0]], valid_list_sp[1:],
                        np.array(data.train), num_nodes, num_rels, mode="valid", model_name=model_state_file)
    elif args.test == 3:
        continuous_test(args, model, train_graph_list+valid_graph_list, test_graph_list, train_list_sp+valid_list_sp,
                        test_list_sp, np.array(data.train), num_nodes, num_rels, mode="test",
                        model_name=model_state_file)
    else:
        train_and_validate(args, model, train_g_list, valid_g_list, test_g_list, num_nodes, num_rels, model_state_file)

    print(args)
    print("datetime" + datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f'))
    sys.exit()
