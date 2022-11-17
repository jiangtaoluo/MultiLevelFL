from __future__ import division

import copy
import gc
# import os

import cache_env
import content
# import gym
import numpy as np
import pandas as pd
# import psutil
import torch
# from torch.autograd import Variable

import buffer
from Fed import FedAvg
from model import Qnet
from options import args_parser
import matplotlib.pyplot as plt
import time
import train
import content
import cache
# env = gym.make('BipedalWalker-v3')
args = args_parser()

MAX_BUFFER = 500
BS_NUM = 2

# 初始化参数
MAX_EPISODES = 20000
C_BS1 = 600
C_BS2 = 600
RSU = 600
bsNum = 2
content_number = 9724
S_DIM = content_number * 4
A_DIM = 3
d_cdc = 200
d_cobs = 20

ep_hit_list = []

# 引入电影数据作为内容库
df = pd.DataFrame(pd.read_csv('content-1000.csv', header=None))
content_popu_list = df.values.tolist()
content_popu_list = list(np.ravel(content_popu_list))
# print(content_popu_list)

global_content_size = []*content_number
global_content_size = content.gcontent_size(content_number, global_content_size)

# content_popu_list = content.content_popu(content_number, global_content_size)

content_stream1 = content.request_list(content_popu_list, MAX_EPISODES)
content_stream2 = content.request_list(content_popu_list, MAX_EPISODES)

content_stream = []
content_stream_no = content_stream1 + content_stream2
content_stream.append(content_stream1)
content_stream.append(content_stream2)
# print(content_stream[0][0])
# print(content_stream[1][0])

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)

Qnet_glob = Qnet(state_dim=S_DIM, action_dim=A_DIM)
Qnet_glob.load_state_dict(torch.load('net-cocarl.pth'))

co_ep_hit = 0
co_delay = 0
co_bc = 0
co_ep_hit_ls = []
co_bc_ls = []
co_delay_ls = []

lfu_ep_hit = 0
lfu_delay = 0
lfu_bc = 0
lfu_ep_hit_ls = []
lfu_bc_ls = []
lfu_delay_ls = []

lru_ep_hit = 0
lru_delay = 0
lru_bc = 0
lru_ep_hit_ls = []
lru_bc_ls = []
lru_delay_ls = []

fifo_ep_hit = 0
fifo_delay = 0
fifo_bc = 0
fifo_ep_hit_ls = []
fifo_bc_ls = []
fifo_delay_ls = []

# opt_ep_hit = 0
# opt_delay = 0
# opt_bc = 0
# opt_ep_hit_ls = []
# opt_bc_ls = []
# opt_delay_ls = []

fl_ep_hit = 0
fl_delay = 0
fl_bc = 0
fl_ep_hit_ls = []
fl_bc_ls = []
fl_delay_ls = []

fifo_cache = []
print(content_stream_no)
for f in range(len(content_stream_no)):
    # print(content_stream_no[f])
    # print(fifo_cache)
    if content_stream_no[f] not in fifo_cache:
        fifo_delay += 200 + global_content_size[content_stream_no[f]] / 8
        fifo_bc += 1
        if len(fifo_cache) < RSU:
            fifo_cache.append(content_stream_no[f])
        else:
            fifo_cache.pop(0)
            fifo_cache.append(content_stream_no[f])
    else:
        fifo_ep_hit += 1
        fifo_delay += global_content_size[content_stream_no[f]] / 8
    if (f + 1) % 100 == 0:
        print(fifo_ep_hit / (f + 1), fifo_bc / (f + 1), fifo_delay / (f + 1))
        fifo_ep_hit_ls.append(fifo_ep_hit / (f + 1))
        fifo_bc_ls.append(fifo_bc / (f + 1))
        fifo_delay_ls.append(fifo_delay / (f + 1))


lru_cache = []
for f in range(len(content_stream_no)):
    # print(content_stream_no[f])
    # print(fifo_cache)
    if content_stream_no[f] not in lru_cache:
        lru_delay += 200 + global_content_size[content_stream_no[f]] / 8
        lru_bc += 1
        if len(lru_cache) < RSU:
            lru_cache.append(content_stream_no[f])
        else:
            lru_cache.pop(0)
            lru_cache.append(content_stream_no[f])
    else:
        lru_ep_hit += 1
        lru_delay += global_content_size[content_stream_no[f]] / 8
        lru_cache.append(lru_cache.pop(lru_cache.index(content_stream_no[f])))# 弹出后插入到最近刚刚访问的一端
    if (f + 1) % 100 == 0:
        print(lru_ep_hit / (f + 1), lru_bc / (f + 1), lru_delay / (f + 1))
        lru_ep_hit_ls.append(lru_ep_hit / (f + 1))
        lru_bc_ls.append(lru_bc / (f + 1))
        lru_delay_ls.append(lru_delay / (f + 1))


bad, bad_i = 1 << 31 - 1, 0
lfu_cache = {}
for i, f in enumerate(content_stream_no):
    # print(i, f)
    if f not in lfu_cache:
        lfu_delay += 200 + global_content_size[f] / 8
        lfu_bc += 1
        if len(lfu_cache) < RSU:  # 内存还未满
            lfu_cache[f] = 1
        else:
            for j, v in lfu_cache.items():
                if v < bad:
                    bad, bad_i = v, j
            lfu_cache.pop(bad_i)
            lfu_cache[f] = 1
            bad, bad_i = 2 ** 32 - 1, 0
    else:
        lfu_delay += global_content_size[f] / 8
        lfu_ep_hit += 1
        lfu_cache[f] += 1
    if (i + 1) % 100 == 0:
        print(lfu_ep_hit / (i + 1), lfu_bc / (i + 1), lfu_delay / (i + 1))
        lfu_ep_hit_ls.append(lfu_ep_hit / (i + 1))
        lfu_bc_ls.append(lfu_bc / (i + 1))
        lfu_delay_ls.append(lfu_delay / (i + 1))


s_c = []
s_c1, s_c2 = [0]*content_number, [0]*content_number
s_c.append(s_c1)
s_c.append(s_c2)

C_BS = []
C_BS.append(C_BS1)
C_BS.append(C_BS2)


for _ep in range(MAX_EPISODES):
    # print('EPISODE :- ', _ep)
    for bsNum in range(BS_NUM):
        # print(C_BS[bsNum])
        # print(content_stream[0][0])
        s_r_list = content_stream[bsNum][_ep]
    # for i in range(len(s_r_list)):
        s_r = [0]*content_number
        f_index = s_r_list
        s_r[f_index] = 1
        content_popu_list[f_index] += 1
        if bsNum == 0:
            neibor_cachestate = s_c[1]
        else:
            neibor_cachestate = s_c[0]
        if s_c[bsNum][f_index] == 1:
            co_ep_hit += 1
            co_delay += global_content_size[f_index] / 8
            continue
        else:
            if C_BS[bsNum] - 1 >= 0:
                C_BS[bsNum] = C_BS[bsNum] - 1
                state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                action = train.Trainer(S_DIM, A_DIM, 1, Qnet_glob, Qnet_glob).get_action(state, f_index, neibor_cachestate)
                # print(action)
                if action[1] == 1:
                    co_delay += 20 + global_content_size[f_index] / 8
                    s_c[bsNum][f_index] = 1
                if action[2] == 1:
                    co_delay += 200 + global_content_size[f_index] / 8
                    co_bc += 1
                    s_c[bsNum][f_index] = 1
                continue
            if C_BS[bsNum] == 0:
                state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                action = train.Trainer(S_DIM, A_DIM, 1, Qnet_glob, Qnet_glob).get_action(state, f_index, neibor_cachestate)
                # print(action)
                if action[1] == 1:
                    co_delay += 20 + global_content_size[f_index] / 8
                if action[2] == 1:
                    co_delay += 200 + global_content_size[f_index] / 8
                    co_bc += 1
                if action[0] == 1:
                    minpopu = 0
                    for i in range(content_number - 1, -1, -1):
                        if s_c[bsNum][i] > 0:
                            minpopu = i
                            break;
                    s_c[bsNum][i] = 0
                    s_c[bsNum][f_index] = 1
    if (_ep + 1) % 50 == 0:
        print(co_ep_hit / ((_ep + 1) * 2), co_bc / ((_ep + 1) * 2), co_delay / ((_ep + 1) * 2))
        co_ep_hit_ls.append(co_ep_hit / ((_ep + 1) * 2))
        co_bc_ls.append(co_bc / ((_ep + 1) * 2))
        co_delay_ls.append(co_delay / ((_ep + 1) * 2))
    # check memory consumption and clear memory
    gc.collect()
print('Completed episodes')

s_c = []
s_c1, s_c2 = [0]*content_number, [0]*content_number
s_c.append(s_c1)
s_c.append(s_c2)

C_BS = []
C_BS1 = RSU
C_BS2 = RSU
C_BS.append(C_BS1)
C_BS.append(C_BS2)
Qnet_glob_fl = Qnet(state_dim=S_DIM, action_dim=A_DIM)
Qnet_glob_fl.load_state_dict(torch.load('net-fl.pth'))
for _ep in range(MAX_EPISODES):
    # print('EPISODE :- ', _ep)
    for bsNum in range(BS_NUM):
        # print(content_stream[0][0])
        s_r_list = content_stream[bsNum][_ep]
        # for i in range(len(s_r_list)):
        s_r = [0]*content_number
        f_index = s_r_list
        s_r[f_index] = 1
        content_popu_list[f_index] += 1
        if bsNum == 0:
            neibor_cachestate = s_c[1]
        else:
            neibor_cachestate = s_c[0]
        if s_c[bsNum][f_index] == 1:
            fl_ep_hit += 1
            fl_delay += global_content_size[f_index] / 8
            continue
        else:
            if C_BS[bsNum] - 1 >= 0:
                C_BS[bsNum] = C_BS[bsNum] - 1
                state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                action = train.Trainer(S_DIM, A_DIM, 1, Qnet_glob, Qnet_glob).get_action(state, f_index, neibor_cachestate)
                # print(action)
                if action[1] == 1:
                    fl_delay += 20 + global_content_size[f_index] / 8
                    s_c[bsNum][f_index] = 1
                if action[2] == 1:
                    fl_delay += 200 + global_content_size[f_index] / 8
                    fl_bc += 1
                    s_c[bsNum][f_index] = 1
                continue
            if C_BS[bsNum] == 0:
                state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                action = train.Trainer(S_DIM, A_DIM, 1, Qnet_glob, Qnet_glob).get_action(state, f_index, neibor_cachestate)
                # print(action)
                if action[1] == 1:
                    fl_delay += 20 + global_content_size[f_index] / 8
                if action[2] == 1:
                    fl_delay += 200 + global_content_size[f_index] / 8
                    fl_bc += 1
                if action[0] == 1:
                    minpopu = 0
                    for i in range(content_number - 1, -1, -1):
                        if s_c[bsNum][i] > 0:
                            minpopu = i
                            break;
                    s_c[bsNum][i] = 0
                    s_c[bsNum][f_index] = 1
    if (_ep + 1) % 50 == 0:
        print(fl_ep_hit / ((_ep + 1) * 2), fl_bc / ((_ep + 1) * 2), fl_delay / ((_ep + 1) * 2))
        fl_ep_hit_ls.append(fl_ep_hit / ((_ep + 1) * 2))
        fl_bc_ls.append(fl_bc / ((_ep + 1) * 2))
        fl_delay_ls.append(fl_delay / ((_ep + 1) * 2))
    # check memory consumption and clear memory
    gc.collect()
print('Completed episodes')


name_hit = ['co_hit_rate', 'fl_hit_rate', 'fifo_hit', 'lru_hit', 'lfu_hit']
hit = []
hit.append(co_ep_hit_ls)
hit.append(fl_ep_hit_ls)
hit.append(fifo_ep_hit_ls)
hit.append(lru_ep_hit_ls)
hit.append(lfu_ep_hit_ls)
test_hitrate = pd.DataFrame(index=name_hit, data=hit)
test_hitrate.to_csv('hit_rate.csv', encoding='gbk')

name_bc = ['co_bc', 'fl_bc', 'fifo_bc', 'lru_bc', 'lfu_bc']
bc = []
bc.append(co_bc_ls)
bc.append(fl_bc_ls)
bc.append(fifo_bc_ls)
bc.append(lru_bc_ls)
bc.append(lfu_bc_ls)
test_bc = pd.DataFrame(index=name_bc, data=bc)
test_bc.to_csv('bc.csv', encoding='gbk')

name_delay = ['co_delay', 'fl_delay', 'fifo_delay', 'lru_delay', 'lfu_delay']
delay = []
delay.append(co_delay_ls)
delay.append(fl_delay_ls)
delay.append(fifo_delay_ls)
delay.append(lru_delay_ls)
delay.append(lfu_delay_ls)
test_delay = pd.DataFrame(index=name_delay, data=delay)
test_delay.to_csv('delay.csv', encoding='gbk')