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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from random import choice

import buffer
import train
from Fed import FedAvg
from model import Qnet
from options import args_parser
import matplotlib.pyplot as plt
import time


# env = gym.make('BipedalWalker-v3')
args = args_parser()
ramList, trainer, qnet, qnet_target= [], [], [], []

MAX_BUFFER = 500
BS_NUM = 2
ram = buffer.MemoryBuffer(MAX_BUFFER)
for i in range(args.num_users * BS_NUM):
    ramList.append(ram)

# 初始化参数
MAX_EPISODES = 400
C_BS1 = 100
C_BS2 = 100
content_number = 9724
S_DIM = content_number * 4
A_DIM = 3

d_b = 20
d_p = 200

# 引入电影数据作为内容库
df = pd.DataFrame(pd.read_csv('content.csv', header=None))
content_popu_list = df.values.tolist()
content_popu_list = list(np.ravel(content_popu_list))

# print(len(content_popu_list))

global_content_size = []*content_number
global_content_size = content.gcontent_size(content_number, global_content_size)

local1_content_size, local1_content_popular = []*content_number, []*content_number
local1_content_size = copy.deepcopy(global_content_size)

# local1_content_size = content.set_zero(content_number, local1_content_size)
# local1_content_popular = content.content_popu(content_number, local1_content_size, local1_content_popular)
# C_BS1 = C_BS1 - sum(local1_content_size)
# print(C_BS1)

local2_content_size, local2_content_popular = []*content_number, []*content_number
local2_content_size = copy.deepcopy(global_content_size)
# local2_content_size = content.set_zero(content_number, local2_content_size)
# local2_content_popular = content.content_popu(content_number, local2_content_size, local2_content_popular)
# C_BS2 = C_BS2 - sum(local2_content_size)
# print(C_BS2)

localContent, C_BS, localPopularity = [], [], []
localContent.append(local1_content_size)
localContent.append(local2_content_size)
C_BS.append(C_BS1)
C_BS.append(C_BS2)
# localPopularity.append(local1_content_popular)
# localPopularity.append(local2_content_popular)

# neibor_content_size, neibor_content_popular = []*content_number, []*content_number
# neibor_content_size = copy.deepcopy(global_content_size)
# neibor_content_size = content.set_zero(content_number, neibor_content_size)
# neibor_content_popular = content.content_popu(content_number, neibor_content_size, neibor_content_popular)

s_c = []
s_c1, s_c2 = [0]*content_number, [0]*content_number
s_c.append(s_c1)
s_c.append(s_c2)

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)

for w in range(BS_NUM):
    qnet.append(Qnet(state_dim=S_DIM, action_dim=A_DIM))
    qnet_target.append(Qnet(state_dim=S_DIM, action_dim=A_DIM))
    qnet[w].train()
    qnet_target[w].train()

for i in range(BS_NUM):
    temtra = train.Trainer(S_DIM, A_DIM, ramList[i], qnet[i], qnet_target[i])
    trainer.append(temtra)

Qloss, Reward = [], []  # 每轮损失
for _ep in range(MAX_EPISODES):
    print('EPISODE :- ', _ep)
    ep_Qloss, ep_reward, ep_hit, delay = [], 0, 0, 0
    Qw_locals = []  # 网络权重
    for bsNum in range(BS_NUM):
        for worker in range(2):
            s_r_list = content.request_list(content_popu_list, 25)
            # print(s_r_list)
            for i in range(len(s_r_list)):
                s_r = [0]*content_number
                f_index = s_r_list[i]
                s_r[f_index] = 1
                Qw = None
                content_popu_list[f_index] += 1
                if bsNum == 0:
                    neibor_cachestate = s_c[1]
                else:
                    neibor_cachestate = s_c[0]
                if s_c[bsNum][f_index] == 1:
                    ep_hit += 1
                    delay += global_content_size[f_index] / 8
                    state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                    action = [0, 0, 0]
                    reward, new_state = cache_env.step(action)
                    new_state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                    ep_reward += reward
                    ramList[2 * bsNum + worker].add(state, action, reward, new_state)
                    step_Qloss, Qw = trainer[bsNum].optimize()
                    ep_Qloss.append(step_Qloss)
                    Qw_locals.append(copy.deepcopy(Qw))
                    continue
                else:
                    if C_BS[bsNum] - global_content_size[f_index] >= 0:
                        # C_BS[bsNum] = C_BS[bsNum] - global_content_size[f_index]
                        C_BS[bsNum] = C_BS[bsNum] - 1
                        # localContent[bsNum][f_index] = global_content_size[f_index]
                        state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                        s_c[bsNum][f_index] = 1
                        action = trainer[bsNum].get_exploitation_action(state, f_index, neibor_cachestate)
                        if action[1] == 1:
                            delay += 20 + global_content_size[f_index] / 8
                        if action[2] == 1:
                            delay += 200 + global_content_size[f_index] / 8
                        reward, new_state = cache_env.step(action)
                        new_state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                        ep_reward += reward
                        ramList[bsNum].add(state, action, reward, new_state)
                        step_Qloss, Qw = trainer[bsNum].optimize()
                        ep_Qloss.append(step_Qloss)
                        Qw_locals.append(copy.deepcopy(Qw))
                        continue
                    state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                    action = trainer[bsNum].get_exploitation_action(state, f_index, neibor_cachestate)
                    if action[1] == 1:
                        delay += 20 + global_content_size[f_index] / 8
                    if action[2] == 1:
                        delay += 200 + global_content_size[f_index] / 8
                    reward, new_state = cache_env.step(action)
                    minpopu_f_index = content_popu_list.index(min(content_popu_list))
                    # if localPopularity[bsNum][minpopu_f_index] < localPopularity[bsNum][f_index]:
                    if action[0] == 1 and C_BS[bsNum] == 0:
                        s_c[bsNum][minpopu_f_index] = 0
                        s_c[bsNum][f_index] = 1
                    # reward = reward * localPopularity[bsNum][f_index]
                    new_state = s_r + s_c[bsNum] + neibor_cachestate + content_popu_list
                    ramList[bsNum].add(state, action, reward, new_state)
                    ep_reward += reward
                    # perform optimization
                    step_Qloss, Qw = trainer[bsNum].optimize()
                    ep_Qloss.append(step_Qloss)
                    Qw_locals.append(copy.deepcopy(Qw))
    cur_time = time.time()
    print(cur_time)
    # check memory consumption and clear memory
    gc.collect()
    # 所有车平均损失
    Qloss.append(sum(ep_Qloss) / len(ep_Qloss))
    Reward.append(ep_reward / 2)
    if _ep % 5 == 0:
        for w in range(BS_NUM):
            qnet_target[w].load_state_dict(qnet[w].state_dict())
    print(Qloss[_ep], Reward[_ep], ep_hit / 100, delay / 100)
    # if (_ep+1) % 10 == 0:
	# 	train.LEARNING_RATE = train.LEARNING_RATE * 0.9
print('Completed episodes')

name = ['Reward']
test1 = pd.DataFrame(columns=name, data=Reward)
# print(test)
test1.to_csv('reward-fed.csv', encoding='gbk')

name = ['Loss']
test2 = pd.DataFrame(columns=name, data=Qloss)
# print(test)
test2.to_csv('loss-fed.csv', encoding='gbk')

