import numpy as np
import pandas as pd
import random
import time
# 随机生成全局内容库
def gcontent_size(content_num, size_list):
    for i in range(content_num):
        size_list.append(np.random.randint(1, 160))
        size_list.append(8)
    return size_list
# 随机生成内容热度
def content_popu(content_num, size_list):
    popu_list = []
    for i in range(content_num):
        if size_list[i] > 0:
            popu_list.append(np.random.randint(1, 100))
        else:
            popu_list.append(0)
    return popu_list
#内容库随机置0
def set_zero(content_num, size_list):
    a = random.sample(range(1, content_num), content_num - 1)
    for i in range(len(a)):
        size_list[a[i]] = 0
    return size_list

#引入电影数据作为内容库
# df = pd.DataFrame(pd.read_csv('content.csv', header=None))
# popu_list = df.values.tolist()
# popu_list = list(np.ravel(popu_list))
# print(len(popu_list))

# 根据内容热度产生请求列表
def request_list(popu_list, number):
    popu_rate_list = []
    s_r_list = []
    content_index = []
    for i in range(len(popu_list)):
        popu_rate_list.append(popu_list[i] / sum(popu_list))
    # print(popu_rate_list)
    for i in range(number):
        s_r_list.append(np.random.choice(a=len(popu_list), size=1, replace=True, p=popu_rate_list)[0])
    # print(s_r_list)
        # s_r_list.append(np.random.randint(0,100))
    return s_r_list
