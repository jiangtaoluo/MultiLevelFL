# import gym
# from gym import spaces
# from gym.utils import seeding
import numpy as np
# from os import path
d_c = 5
d_b = 20
d_p = 200
xi1 = 0.001
xi2 = 0.1
xi3 = 0.899

def Reward(act):
    if act == [0, 0, 0]:
        # return np.exp(-1 * (xi2*d_b + xi1*d_c))
        return 1
    elif act == [0, 1, 0] or act == [1, 1, 0]:
        return 0.5
        # return np.exp(-1 * (xi3*d_p + xi1*d_c))
    else:
        # return np.exp(-1*xi1*d_c)
        return 0
def step(a):
    if a[0] == 1 and a[1] == 1:
        r = Reward(a)
        s_ = [1, 1]
        return r, s_
    if a[0] == 0 and a[1] == 1:
        r = Reward(a)
        s_ = [1, 0]
        return r, s_
    if a[0] == 1 and a[2] == 1:
        r = Reward(a)
        s_ = [1, 1]
        return r, s_
    if a[0] == 0 and a[2] == 1:
        r = Reward(a)
        s_ = [1, 0]
        return r, s_
    if a[0] == 0 and a[1] == 0 and a[2] == 0:
        r = Reward(a)
        s_ = [1, 1]
        return r, s_
