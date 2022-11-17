from __future__ import division
import torch
# import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

import numpy as np
# import math

import utils
import model
# from options import args_parser

BATCH_SIZE = 20
LEARNING_RATE = 0.05
GAMMA = 0.9
EPSILON = 0.9
TAU = 0.001

class Trainer:

	def __init__(self, state_dim, action_dim, ram, Qnet, Q_tar):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ram = ram
		self.iter = 0

		self.Qnet = Qnet
		self.target_Qnet = Q_tar
		self.Qnet_optimizer = torch.optim.Adam(self.Qnet.parameters(), LEARNING_RATE)

		# utils.hard_update(self.target_Qnet, self.Qnet)

	def get_exploitation_action(self, state, req_index, neibor_state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		if isinstance(state, list):
			state = torch.from_numpy(np.array(state))
		# print("1", state.shape)
		if neibor_state[req_index] == 1:
			actions_value1 = self.Qnet.forward(state, torch.from_numpy(np.array([1, 1, 0]))).detach()
			actions_value2 = self.Qnet.forward(state, torch.from_numpy(np.array([0, 1, 0]))).detach()
			actions_value1 = actions_value1.numpy()
			actions_value2 = actions_value2.numpy()
			max_actionvalue = max(actions_value1, actions_value2)
			if max_actionvalue == actions_value1:
				action_max = [1, 1, 0]
				action_random = [0, 1, 0]
			else:
				action_max = [0, 1, 0]
				action_random = [1, 1, 0]
			if np.random.rand(1) >= EPSILON:
				return action_random
			else:
				return action_max
		else:
			actions_value1 = self.Qnet.forward(state, torch.from_numpy(np.array([1, 0, 1]))).detach()
			actions_value2 = self.Qnet.forward(state, torch.from_numpy(np.array([0, 0, 1]))).detach()
			actions_value1 = actions_value1.numpy()
			actions_value2 = actions_value2.numpy()
			max_actionvalue = max(actions_value1, actions_value2)
			if max_actionvalue == actions_value1:
				action_max = [1, 0, 1]
				action_random = [0, 0, 1]
			else:
				action_max = [0, 0, 1]
				action_random = [1, 0, 1]
			if np.random.rand(1) >= EPSILON:
				return action_random
			else:
				return action_max
	def get_action(self, state, req_index, neibor_state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		if isinstance(state, list):
			state = torch.from_numpy(np.array(state))
		# print("1", state.shape)
		if neibor_state[req_index] == 1:
			actions_value1 = self.Qnet.forward(state, torch.from_numpy(np.array([1, 1, 0]))).detach()
			actions_value2 = self.Qnet.forward(state, torch.from_numpy(np.array([0, 1, 0]))).detach()
			actions_value1 = actions_value1.numpy()
			actions_value2 = actions_value2.numpy()
			max_actionvalue = max(actions_value1, actions_value2)
			if max_actionvalue == actions_value1:
				action_max = [1, 1, 0]
			else:
				action_max = [0, 1, 0]
			return action_max
		else:
			actions_value1 = self.Qnet.forward(state, torch.from_numpy(np.array([1, 0, 1]))).detach()
			actions_value2 = self.Qnet.forward(state, torch.from_numpy(np.array([0, 0, 1]))).detach()
			actions_value1 = actions_value1.numpy()
			actions_value2 = actions_value2.numpy()
			max_actionvalue = max(actions_value1, actions_value2)
			if max_actionvalue == actions_value1:
				action_max = [1, 0, 1]
			else:
				action_max = [0, 0, 1]
			return action_max



	'''
	def get_exploration_action(self, state, action):
		"""

		'''
	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1, a1, r1, s2 = self.ram.sample(BATCH_SIZE)
		s1 = torch.from_numpy(s1)
		a1 = torch.from_numpy(a1)
		r1 = torch.from_numpy(r1)
		# s2 = torch.from_numpy(s2)
		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		q_eval = torch.mean(torch.squeeze(self.Qnet.forward(s1, a1)))
		# print(s1.shape)
		# print(s2.shape)
		s2 = s2.tolist()
		a2 = []
		req_index = []
		neibor_state = []
		for i in range(len(s2)):
			neibor_state.append(s2[i][19448:29172])
			for j in range(9724):
				if s2[i][j] == 1:
					req_index.append(j)
		for i in range(len(s2)):
			a2.append(self.get_exploitation_action(s2[i], req_index[i], neibor_state[i]))
		a2 = np.array(a2)
		a2 = torch.from_numpy(a2)
		s2 = np.array(s2)
		s2 = torch.from_numpy(s2)
		# print("2", s2.shape)
		q_next = torch.mean(torch.squeeze(self.target_Qnet.forward(s2, a2).detach()))
		q_target = torch.mean(torch.squeeze(r1)) + GAMMA * q_next
		# compute critic loss, and update the critic
		loss_Qnet = F.smooth_l1_loss(q_eval, q_target)  # 计算状态当前网络的损失函数
		self.Qnet_optimizer.zero_grad()
		loss_Qnet.backward()  # 反向传播更新各隐藏层权重W
		self.Qnet_optimizer.step()
		utils.soft_update(self.target_Qnet, self.Qnet, TAU)
		return loss_Qnet.item(), self.Qnet.state_dict()
