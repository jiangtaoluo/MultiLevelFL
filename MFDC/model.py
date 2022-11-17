import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Qnet(nn.Module): 

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Qnet, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim + action_dim, 256)
		self.fcs2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 1)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		state = state.float()
		action = action.float()
		# print("3", state.shape)
		# print("4", action.shape)
		sa = torch.cat((state, action), -1)
		s1 = F.relu(self.fcs1(sa))
		s2 = F.relu(self.fcs2(s1))
		x = self.fc3(s2)
		return x