"""
Used simple NN network to represent Generator

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ExpertTraj  # need to look into this
import numpy as np

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, lr, beta_init):

		super(Policy, self).__init__()

		#################
		# define Network
		#################
		self.fc1 = nn.Linear(state_dim, 400)
		self.fc2 = nn.Linear(400, 300)
		self.fc3 = nn.Linear(300, action_dim)

		"""find out reason of using max_action"""
		self.max_action = max_action

		####################################
		# define optimizer and loss function
		####################################
		self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta_init, 0.999))
		self.loss_func = F.binary_cross_entropy_with_logits  # binary cross entropy loss

		# torch stuff for GPU compute
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.to(self.device)

	def forward(self, state):
		x = torch.relu(self.fc1(state))
		x = torch.relu(self.fc2(x))
		action = torch.tanh(self.fc3(x)) * self.max_action   #"""didn't get the logic"""
		#action_prob = torch.softmax(self.fc3(x), dim=1)
		# will return probability of taking particular action
		return action
	"""
	def select_action(self, state):
		action_prob = self.forward(state)
		action = action_prob.multinomial(1) # will select the most probable action
		return action
	"""