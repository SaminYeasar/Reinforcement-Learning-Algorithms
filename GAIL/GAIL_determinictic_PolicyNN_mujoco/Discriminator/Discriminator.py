
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ExpertTraj  # need to look into this
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################

class Disciminator(nn.Module):
	def __init__(self, state_dim, action_dim, lr, beta_init):
		super(Disciminator, self).__init__()

		#################
		# define Network
		#################
		# torch.nn.Linear (input,output)
		self.fc1 = nn.Linear(state_dim + action_dim, 400)
		self.fc2 = nn.Linear(400, 300)
		self.fc3 = nn.Linear(300, 1)  # output will be probability of being "Expert" or "Generator (Learner)"

		####################################
		# define optimizer and loss function
		####################################
		self.optimizer=optim.Adam(self.parameters(), lr=lr, betas=(beta_init, 0.999))
		self.loss_func=F.binary_cross_entropy_with_logits   # binary cross entropy loss
		#self.loss_func = torch.nn.BCEWithLogitsLoss
		# torch stuff for GPU compute
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)


	def forward(self,state_action):
		# state_action = torch.cat([state,action],1)
		x = torch.tanh(self.fc1(state_action))
		x = torch.tanh(self.fc2(x))
		"""if using BCEwithlogitloss from torch that itself does sigmoid to inputs so no need to do it here"""
		#prob = torch.sigmoid(self.fc3(x))    # used in the "link1"
		#return prob
		x = self.fc3(x)
		return x


	def expert_reward(self,state_action):
		return -torch.log(self.forward(state_action))

