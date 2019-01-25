#link1: https://github.com/nikhilbarhate99/Deterministic-GAIL-PyTorch/blob/master/GAIL.py

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

		# torch stuff for GPU compute
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)


	def forward(self,state_action):
		# state_action = torch.cat([state,action],1)
		x = torch.tanh(self.fc1(state_action))
		x = torch.tanh(self.fc2(x))
		prob = torch.sigmoid(self.fc3(x))    # used in the "link1"
		return prob

	def expert_reward(self,state_action):
		return -np.log(self.forward(state_action).cpu().data.numpy())


#############################################################################

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
		return action

###############################################################################

class GAIL(object):
	def __init__(self, env_name, state_dim, action_dim,max_action,lr,beta_init):

		##################################################
		# declare (Learner/Generator) Actor & Actor target
		##################################################
		self.policy = Policy(state_dim,action_dim,max_action,lr,beta_init) # doesn't need to do .to(device) as already done in Class Actor

		""" have not used the target for this algorithm, should be used for off-policy learning"""
		self.policy_target = Policy(state_dim,action_dim,max_action,lr,beta_init)
		# update weights of "TARGET" same as "LEARNER" after specific iteration
		self.policy_target.load_state_dict (self.policy.state_dict())

		#######################
		# declare Discriminator
		#######################
		self.discriminator = Disciminator (state_dim,action_dim ,lr,beta_init)

		""" USE A BUFFER LAYER INSTEAD """
		self.expert = ExpertTraj(env_name)


	""" for a given state to GAIL provide learnt action"""
	def select_action(self, state):
		state = torch.FloatTensor(state).to(device)
		action = self.policy.forward(state)
		# check of it's true cause it's different than main code
		# also check input output dimension
		return action

	def update(self, n_itr, batch_size=100):
		# update GAIL
		for itr in range(n_itr):

			################################################
			# (1)
			# Sample "Expert trajectory" (state,action) and
			# for the same "state" as the expert, evaluate action from "Policy"
			################################################
			""" COMPLETE THIS PART """
			expert_state, expert_action = self.expert.sample(batch_size)

			# convert to list because memory is an array of numpy objects
			#expert_state = list(expert_state[:])
			#expert_action = list(expert_action[:])

			####################
			# convert to Tensors
			###################
			expert_state = torch.FloatTensor(expert_state).to(device)
			expert_action = torch.FloatTensor(expert_action).to(device)

			""" takes expert actions (batch_size = 100) . Try evaluating by resetting
			and interacting with Environment instead of evaluating action for expert state"""
			policy_state = expert_state
			""" takes deterministic action . Need ot try stochastic action as well"""
			policy_action = self.policy.forward(policy_state)




			###########################
			# (2) Feed to discriminator
			###########################
			expert_d = self.discriminator.forward(torch.cat([expert_state,expert_action], dim=1) )
			policy_d = self.discriminator.forward(torch.cat([policy_state,policy_action], dim=1) )

			###################
			# (3) Compute GAIL loss
			###################
			# binary_cross_entropy_with_logits (input, target):
			expert_loss = self.discriminator.loss_func(expert_d, torch.ones(expert_d.size()).to(device), reduction = 'sum' )
			policy_loss = self.discriminator.loss_func(policy_d, torch.zeros(policy_d.size()).to(device), reduction ='sum' )

			gail_loss = expert_loss + policy_loss

			if (itr+1) == n_itr:
				print('---------------------------------------------------------------------')
				print('Expert loss = {} | Policy loss = {}'.format(expert_loss, policy_loss))
				print('Expert Prob = {} | Policy prob = {}'.format(torch.mean(expert_d).detach().numpy(), torch.mean(policy_d).detach().numpy()))
			########################################################
			# (4) Update discriminator weight using back-propagation
			########################################################
			self.discriminator.optimizer.zero_grad()
			gail_loss.backward(retain_graph=True)
			self.discriminator.optimizer.step()

			""" we may NOT want to update the policy every time step"""
			""" but that's the update rule in original GAIL paper"""
			#################################################
			# (5) Update Policy weight using back-propagation
			#################################################
			#policy_d = self.discriminator.forward(torch.cat([policy_state, policy_action], dim=1))
			#policy_loss = self.discriminator.loss_func(policy_d, torch.zeros(policy_d.size()).to(device),reduction='sum')
			""" CORRECT THIS : original paper updated based on expert_reward function will implement later"""
			#reward = self.discriminator.expert_reward(torch.cat([policy_state, policy_action]), dim=1)
			self.policy.optimizer.zero_grad()
			policy_loss.backward()
			self.policy.optimizer.step()

	"""

	def save():
		# save learnt policy

	def load():

	"""





