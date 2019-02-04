#link1: https://github.com/nikhilbarhate99/Deterministic-GAIL-PyTorch/blob/master/GAIL.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ExpertTraj  # need to look into this
import numpy as np
from utils import create_folder
import os

from Discriminator.Discriminator import Disciminator
from Generator.Policy_NN import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAIL(object):
	def __init__(self, env_name, state_dim, action_dim, max_action, lr, beta_init):

		##################################################
		# declare (Learner/Generator) Actor & Actor target
		##################################################
		self.policy = Policy(state_dim, action_dim, max_action, lr, beta_init) # doesn't need to do .to(device) as already done in Class Actor

		""" have not used the target for this algorithm, should be used for off-policy learning"""
		self.policy_target = Policy(state_dim, action_dim, max_action, lr, beta_init)
		# update weights of "TARGET" same as "LEARNER" after specific iteration
		self.policy_target.load_state_dict (self.policy.state_dict())

		#######################
		# declare Discriminator
		#######################
		self.discriminator = Disciminator (state_dim,action_dim ,lr,beta_init)

		""" USE A BUFFER LAYER INSTEAD """
		self.expert = ExpertTraj(env_name)

		# for the purpose of saving
		self.create_folder = create_folder

	""" for a given state to GAIL provide learnt action"""
	def select_action(self, state):
		state = torch.FloatTensor(state).to(device) #.reshape(1,-1)
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
			expert_d = self.discriminator(torch.cat([expert_state, expert_action], dim=1))
			policy_d = self.discriminator(torch.cat([policy_state, policy_action], dim=1))

			###################
			# (3) Compute GAIL loss
			###################
			# binary_cross_entropy_with_logits (input, target):
			policy_loss = self.discriminator.loss_func(policy_d, torch.zeros(policy_d.size()).to(device), reduction='sum')
			expert_loss = self.discriminator.loss_func(expert_d, torch.ones(expert_d.size()).to(device), reduction='sum')

			#policy_loss = self.discriminator.compute_loss(policy_d, torch.zeros(policy_d.size()).to(device))
			#expert_loss = self.discriminator.compute_loss(expert_d, torch.ones(expert_d.size()).to(device))

			gail_loss = (expert_loss + policy_loss)

			if (itr+1) == 1:
				print('---------------------------------------------------------------------')
				print('Expert loss = {} | Policy loss = {}'.format(expert_loss, policy_loss))
				print('Expert Prob = {} | Policy prob = {}'.format(torch.sigmoid(torch.mean(expert_d)).detach().numpy(), torch.sigmoid(torch.mean(policy_d)).detach().numpy()))
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
			# use loss function from discriminator but
			# policy want to be line expert thus compare with torch.ones to get the loss
			policy_loss = (self.discriminator.loss_func(policy_d, torch.ones(policy_d.size()).to(device),reduction='sum'))

			#########################################
			if (itr + 1) == 1:
				print('---------------------------------------------------------------------')

				print('| Policy loss = {}'.format(policy_loss))
			########################################################


			""" CORRECT THIS : original paper updated based on expert_reward function will implement later"""
			#reward = self.discriminator.expert_reward(torch.cat([policy_state, policy_action], dim=1))
			self.policy.optimizer.zero_grad()
			#reward.backward()
			policy_loss.backward()
			self.policy.optimizer.step()



	def save(self, directory='./preTrained', name='GAIL'):
		# see if the folder exit if note create one
		self.create_folder(directory)
		torch.save(self.policy.state_dict(), '{}/{}_actor.pth'.format(directory, name))
		torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory, name))

	def load(self, directory='./preTrained', name='GAIL'):
		if os.path.exists(directory):
			print("Loading PreTrained Weights")
			self.policy.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, name)))
			self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory, name)))
		else:
			print("PreTrained Weights don't exists. Training Agent from scratch")

	"""

	def save():
		# save learnt policy

	def load():

	"""





