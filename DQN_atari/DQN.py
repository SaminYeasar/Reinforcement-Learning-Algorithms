# link 1: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# link 2: https://www.kaggle.com/ashishpatel26/zero-to-hero-in-pytorch-ml-dl-rl/notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
	def __init__(self, ALPHA):
		super(DeepQNetwork, self).__init__()

		# construct NN layers
		#self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
		self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		#self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		#self.fc1 = nn.Linear(128 * 19 * 8, 512)
		self.fc1 = nn.Linear( 64*9*6, 512)  # no of channels * width * height
		self.fc2 = nn.Linear(512, 6)

		self.relu = nn.ReLU()

		# optimizer
		self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)

		# compute loss
		self.loss = nn.MSELoss()

		# torch stuff
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		# send the network to device
		self.to(self.device)
	"""
	def forward(self, observation):
		# convert observation to tensor
		# send variable to device
		observation = torch.Tensor(observation).to(self.device)
		# crop shape
		observation = observation.view(-1, 1, 185,
									   95)  # (-1 so that can handle any number of samples, no of channel , frame X, frame Y)

		# feed observation to NN
		observation = F.relu(self.conv1(observation))
		observation = F.relu(self.conv2(observation))
		observation = F.relu(self.conv3(observation))
		observation = observation.view(-1, 128 * 19 * 8)
		observation = F.relu(self.fc1(observation))
		action = self.fc2(observation)  # action is a matrix
		return action
	"""

	def forward(self, obs):
		"""
		obs(observation) = (num of samples , width , height , num of input channel = 1)
		"""
		obs = torch.Tensor(obs).to(self.device)
		obs = obs.view(-1, 1, obs.shape[1],obs.shape[2]) # 32,number of channels = 1 , 105, 80
		obs = self.relu(self.conv1(obs))  # (num of samples , updated width , updated height , num of input channel = 32)
		obs = self.relu(self.conv2(obs))  # (num of samples , updated width , updated height , num of input channel = 64)
		obs = self.relu(self.conv3(obs))  # (num of samples , updated width , updated height , num of input channel = 64)

		# x.size(0) =  number of samples
		obs = obs.view(obs.size(0), -1)  # (num of samples , updated width * updated height * num of input channel = 64)
		obs = self.relu(self.fc1(obs))  # (num of samples , 512)
		action = self.fc2(obs)  # (num of samples, possible actions)
		return action

#####################################################################

class Replay_Buffer(object):
	def __init__(self,maxMemorySize):
		self.memSize = maxMemorySize
		self.memory = []
		self.memCounter = 0

	def storeTransition(self, state, action, reward, state_,done):

		if self.memCounter < self.memSize:
			self.memory.append([state, action, reward, state_,done])
		else:
			# 10%100 = 10; 35%100=35; 100%100=0; 105%100=5; 113%100=100
			self.memory[self.memCounter % self.memSize] = [state, action, reward, state_,done]
		self.memCounter += 1

	def sample_miniBatch(self, batch_size):
		# Sample Mini-Batch
		if self.memCounter + batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCounter)))
		else:
			memStart = int(np.random.choice(range(self.memSize - batch_size - 1)))

		miniBatch = np.array( self.memory[memStart:memStart + batch_size] )
		return miniBatch

#####################################################################
class Agent(object):
	def __init__(self, env, maxMemorySize, alpha, epsilon, minEpsilon, gamma, replace):
		"""
		gamma = discount factor
		alpha = agent learning rate
		epsilon = with probability agent takes random action
		"""
		self.ALPHA = alpha

		self.policy_net = DeepQNetwork(self.ALPHA)  # feeding the network
		self.target_net = DeepQNetwork(self.ALPHA)  # feeding the network

		self.buffer = Replay_Buffer(maxMemorySize)

		#######################################################
		# construct memory (will be used in "storeTransition")
		"""
		self.memSize = maxMemorySize
		self.memory = []
		self.memCounter = 0
		"""

		#######################################################
		# (will be used in "chooseAction")

		# define the start and end of epsilon updates
		# decrease epsilon after fixed number of episodes (ex: 500 used in here)
		self.EPSILON = epsilon
		self.minEPSILON = minEpsilon
		self.actionSpace = env.action_space.n
		self.steps = 0  # keep track number of times action os taken

		#######################################################
		# (will be used in "learn")

		self.GAMMA = gamma
		self.learn_step_counter = 0  # keep track number of times "learn" executed
		self.replace_target_counter = replace
	"""
	def storeTransition(self, state, action, reward, state_):

		if self.memCounter < self.memSize:
			self.memory.append([state, action, reward, state_])
		else:
			# 10%100 = 10; 35%100=35; 100%100=0; 105%100=5; 113%100=100
			self.memory[self.memCounter % self.memSize] = [state, action, reward, state_]
		self.memCounter += 1
	"""
	def chooseAction(self, env, observation):
		rand = np.random.random()

		# get action(matrix) for given observation using NN
		"""action = probability matrix of possible actions
			        action 1 | action 2| action 3| ..
			frame 1|
			frame 2|
			frame 3|
			 .
			 .
		"""
		action = self.policy_net.forward(observation)

		# implement simple epsilon-greedy
		if rand < 1 - self.EPSILON:
			action = torch.argmax(action[1]).item()
		else:
			action = np.random.choice(self.actionSpace)

		self.steps += 1  # keep track number of times we have taken action

		return action

	def learn(self, sampled_memory):
		# we use batch sampling to learn so that
		#	 	- we don't get stuck at local optimia
		#		- or don't constantly sample from co-related sample



		###########################
		# The target network has its weights kept frozen most of the time,
		# but is updated with the policy network’s weights every so often
		# NOTE: not using it in the code so "replace_target_counter" is not defined.

		if self.replace_target_counter is not None and \
				self.learn_step_counter % self.replace_target_counter == 0:
			self.target_net.load_state_dict(self.policy_net.state_dict())  # load_state_dict is pytorch command
			print("Target network got updated at learning step {}".format(self.learn_step_counter))

		##########################

		# sample Mini-Batch

		"""
		if self.memCounter + batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCounter)))
		else:
			memStart = int(np.random.choice(range(self.memSize - batch_size - 1)))

		miniBatch = self.memory[memStart:memStart + batch_size]
		sampled_memory = np.array(miniBatch)
		"""

		#sampled_memory = self.sample_miniBatch(batch_size)

		#sampled_memory = self.buffer.sample_miniBatch(batch_size)

		# print('shape of memory',memory.shape)
		# shape (batch_size ,number of saved input  )
		#        batch_size = 32
		#        number of saved input = 4 ; state,action,reward,next_state
		#        shape of sampled_memory[:,0][0] = (105,80) it's an image input

		# convert numpy array to list as pytorch doesn't take numpy array
		# sampled_memory[:,0][:] all rows(entire batch) , zero'th element(state) and all of the pixels
		# send it to device
		# (state,action,reward,state_)

		# convert to list because memory is an array of numpy objects
		state 		= list(sampled_memory[:, 0][:])
		action 		= list(sampled_memory[:, 1][:])
		reward		= list(sampled_memory[:, 2][:])
		next_state	= list(sampled_memory[:, 3][:])

		###################
		# Feed-forward :
		####################

		# importance of listing here: if we don't convert into list,
		# it gives error when we "torch.Tensor" at "forward"

		###################
		# predict Q based on current policy
		###################
		# shape (batch_size = 32 ,possible actions  = 6)
		Qpred = self.policy_net.forward(state).to(self.policy_net.device)          # during forward prop converts as tensor
		# shape (batch_size = 32, 1 )
		Qpred = Qpred.gather(1,torch.LongTensor(action).view(-1,1))


		##################
		# compute Q target
		###################
		"""
		 1.    .detach() detaches the output from the computationnal graph.
		 		So no gradient will be backproped along this variable
		 2.    .max(1)[0] return only values[0] of the maximum along column axis(1)
		
		"""
		# shape (batch_size = 32 ,possible actions  = 6)
		Qnext =  self.target_net.forward(next_state).detach() # detach from graph, don't backpropagate

		# output of Qnext if probaility matrix of actions, where column is different actions for different batch along the row
		#maxA = torch.argmax(Qnext, dim=1).to(self.policy_net.device)

		reward = torch.Tensor(reward).to(self.policy_net.device)               # need to convert as tensor

		# shape (batch_size)
		Qtarget = reward + self.GAMMA*Qnext.max(1)[0] # consider the Qnext values that gives [ max_a Q(s',a) ]
		# shape (batch_size,1)
		Qtarget = Qtarget.unsqueeze(1)



		"""
		policy (state) >>> Qpred(s,A) | shape=(batch = 32 * actions = 6) | A = [action 1, action 2 ...] 
		compute Qpred (s,a) | shape = (32 * 1)
		
		target_network (next_state) >>> Qnext (32*6)
		max_a Qnext | shape ( 32 )
		Qtarget = reward + self.GAMMA * max_a Qnext
		convert Qtarget (32) >>>> Qtarget (32,1)
		"""

		#########################################

		"""
		# Update epsilon
		if self.steps > 500:
			if self.EPSILON - 1e-4 > self.minEPSILON:
				self.EPSILON -= 1e-4
			else:
				self.EPSILON = self.minEPSILON
		"""
		# update epsilon
		self.update_epsilon()

		###################
		# clear out the gradients of all Variables every-time we do backprop to make sure it's a pure "batch learning"
		# IMPORTANT:https://stackoverflow.com/questions/48001598/why-is-zero-grad-needed-for-optimization
		###################
		self.policy_net.optimizer.zero_grad()

		###################
		# Compute loss:
		###################
		loss = self.policy_net.loss(Qtarget, Qpred).to(self.policy_net.device)

		###################
		# Back-propagation:
		###################
		loss.backward()

		###################
		# Update network parameter
		###################
		self.policy_net.optimizer.step()  # torch stuff

		self.learn_step_counter += 1

	def update_epsilon(self):
		if self.steps > 500:
			if self.EPSILON - 1e-4 > self.minEPSILON:
				self.EPSILON -= 1e-4
			else:
				self.EPSILON = self.minEPSILON


