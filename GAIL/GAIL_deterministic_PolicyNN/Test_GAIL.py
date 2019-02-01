""" To save video run the following command"""
# xvfb-run -s "-screen 0 640x480x24" python Test_GAIL.py
# link: https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/

import gym
import torch
from GAIL import GAIL
import os
# for recording the agent performance (couldn't get it running, running using gym.wrapper.Monitor command)
from gym_recording.wrappers import TraceRecordingWrapper

def test():
	###################
	# define parameters
	###################

	env_name = "BipedalWalker-v2"
	# env_name = "LunarLanderContinuous-v2"
	solved_reward = 1
	lr = 0.0002       # learning rate  $ Make it adaptive with iteration$
	beta_init = 0.5   # parameter for Adam optimizer

	n_epoch = 10000   			 	# number of epoch
	batch_size = 100          		# number of transitions samples each time
	n_itr = 100        	 	   		# update policy "n_itr" times at each epoch
	n_eval_trials = 20            	# evaluate learnt policy for fixed number of trials and calculate the mean rewards
	max_trajectory_steps = 1000  	# max time steps (s,a) pair taken for one epoch

	record_runs = True

	""" here Total number of iterations = epoch * n_itr"""
	###############################
	# Define environment parameters
	###############################

	env = gym.make(env_name)
	""" Command to start recoding runs"""
	if record_runs ==True and not os.path.exists(env_name):
		env = gym.wrappers.Monitor(env, directory='{}/'.format(env_name), force=True, write_upon_reset=True)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	""" don't know why we need this"""
	max_action = float(env.action_space.high[0])

	################################
	# to record agent performance
	################################

	###########
	# call GAIL
	###########
	Policy = GAIL(env_name, state_dim, action_dim, max_action, lr, beta_init)
	# load pre-trained policy
	Policy.load()

	total_reward = 0
	"""use avg reward to evaluate policy over specific number of trials"""
	""" We may not want to evaluate at every epochs. Do it maybe after 100 epoch"""
	for trial in range(n_eval_trials):
		state = env.reset()
		done = False
		itr = 0
		reward_per_run = 0
		while not done:
			env.render()
			""" we don't need to store state,"""
			# take action using policy
			action = Policy.select_action(state)
			# observe the next state and reward for taking action
			next_state, reward, done, _ = env.step(action.detach()) # detach from tensor

			# append reward
			reward_per_run += reward
			# declare state to be next_state at next iteration
			state = next_state

			# as for perfect expert it's in continuous state. Fix the number of maximum time step that it's let go in environment
			itr += 1
			if itr >= max_trajectory_steps:
				break
		print("Reward for {} run is {}".format(trial, reward_per_run) )
		total_reward += reward_per_run
	avg_reward = int(total_reward/n_eval_trials)
	print("Avg Reward: {}".format(avg_reward))
	#monitor.close()



if __name__=='__main__':
	test()
