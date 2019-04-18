import numpy as np
import h5py
import torch
import torch.utils.data
import os

class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(
            -1, 1), np.array(d).reshape(-1, 1)

class ExpertBuffer(ReplayBuffer):
    def __init__(self, file_name, num_traj=4, subsamp_freq=20):
        super(ExpertBuffer, self).__init__()

        with h5py.File(file_name, 'r') as f:
            dataset_size = f['obs_B_T_Do'].shape[0] # full dataset size

            states = f['obs_B_T_Do'][:dataset_size,...][...]
            actions = f['a_B_T_Da'][:dataset_size,...][...]
            rewards = f['r_B_T'][:dataset_size,...][...]
            lens = f['len_B'][:dataset_size,...][...]

        # Stack everything together
        random_idxs = np.random.permutation(np.arange(dataset_size))[:num_traj]
        start_times = np.random.randint(
            0, subsamp_freq, size=lens.shape[0])

        for i in random_idxs:
            l = lens[i]
            for j in range(start_times[i], l, subsamp_freq):
                state = states[i, j]
                action = actions[i, j]
                self.add((state, np.empty([]), action, np.empty([]), np.empty([])))


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def save_weights(model, args, directory='./preTrained'):
    # see if the folder exit if note create one
    create_folder(directory)
    print("Saving weights")
    torch.save(model.state_dict(), '{}/{}_encoder.pth'.format(directory, args.env_name))
    #torch.save(model.decode.state_dict(), '{}/{}/decoder.pth'.format(directory, args.env_name))


def load_weights(model, args, directory='./preTrained'):
    if os.path.exists(directory):
        print("Loading PreTrained Weights")
        model.load_state_dict(torch.load('{}/{}_encoder.pth'.format(directory, args.env_name)))
        #model.decode.load_state_dict(torch.load('{}/{}/decoder.pth'.format(directory, args.env_name)))
    else:
        print("PreTrained Weights don't exists. Training Agent from scratch")
