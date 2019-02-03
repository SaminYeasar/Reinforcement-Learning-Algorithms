import numpy as np
import h5py
import torch
import torch.utils.data


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


def create_dataset(file_name, batch_size, num_train_traj=4, num_valid_traj=4, subsamp_freq=20):
    with h5py.File(file_name, 'r') as f:
        dataset_size = f['obs_B_T_Do'].shape[0] # full dataset size

        states = f['obs_B_T_Do'][:dataset_size,...][...]
        actions = f['a_B_T_Da'][:dataset_size,...][...]
        rewards = f['r_B_T'][:dataset_size,...][...]
        lens = f['len_B'][:dataset_size,...][...]

    # Stack everything together
    perm = np.random.permutation(np.arange(dataset_size))
    train_random_idxs = perm[:num_train_traj]
    valid_random_idxs = perm[num_train_traj:num_train_traj + num_valid_traj]

    start_times = np.random.randint(
        0, subsamp_freq, size=lens.shape[0])

    def make_tensor(idxs):
        xs, ys = [], []
        for i in idxs:
            l = lens[i]
            for j in range(start_times[i], l, subsamp_freq):
                state = states[i, j].reshape(1, -1)
                action = actions[i, j].reshape(1, -1)
                xs.append(state)
                ys.append(action)
        x = np.concatenate(xs, axis=0)
        x = torch.from_numpy(x).float()
        y = np.concatenate(ys, axis=0)
        y = torch.from_numpy(y).float()
        return x, y

    train_x, train_y = make_tensor(train_random_idxs)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

    valid_x, valid_y = make_tensor(valid_random_idxs)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,  **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    return train_loader, valid_loader
