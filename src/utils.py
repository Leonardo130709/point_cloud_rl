import copy
import torch
import random
import numpy as np
from typing import NamedTuple, Any
from collections import deque
from dm_control import suite
from torch.utils.data import Dataset, Subset
nn = torch.nn
td = torch.distributions


def build_mlp(*sizes, act=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model):
    s = 0
    for p in model.parameters():
        if torch.is_tensor(p.grad):
            s += p.grad.pow(2).sum().item()
    return np.sqrt(s)


def make_env(name, task_kwargs=None, environment_kwargs=None):
    domain, task = name.split('_', 1)
    if domain == 'ball':
        domain = 'ball_in_cup'
        task = 'catch'
    return suite.load(domain, task, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs)


def set_seed(seed):
    #TODO: fix seed in dataloaders
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate(env, policy):
    obs = env.reset().observation
    done = False
    total_reward = 0
    while not done:
        action = policy(obs, training=False)
        timestep = env.step(action)
        obs = timestep.observation
        done = timestep.last()
        total_reward += timestep.reward
    return total_reward


class Transition(NamedTuple):
    observation: Any
    action: Any
    reward: Any
    done_flag: Any
    next_observation: Any


class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self._data = deque(maxlen=capacity)
        self._size = 0
        self.capacity = capacity

    def add(self, transition: Transition):
        self._size = min(self.capacity, self._size+1)
        self._data.append(transition)

    def __getitem__(self, idx):
        observations, actions, rewards, done_flag, next_observations = self._data[idx]
        return observations, actions, rewards, done_flag, next_observations

    def __len__(self):
        return self._size

    def sample(self, size):
        idx = np.random.randint(self._size, size=size)
        return Subset(self._data, idx)


def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.copy_((1. - rho) * pt.data + rho * po.detach())


def make_targets(*modules):
    return map(lambda m: copy.deepcopy(m).requires_grad_(False), modules)
