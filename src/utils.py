import copy
import torch
import random
import numpy as np
from typing import NamedTuple
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


def make_env(name, **task_kwargs):
    domain, task = name.split('_', 1)
    if domain == 'ball':
        domain = 'ball_in_cup'
        task = 'catch'
    return suite.load(domain, task, task_kwargs=task_kwargs)


def set_seed(seed):
    #TODO: fix seed in dataloaders
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def simulate(env, policy, training):
    obs = env.reset()
    done = False
    observations, actions, rewards, dones, log_probs = [[] for _ in range(5)]
    while not done:
        action, log_prob = policy(obs, training)
        new_obs, reward, done = env.step(action)
        observations.append(obs)
        actions.append(action)
        dones.append([done])
        rewards.append(np.float32([reward]))
        log_probs.append(log_prob)
        obs = new_obs

    tr = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        done_flags=dones,
        log_probs=log_probs,
    )
    for k, v in tr.items():
        tr[k] = np.stack(v)
    return tr


def evaluate(env, policy):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(obs, training=False)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


class Transition(NamedTuple):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done_flag: np.ndarray
    next_observation: np.ndarray


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
        idx = np.random.randint(self._size, size=min(size, self._size))
        return Subset(self._data, idx)


class TruncatedTanhTransform(td.transforms.TanhTransform):
    _lim = .999

    def _inverse(self, y):
        y = torch.clamp(y, min=-self._lim, max=self._lim)
        return y.atanh()


@torch.no_grad()
def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.copy_((1. - rho) * pt.data + rho * po.detach())


def make_targets(*modules):
    return map(lambda m: copy.deepcopy(m).requires_grad_(False), modules)
