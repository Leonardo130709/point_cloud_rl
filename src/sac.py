import pickle
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .agent import RSAC
from . import wrappers, utils

torch.autograd.set_detect_anomaly(True)


class RLAlg:
    def __init__(self, config):
        utils.set_seed(config.seed)
        self.config = config
        self.env = self.make_env()
        self.task_path = pathlib.Path(config.logdir)
        self.callback = SummaryWriter(log_dir=self.task_path)
        self.agent = RSAC(self.env, config, self.callback)
        self.buffer = utils.ReplayBuffer(config.buffer_size)
        self.interactions_count = 0

    def learn(self):
        obs = None
        while self.interactions_count < self.config.total_steps:
            if obs is None:
                obs = self.env.reset()

            action = self.policy(obs, training=True)
            next_obs, reward, done, _ = self.env.step(action)

            self.interactions_count += self.config.action_repeat

            transition = utils.Transition(
                observation=obs,
                action=action,
                reward=reward,
                done_flag=done,
                next_observation=next_obs
            )
            self.buffer.add(transition)

            dl = DataLoader(
                self.buffer.sample(self.config.spi),
                batch_size=self.config.batch_size,
            )

            for transitions in dl:
                self.agent.step(*transitions)

            if self.interactions_count % self.config.eval_freq == 0:
                scores = [utils.evaluate(self.env, self.policy) for _ in range(10)]
                self.callback.add_scalar('test/eval_reward', np.mean(scores),
                                         self.interactions_count)
                self.callback.add_scalar('test/eval_std', np.std(scores), self.interactions_count)

                self.save()

    def save(self):
        self.config.save(self.task_path / 'config.yml')
        torch.save({
            'interactions': self.interactions_count,
            'params': self.agent.state_dict(),
            'optim': self.agent.optim.state_dict(),
        }, self.task_path / 'checkpoint')
        # TODO: restore buffer saving
        # with open(self.task_path / 'buffer', 'wb') as buffer:
        #     pickle.dump(self.buffer, buffer)

    @classmethod
    def load(cls, path, **kwargs):
        path = pathlib.Path(path)
        config = Config.load(path / 'config.yml', **kwargs)
        alg = cls(config)

        if (path / 'checkpoint').exists():
            chkp = torch.load(
                path / 'checkpoint',
                map_location=torch.device(config.device if torch.cuda.is_available() else 'cpu')
            )
            with torch.no_grad():
                alg.agent.load_state_dict(chkp['params'], strict=False)
                alg.agent.optim.load_state_dict(chkp['optim'])
            alg.interactions_count = chkp['interactions']

        if (path / 'buffer').exists():
            with open(path / 'buffer', 'rb') as b:
                alg.buffer = pickle.load(b)
        return alg

    def make_env(self):
        env = utils.make_env(self.config.task, random=self.config.seed)
        env = wrappers.PointCloudWrapper(
            env,
            pn_number=self.config.pn_number,
            downsample=self.config.downsample,
            apply_segmentation=True
        )
        env = wrappers.ActionRepeat(env, self.config.action_repeat)
        env = wrappers.FrameStack(env, self.config.frames_stack, stack=True)
        return env

    def policy(self, obs, training):
        obs = torch.from_numpy(obs[None]).to(self.agent.device)
        action = self.agent.policy(obs, training)
        return action.detach().cpu().numpy()
