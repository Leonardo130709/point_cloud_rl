import time
import pickle
import pathlib

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .agent import RSAC
from . import wrappers, utils


class RLAlg:
    def __init__(self, config):
        utils.set_seed(config.seed)
        self.config = config
        self.env = self.make_env(random=config.seed)
        self.task_path = pathlib.Path(config.logdir)
        self.callback = SummaryWriter(log_dir=self.task_path)
        self.agent = RSAC(self.env, config, self.callback)
        self.buffer = utils.ReplayBuffer(config.buffer_size)
        self.interactions_count = 0

    def learn(self):
        dur = time.time()
        obs = None
        while self.interactions_count < self.config.total_steps:
            if obs is None:
                obs = self.env.reset().observation

            action = self.policy(obs, training=True)
            timestep = self.env.step(action)

            self.interactions_count += self.config.action_repeat

            transition = utils.Transition(
                observation=obs,
                action=action,
                reward=np.array(timestep.reward, dtype=np.float32)[np.newaxis],
                done_flag=np.array(timestep.last(), dtype=bool)[np.newaxis],
                next_observation=timestep.observation
            )
            self.buffer.add(transition)

            if timestep.last():
                obs = None
            else:
                obs = timestep.observation

            dl = DataLoader(
                self.buffer.sample(self.config.spi),
                batch_size=self.config.batch_size,
            )

            for transitions in dl:
                observations, actions, rewards, dones, next_observations =\
                    map(lambda t: t.to(self.agent.device), transitions)
                self.agent.step(observations, actions, rewards, dones, next_observations)

            if self.interactions_count % self.config.eval_freq == 0:
                scores = [utils.evaluate(self.make_env(), self.policy) for _ in range(10)]
                self.callback.add_scalar('test/eval_reward', np.mean(scores),
                                         self.interactions_count)
                self.callback.add_scalar('test/eval_std', np.std(scores), self.interactions_count)

                self.save()

        dur = time.time() - dur
        self.callback.add_hparams(
            vars(self.config),
            dict(duration=dur, score=np.mean(scores))
        )

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
                alg.agent.load_state_dict(chkp['params'])
                alg.agent.optim.load_state_dict(chkp['optim'])
            alg.interactions_count = chkp['interactions']

        if (path / 'buffer').exists():
            with open(path / 'buffer', 'rb') as b:
                alg.buffer = pickle.load(b)
        return alg

    def make_env(self, **task_kwargs):
        env = utils.make_env(self.config.task, **task_kwargs)
        # env = wrappers.PointCloudWrapper(
        #     env,
        #     pn_number=self.config.pn_number,
        #     downsample=self.config.downsample,
        #     render_kwargs=dict(camera_id=0, height=240, width=320)
        # )
        env = wrappers.PointCloudWrapperV2(
            env,
            pn_number=self.config.pn_number,
            stride=self.config.downsample,
            render_kwargs=dict(camera_id=0, height=84, width=84)
        )
        env = wrappers.ActionRepeat(env, self.config.action_repeat, discount=self.config.discount)
        env = wrappers.FrameStack(env, self.config.frames_stack)
        return env

    def policy(self, obs, training):
        obs = torch.from_numpy(obs[None]).to(self.agent.device)
        action = self.agent.policy(obs, training)
        return action.detach().cpu().numpy().flatten()
