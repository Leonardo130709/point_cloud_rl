import os
os.environ['MUJOCO_GL']='egl'

import shlex
import argparse
import subprocess
import pathlib
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=pathlib.Path)
    parser.add_argument('tasks', type=str, nargs='+')
    return parser.parse_args()


scenarios = dict(
    reference='',
    increased_frames='--frames_stack 4',
    deeper_pointnet='--pn_layers 64 128 128 256',
    tighter_bottleneck='--pn_layers 64 128 128',
    larger_bottleneck='--pn_layers 64 128 512',
    with_features='--features_from_layers 0',
    thinner_actor='--actor_layers 200 200',
    wider_critic='--critic_layers 512 256',
    reconstruction='--aux_loss reconstruction',
    keypoints='--aux_loss reconstruction --pn_layers 64 128 16 --features_from_layers 0'
)


def make_dir(logdir, task, scenario):
    task_dir = logdir / task / scenario
    task_dir.mkdir(parents=True, exist_ok=False)
    return task_dir


args = parse_args()
procs = []
for task, (name, flags) in product(args.tasks, scenarios.items()):
    task_dir = make_dir(args.logdir, task, name)
    command = f'python -m src.train --logdir {str(task_dir)} --task {task} ' + flags

    with (task_dir / 'stderr.txt').open('w') as stderr, (task_dir / 'stdout.txt').open('w') as stdout:
        proc = subprocess.Popen(
                shlex.split(command),
                stdout=stdout,
                stderr=stderr
            )
    procs.append(proc)
