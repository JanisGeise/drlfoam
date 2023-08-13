from os.path import join, exists
from abc import ABC, abstractmethod
from subprocess import Popen
from typing import Tuple, List
from shutil import copytree
from copy import deepcopy
import torch as pt
from .manager import TaskManager
from .. import get_time_folders
from ..agent import FCPolicy
from ..environment import Environment


class Buffer(ABC):
    def __init__(
        self,
        path: str,
        base_env: Environment,
        buffer_size: int,
        n_runners_max: int,
        keep_trajectories: bool,
        timeout: int,
    ):
        self._path = path
        self._base_env = base_env
        self._buffer_size = buffer_size
        self._n_runners_max = n_runners_max
        self._keep_trajectories = keep_trajectories
        self._timeout = timeout
        self._manager = TaskManager(self._n_runners_max)
        self._envs = None
        self._n_fills = 0

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def fill(self):
        pass

    def create_copies(self):
        envs = []
        for i in range(self._buffer_size):
            dest = join(self._path, f"copy_{i}")
            if not exists(dest):
                copytree(self._base_env.path, dest, dirs_exist_ok=True)
            envs.append(deepcopy(self._base_env))
            envs[-1].path = dest
            envs[-1].seed = i
        self._envs = envs

    def update_policy(self, policy: FCPolicy):
        for env in self.envs:
            policy.save(join(env.path, env.policy))
        policy.save(join(self.base_env.path, self.base_env.policy))

    def reset(self):

        # sample a start time for each copy based on all available start times from the base case in order to
        # increase variance while keeping the trajectory length & run times low, all times have the same probability
        start_times = sorted(map(float, get_time_folders(join(self._base_env.path, "processor0"))))
        idx = pt.multinomial(pt.ones(len(start_times)) / len(start_times), self._buffer_size)

        for i, env in enumerate(self.envs):
            # reset the environment but leave all time folders, so we can initialize a new start time
            env.reset()

            # compute the start & end time (and trajectory length), so that in each episode statistically all parts of
            # the simulation are seen by the agent (trajectory length = const. for all episodes)
            env.start_time = start_times[idx[i]]
            env.end_time = round(env.start_time + start_times[-1] / self._buffer_size, 6)

            # now set the beginning of control
            env.start_control = env.start_time

    def clean(self):
        for env in self.envs:
            proc = Popen([f"./{env.clean_script}"], cwd=env.path)
            proc.wait()

    def save_trajectories(self):
        pt.save(
            [env.observations for env in self.envs],
            join(self._path, f"observations_{self._n_fills}.pt")
        )

    @property
    def base_env(self) -> Environment:
        return self._base_env

    @property
    def envs(self):
        if self._envs is None:
            self.create_copies()
        return self._envs

    @property
    def observations(self) -> Tuple[List[pt.Tensor]]:
        states, actions, rewards = [], [], []
        for env in self.envs:
            obs = env.observations
            if all([key in obs for key in ("states", "actions", "rewards")]):
                states.append(obs["states"])
                actions.append(obs["actions"])
                rewards.append(obs["rewards"])
            else:
                print(f"Warning: environment {env.path} returned empty observations")
        return states, actions, rewards
