from os.path import join, exists
from abc import ABC, abstractmethod
from subprocess import Popen
from typing import Tuple, List
from shutil import copytree
from copy import deepcopy
import torch as pt
from .manager import TaskManager
from .. import get_time_folders, fetch_line_from_file
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
        trajectory_length: int = 1000
    ):
        self._counter = None
        self._path = path
        self._base_env = base_env
        self._buffer_size = buffer_size
        self._n_runners_max = n_runners_max
        self._keep_trajectories = keep_trajectories
        self._timeout = timeout
        self._manager = TaskManager(self._n_runners_max)
        self._envs = None
        self._n_fills = 0
        self._len_traj = trajectory_length

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

        # initialize counter for counting how often which start time was already sampled,
        # ones because weights =  1 / counter
        if self._counter is None:
            self._counter = pt.ones(len(start_times))

        # get the time step from the 'controlDict'
        dt = float(fetch_line_from_file(join(self._base_env.path, "system", "controlDict"),
                                        "deltaT ").strip("\n;").split(" ")[-1])

        # make sure that we don't exceed the end time of the base case, because then we wouldn't have data of the base
        # case available for reward function (if sampled t_start + len_trajectory > t_end_base)
        weights = 1 / self._counter

        # set the weights for sampling to zero if the trajectory would exceed last dt of base case to make sure that
        # they are never drawn, else multiply with one (= leave weights as they are), * 1.1 as safety factor in case
        # dt != const. the trajectory length may differ by a few dt depending on solver settings
        max_t = pt.tensor([0 if t + self._len_traj * dt * 1.1 >= max(start_times) else 1 for t in start_times])
        weights *= max_t

        # sample start times based on the counter, if the buffer is larger than we have start times then we need to
        # sample some start times multiple times. Here all valid start points are taken, because we can't use the last
        # N starting points (see comment above)
        avail_t_start = len([i for i in max_t if i > 0])

        if avail_t_start < self._buffer_size:
            # make sure all available starting points are sampled at least once
            idx1 = pt.multinomial(weights, avail_t_start)

            # then sample the remaining starting points 'replacement=True' leads
            idx2 = pt.multinomial(weights, self._buffer_size - avail_t_start, replacement=True)

            # get the occurrences of each sampled index and update the counter with them
            idx, amount = pt.unique(pt.cat([idx1, idx2]), return_counts=True)

            # update the counter with the sampled starting points
            self._counter[idx] += amount

            # overwrite the idx with the sampled starting points, because the current idx is unique
            idx = pt.cat([idx1, idx2])
        else:
            idx = pt.multinomial(weights, self._buffer_size)

            # update the counter
            self._counter[idx] += 1

        for i, env in enumerate(self.envs):
            # reset the environment but leave all time folders, so we can initialize a new start time
            env.reset()

            # compute the start & end time (and trajectory length), so that in each episode statistically all parts of
            # the simulation are seen by the agent (trajectory length = const. for all episodes)
            env.start_time = start_times[idx[i]]

            # compute the end time of the simulation based on the dt and traj. length -> the actual length may differ if
            # dt != const.
            env.end_time = round(env.start_time + self._len_traj * dt, 6)

            # now set the beginning of control
            env.start_control = env.start_time

    def clean(self):
        for env in self.envs:
            proc = Popen([f"./{env.clean_script}"], cwd=env.path)
            proc.wait()

    def save_trajectories(self):
        pt.save([env.observations for env in self.envs], join(self._path, f"observations_{self._n_fills}.pt"))

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
