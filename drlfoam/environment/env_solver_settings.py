from typing import Tuple
from os import remove
from os.path import join, isfile, isdir
from glob import glob
from re import sub
from io import StringIO
from shutil import rmtree
import logging

import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
import torch as pt
from .environment import Environment
from ..constants import TESTCASE_PATH, DEFAULT_TENSOR_TYPE
from ..utils import (check_pos_int, check_pos_float, replace_line_in_file,
                     get_time_folders, get_latest_time, replace_line_latest)


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def _parse_cpu_times(path: str) -> DataFrame:
    times = read_csv(path, sep="\t", comment="#", header=None, names=["t", "t_tot", "t_per_dt"], usecols=[0, 1, 3])
    return times


def _parse_residuals(path: str) -> DataFrame:
    names = ["p_initial", "p_rate_median", "p_rate_max", "p_rate_min", "p_sum_iters", "p_max_iters", "p_pimple_iters"]
    residuals = read_csv(path, sep="\t", comment="#", header=None, names=names, usecols=range(2, 9))

    return residuals


def _parse_trajectory(path: str, n_outputs: int, n_actions: int) -> DataFrame:
    names = ["t"] + [f"prob{i}" for i in range(n_outputs)] + [f"action{i}" for i in range(n_actions)]
    tr = pd.read_table(path, sep=",", header=0, names=names)
    return tr


class GAMGSolverSettings(Environment):
    def __init__(self, r1: float = 100.0, r2: float = 1.0):
        super(GAMGSolverSettings, self).__init__(
            join(TESTCASE_PATH, "cylinder2D"), "Allrun.pre",
            "Allrun", "Allclean", mpi_ranks=2, n_states=7, n_actions=2, n_output=7
        )
        self._const_dt = None
        self._t_base = None
        self._r1 = r1
        self._r2 = r2
        self._initialized = False
        self._start_time = 0
        self._end_time = 0.01
        self._control_interval = 0.01
        self._train = True
        self._seed = 0
        self._action_bounds = [0, 6]    # not used because so far we don't have constraints
        self._n_outputs = 7             # output neurons for policy network
        self._policy = "policy.pt"

    def _reward(self, t_cpu: pt.Tensor, dt: pt.Tensor, t_traj: pt.Tensor) -> pt.Tensor:
        # if the time step is const., we can directly compute the difference
        if self._const_dt:
            # determine the start idx of the dt of the base case from which the current trajectory is starting
            idx_start = pt.where(self._t_base[:, 0] == dt[0])[0]

            # save the part of the base case which is present in the trajectory for clarity
            t_cpu_base = self._t_base[idx_start:idx_start+dt.size()[0], 1]
        else:
            # otherwise dt != const., so we have to interpolate the time steps and corresponding CPU time per time step
            # of the base case. We want to interpolate the dt of the base case on the dt of the trajectory
            t_cpu_base = pt.from_numpy(np.interp(dt, self._t_base[:, 0], self._t_base[:, 1]))

        # scale with mean execution time per dt of the base case
        # return (t_cpu_base - t_cpu) / pt.mean(t_cpu_base) * self._r1

        # duration to compute the trajectory for normalization maybe better if multiple envs of different complexity
        # should be combined in PPO-training routine. r1 = 100 in order to have rewards in a range of [0, 1]
        return (t_cpu_base - t_cpu) / t_traj * self._r1

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        check_pos_float(value, "start_time", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "startTime",
            f"startTime       {value};",
            startwith_keyword=True
        )
        self._start_time = value

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, value: float):
        check_pos_float(value, "end_time", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "endTime ",
            f"endTime         {value};"
        )
        self._end_time = value

    @property
    def control_interval(self) -> int:
        return self._control_interval

    @control_interval.setter
    def control_interval(self, value: int):
        check_pos_float(value, "control_interval")
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "executeInterval",
            f"        executeInterval {value};",
        )
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "writeInterval",
            f"        writeInterval   {value};",
        )
        self._control_interval = value

    @property
    def start_control(self) -> float:
        return self._start_control

    @start_control.setter
    def start_control(self, value: float):
        check_pos_float(value, "timeStart ", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "timeStart ",
            f"        timeStart {value};"
        )
        self._start_control = value

    @property
    def actions_bounds(self) -> float:
        return self._action_bounds

    @actions_bounds.setter
    def action_bounds(self, value: float):
        proc = True if self.initialized else False
        new = f"        absOmegaMax     {value:2.4f};"
        replace_line_latest(self.path, "U", "absOmegaMax", new, proc)
        self._action_bounds = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        check_pos_int(value, "seed", with_zero=True)
        new = f"        seed     {value};"
        replace_line_in_file(join(self.path, "system", "controlDict"), "seed", new)
        self._seed = value

    @property
    def policy(self) -> str:
        return self._policy

    @policy.setter
    def policy(self, value: str):
        # proc = True if self.initialized else False
        new = f"        policy     {value};"
        replace_line_in_file(join(self.path, "system", "controlDict"), "policy", new)
        self._policy = value

    @property
    def train(self) -> bool:
        return self._train

    @train.setter
    def train(self, value: bool):
        value_cpp = "true" if value else "false"
        new = f"        train           {value_cpp};"
        replace_line_in_file(join(self.path, "system", "controlDict"), "train", new)
        self._train = value

    @property
    def observations(self) -> dict:
        obs = {}
        try:
            # load the execution times per time step
            t_exec_path = glob(join(self.path, "postProcessing", "time", "*", "timeInfo.dat"))[0]
            cpu_times = _parse_cpu_times(t_exec_path)

            # load the trajectory containing the probability and action
            tr = _parse_trajectory(join(self.path, "trajectory.txt"), self._n_outputs, self._n_actions)

            # load the residual data
            residuals_path = glob(join(self.path, "postProcessing", "residuals", "*", "agentSolverSettings.dat"))[0]
            residuals = _parse_residuals(residuals_path)

            # convert the convergence rates etc. to log (compare to agentSolverSettings.C, predictSettings())
            for name in ["p_initial", "p_rate_median", "p_rate_max", "p_rate_min"]:
                residuals[name] = np.abs(np.log(residuals[name]))

            obs["states"] = pt.from_numpy(residuals[residuals.keys()].values)
            # we need to convert the ints of the actions to float, otherwise error when printing the statistics,
            # however, the print_statistic is not really useful at the moment anyway since we have different actions
            obs["actions"] = pt.stack([pt.from_numpy(tr[f"action{i}"].values).float() for i in range(self._n_actions)],
                                      dim=1)
            obs["t_per_dt"] = pt.from_numpy(cpu_times["t_per_dt"].values)
            obs["t_cumulative"] = pt.from_numpy(cpu_times["t_tot"].values)
            obs["t"] = pt.from_numpy(cpu_times["t"].values)

            # prob0 corresponds to 'interpolateCorrection', all other probs to smoother
            obs["probability"] = pt.stack([pt.from_numpy(tr[f"prob{i}"].values) for i in range(self._n_outputs)])
            obs["rewards"] = self._reward(obs["t_per_dt"], obs["t"], obs["t_cumulative"][-1])

        except Exception as e:
            logging.warning("Could not parse observations: ", e)
        finally:
            return obs

    def reset(self):
        # if we are not in base case, then there should be a log-file from the solver used (e.g. interFoam / pimpleFoam)
        solver_log = glob(join(self.path, "log.*Foam"))
        if solver_log:
            files = [f"log.{solver_log[0].split('.')[-1]}", "finished.txt", "trajectory.txt"]
        else:
            # otherwise we are in the base case and have only a log.*Foam.pre, which we don't want to remove
            files = ["finished.txt", "trajectory.txt"]
        for f in files:
            f_path = join(self.path, f)
            if isfile(f_path):
                remove(f_path)
        post = join(self.path, "postProcessing")
        if isdir(post):
            rmtree(post)

    def set_cpu_times_base(self):
        # if counter is None, then we didn't load the CPU times and time steps of base case as well, so load them
        # load the CPU times per time step and physical time of the base case as reference
        self._t_base = _parse_cpu_times(glob(join(self._path, "postProcessing", "time", "*", "timeInfo.dat"))[0])
        self._t_base.drop("t_tot", axis=1, inplace=True)

        # convert to tensor in order to do computations later easier
        self._t_base = pt.tensor(self._t_base.values)
        
        # check if the time step is const. or based on Courant number, therefore crete evenly spaced tensor based on the
        # loaded dt and check if they are the same, if not then we don't have a const. dt,
        # rtol=1e-12 << dt to make sure there are no round-off errors when comparing to the tmp tensor
        tmp = pt.linspace(self._t_base[0, 0], self._t_base[-1, 0], self._t_base.size()[0])
        self._const_dt = True if pt.allclose(tmp, self._t_base[:, 0], rtol=1e-12) else False
