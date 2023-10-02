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
                     get_time_folders, get_latest_time, replace_line_latest, fetch_line_from_file)


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def _parse_cpu_times(path: str) -> DataFrame:
    times = read_csv(path, sep="\t", comment="#", header=None, names=["t", "t_tot", "t_per_dt"], usecols=[0, 1, 3])
    return times


def _parse_residuals(path: str) -> DataFrame:
    # names = ["p_initial", "p_rate_median", "p_rate_max", "p_rate_min", "p_sum_iters", "p_max_iters", "p_pimple_iters"]
    names = ["p_initial", "p_rate_median", "p_rate_max", "p_rate_min", "p_ratio_iter", "p_ratio_pimple_iters"]
    # residuals = read_csv(path, sep="\t", comment="#", header=None, names=names, usecols=range(2, 9))
    residuals = read_csv(path, sep="\t", comment="#", header=None, names=names, usecols=range(2, 8))

    return residuals


def _parse_trajectory(path: str, n_outputs: int, n_actions: int) -> DataFrame:
    names = ["t"] + [f"prob{i}" for i in range(n_outputs)] + [f"action{i}" for i in range(n_actions)]
    tr = pd.read_table(path, sep=",", header=0, names=names)
    return tr


class GAMGSolverSettings(Environment):
    def __init__(self, r1: float = 100.0, r2: float = 1.0):
        super(GAMGSolverSettings, self).__init__(
            join(TESTCASE_PATH, "cylinder2D"), "Allrun.pre",
            "Allrun", "Allclean", mpi_ranks=2, n_states=6, n_actions=2, n_output=7
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

        # new reward fct
        return t_cpu_base / t_cpu

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

            # convert the and scale convergence rates etc (compare to agentSolverSettings.C, predictSettings())
            # keys = ["p_initial", "p_rate_median", "p_rate_max", "p_rate_min", "p_ratio_iter", "p_ratio_pimple_iters"]
            residuals = pt.from_numpy(residuals[residuals.keys()].values)
            residuals[:, 0] = (-residuals[:, 0].log() - 1) / 10
            residuals[:, 1] = (pt.sigmoid(residuals[:, 1]) - 0.5) / 1.5e-4
            residuals[:, 2] = (-residuals[:, 2].log() - 1) / 10
            residuals[:, 3] = (pt.sigmoid(residuals[:, 3]) - 0.5) / 2e-5

            # the time stuff is written out every as batch n time steps, so if OF crashes then we need to make sure the
            # amount of data is consistent, the crash can occur during writing, so make sure tha last line was fully
            # written, otherwise ignore the last line
            if not cpu_times["t_per_dt"].tail(1).isnull().item:
                idx = len(cpu_times["t_per_dt"])
            else:
                idx = len(cpu_times["t_per_dt"]) - 1

            obs["states"] = residuals[:idx, :]
            # we need to convert the ints of the actions to float, otherwise error when printing the statistics,
            # however, the print_statistic is not really useful at the moment anyway since we have different actions
            obs["actions"] = pt.stack([pt.from_numpy(tr[f"action{i}"].values).float() for i in range(self._n_actions)],
                                      dim=1)[:idx, :]
            obs["t_per_dt"] = pt.from_numpy(cpu_times["t_per_dt"].values)[:idx]
            obs["t_cumulative"] = pt.from_numpy(cpu_times["t_tot"].values)[:idx]
            obs["t"] = pt.from_numpy(cpu_times["t"].values)[:idx]

            # prob0 corresponds to 'interpolateCorrection', all other probs to smoother
            obs["probability"] = pt.stack([pt.from_numpy(tr[f"prob{i}"].values) for i in range(self._n_outputs)],
                                          dim=1)[:idx, :]
            obs["rewards"] = self._reward(obs["t_per_dt"], obs["t"], obs["t_cumulative"][-1])

            return obs

        except Exception as e:
            logging.warning("Could not parse observations: ", e)
            logging.info(f"start time of {self.path} was: t_start = {self._start_time}")
            exit()
        # finally:
        #     return obs

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
        
        # check if the time step is const. or based on Courant number
        var_dt = fetch_line_from_file(join(self._path, "system", "controlDict"), "adjustTimeStep ")

        # if the command 'adjustTimeStep' is present, check if it is set to 'on' or 'off'
        var_dt = var_dt.split(" ")[-1].split(";")[0] if var_dt is not None else var_dt

        # set the flag for const. dt accordingly
        self._const_dt = False if var_dt is not None and var_dt == "on" else True
