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


def _parse_trajectory(path: str) -> DataFrame:
    names = ["t", "prob", "interpolateCorr"]
    tr = pd.read_table(path, sep=",", header=0, names=names)
    return tr


class GAMGSolverSettings(Environment):
    def __init__(self, r1: float = 0.0, r2: float = 1.0):
        super(GAMGSolverSettings, self).__init__(
            join(TESTCASE_PATH, "cylinder2D"), "Allrun.pre",
            "Allrun", "Allclean", mpi_ranks=4, n_states=7, n_actions=1
        )
        self._r1 = r1
        self._r2 = r2
        self._initialized = False
        self._start_time = 0
        self._end_time = 0.01
        self._control_interval = 0.01
        self._train = True
        self._seed = 0
        self._action_bounds = [0, 1]
        self._policy = "policy.pt"

    def _reward(self, t: pt.Tensor, t_tot: pt.Tensor) -> pt.Tensor:
        # TODO: define proper reward fct, independently of N_cpu, env, ...
        # due to sampling rewards change a lot -> scale reward per dt with t_exec for complete trajectory, so that the
        # reward is the contribution of each time step to the total required execution time
        return ((self._r2 * t.log().abs()) - self._r1) / t_tot.log().abs()

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
            tr = _parse_trajectory(join(self.path, "trajectory.txt"))

            # load the residual data
            residuals_path = glob(join(self.path, "postProcessing", "residuals", "*", "agentSolverSettings.dat"))[0]
            residuals = _parse_residuals(residuals_path)

            # convert the convergence rates etc. to log (compare to agentSolverSettings.C, predictSettings())
            for name in ["p_initial", "p_rate_median", "p_rate_max", "p_rate_min"]:
                residuals[name] = np.abs(np.log(residuals[name]))

            obs["states"] = pt.from_numpy(residuals[residuals.keys()].values)
            # we need to convert the ints to float, otherwise error when printing the statistics
            obs["actions"] = pt.from_numpy(tr["interpolateCorr"].values).float()
            obs["t_per_dt"] = pt.from_numpy(cpu_times["t_per_dt"].values)
            obs["t_cumulative"] = pt.from_numpy(cpu_times["t_tot"].values)
            obs["t"] = pt.from_numpy(cpu_times["t"].values)
            obs["probability"] = pt.from_numpy(tr["prob"].values)
            obs["rewards"] = self._reward(obs["t_per_dt"], obs["t_cumulative"][-1])

        except Exception as e:
            logging.warning("Could not parse observations: ", e)
        finally:
            return obs

    def reset(self):
        files = ["log.pimpleFoam", "finished.txt", "trajectory.txt"]
        for f in files:
            f_path = join(self.path, f)
            if isfile(f_path):
                remove(f_path)
        post = join(self.path, "postProcessing")
        if isdir(post):
            rmtree(post)
