""" Example training script.
"""
import sys
from glob import glob
from time import time
import logging
import argparse

from torch import manual_seed, cuda
from shutil import copytree, rmtree
from os import makedirs, environ, system, chdir, remove
from os.path import join, exists

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

import torch as pt
from drlfoam.environment import GAMGSolverSettings
from drlfoam.agent import PPOAgent
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig


logging.basicConfig(level=logging.INFO)


SIMULATION_ENVIRONMENTS = {
    "cylinder2D": GAMGSolverSettings,
    "weirOverflow": GAMGSolverSettings
}

DEFAULT_CONFIG = {
    "cylinder2D": {
        "policy_dict": {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        },
        "value_dict": {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        }
    },
    "weirOverflow": {
        "policy_dict": {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        },
        "value_dict": {
            "n_layers": 2,
            "n_neurons": 64,
            "activation": pt.nn.functional.relu
        }
    }
}


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    reward_msg = f"Reward mean/min/max: {sum(rt)/len(rt):2.4f}/{min(rt):2.4f}/{max(rt):2.4f}"
    action_mean_msg = f"Mean action mean/min/max: {sum(at_mean)/len(at_mean):2.4f}/{min(at_mean):2.4f}/{max(at_mean):2.4f}"
    action_std_msg = f"Std. action mean/min/max: {sum(at_std)/len(at_std):2.4f}/{min(at_std):2.4f}/{max(at_std):2.4f}"
    logging.info("\n".join((reward_msg, action_mean_msg, action_std_msg)))


def parseArguments():
    ag = argparse.ArgumentParser()
    ag.add_argument("-o", "--output", required=False, default="test_training", type=str,
                    help="Where to run the training.")
    ag.add_argument("-e", "--environment", required=False, default="local", type=str,
                    help="Use 'local' for local and 'slurm' for cluster execution.")
    ag.add_argument("-i", "--iter", required=False, default=20, type=int,
                    help="Number of training episodes.")
    ag.add_argument("-r", "--runners", required=False, default=4, type=int,
                    help="Number of runners for parallel execution.")
    ag.add_argument("-b", "--buffer", required=False, default=8, type=int,
                    help="Reply buffer size.")
    ag.add_argument("-t", "--timeout", required=False, default=1e15, type=int,
                    help="Maximum allowed runtime of a single simulation in seconds.")
    ag.add_argument("-m", "--manualSeed", required=False, default=0, type=int,
                    help="seed value for torch")
    ag.add_argument("-c", "--checkpoint", required=False, default="", type=str,
                    help="Load training state from checkpoint file.")
    ag.add_argument("-s", "--simulation", required=False, default="cylinder2D", type=str,
                    help="Select the simulation environment.")
    args = ag.parse_args()
    return args


def main(args):
    # settings
    training_path = args.output
    episodes = args.iter
    buffer_size = args.buffer
    n_runners = args.runners
    executer = args.environment
    timeout = args.timeout
    checkpoint_file = args.checkpoint
    simulation = args.simulation

    # set end_time for base case depending on environment (if debug this will be overwritten by the finish parameter)
    end_time = 0.8 if simulation == "cylinder2D" else 81

    # ensure reproducibility
    manual_seed(args.manualSeed)
    if cuda.is_available():
        cuda.manual_seed_all(args.manualSeed)

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    if not simulation in SIMULATION_ENVIRONMENTS.keys():
        msg = (f"Unknown simulation environment {simulation}" +
               "Available options are:\n\n" +
               "\n".join(SIMULATION_ENVIRONMENTS.keys()) + "\n")
        raise ValueError(msg)
    if not exists(join(training_path, "base")):
        copytree(join(BASE_PATH, "openfoam", "test_cases", simulation),
                 join(training_path, "base"), dirs_exist_ok=True)
    env = SIMULATION_ENVIRONMENTS[simulation]()
    env.path = join(training_path, "base")

    # if debug active -> add execution of bashrc to Allrun scripts, because otherwise the path to openFOAM is not set
    if hasattr(args, "debug"):
        args.set_openfoam_bashrc(path=env.path)

        # in case of debug we can manually set the end_time, since we don't need the full base case for testing stuff
        end_time = args.finish

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        if simulation == "weirOverflow":
            t_max = "02:30:00"
            env.mpi_ranks = 4
        else:
            t_max = "01:00:00"
        config = SlurmConfig(
            n_tasks=env.mpi_ranks, n_nodes=1, partition="standard", time=t_max,
            modules=["singularity/latest", "mpi/openmpi/4.1.1/gcc"], job_name="drl_train"
        )
        """
        # for AWS
        config = SlurmConfig(n_tasks=env.mpi_ranks, n_nodes=1, partition="queue-1", time="03:00:00",
                             modules=["openmpi/4.1.5"], constraint = "c5a.24xlarge", job_name="drl_train",
                             commands_pre=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc",
                             "source /fsx/drlfoam/setup-env"], commands=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc",
                             "source /fsx/drlfoam/setup-env"])
        """
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, env.action_bounds[0], env.action_bounds[1], env.n_output,
                     **DEFAULT_CONFIG[simulation])

    # load checkpoint if provided
    if checkpoint_file:
        logging.info(f"Loading checkpoint from file {checkpoint_file}")
        agent.load_state(join(training_path, checkpoint_file))
        starting_episode = agent.history["episode"][-1] + 1
        buffer._n_fills = starting_episode
    else:
        starting_episode = 0
        buffer.base_env.end_time = end_time
        buffer.prepare()

    # load the time steps and corresponding CPU times of the base case for reward function
    env.set_cpu_times_base()

    buffer.reset()

    # begin training
    start_time = time()
    for e in range(starting_episode, episodes):
        logging.info(f"Start of episode {e}")
        buffer.fill()
        states, actions, rewards = buffer.observations
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save_state(join(training_path, f"checkpoint_{e}.pt"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        if not e == episodes - 1:
            buffer.reset()
        # delete all slurm files, not used because they just display the path to OF container
        [remove(s) for s in glob(join(BASE_PATH, "examples", "slurm-*.out")) if executer == "slurm"]
    logging.info(f"Training time (s): {time() - start_time}")


class RunTrainingInDebugger:
    """
    class for providing arguments when running script in IDE (e.g. for debugging). The ~/.bashrc is not executed when
    not running the training from terminal, therefore the environment variables need to be set manually in the Allrun
    scripts
    """

    def __init__(self, episodes: int = 2, runners: int = 2, buffer: int = 2, finish: float = 5.0,
                 seed: int = 0, timeout: int = 1e15, out_dir: str = "examples/TEST", checkpoint: str = ""):
        self.command = ". /usr/lib/openfoam/openfoam2206/etc/bashrc"
        self.output = out_dir
        self.iter = episodes
        self.runners = runners
        self.buffer = buffer
        self.finish = finish
        self.environment = "local"
        self.debug = True
        self.manualSeed = seed
        self.timeout = timeout
        self.checkpoint = checkpoint
        # self.simulation = "cylinder2D"
        self.simulation = "weirOverflow"

    def set_openfoam_bashrc(self, path: str):
        system(f"sed -i '5i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun.pre")
        system(f"sed -i '4i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun")


if __name__ == "__main__":
    # option for running the training in IDE, e.g. in debugger
    DEBUG = False

    if not DEBUG:
        main(parseArguments())
        exit(0)

    else:
        # for debugging purposes, set environment variables for the current directory
        environ["DRL_BASE"] = "/home/janis/Hiwi_ISM/results_drlfoam_MB/drlfoam/"
        environ["DRL_TORCH"] = "".join([environ["DRL_BASE"], "libtorch/"])
        environ["DRL_LIBBIN"] = "".join([environ["DRL_BASE"], "/openfoam/libs/"])
        sys.path.insert(0, environ["DRL_BASE"])
        sys.path.insert(0, environ["DRL_TORCH"])
        sys.path.insert(0, environ["DRL_LIBBIN"])

        # set paths to openfoam
        BASE_PATH = environ.get("DRL_BASE", "")
        sys.path.insert(0, BASE_PATH)
        environ["WM_PROJECT_DIR"] = "/usr/lib/openfoam/openfoam2206"
        sys.path.insert(0, environ["WM_PROJECT_DIR"])
        chdir(BASE_PATH)

        # test MB-DRL on local machine for cylinder2D
        # d_args = RunTrainingInDebugger(episodes=2, runners=2, buffer=2, finish=0.1, seed=0,
        #                                out_dir="examples/TEST")

        # test MB-DRL on local machine for weirOverflow
        d_args = RunTrainingInDebugger(episodes=20, runners=1, buffer=1, finish=70, seed=0,
                                       out_dir="examples/TEST")

        # run PPO training
        main(d_args)
