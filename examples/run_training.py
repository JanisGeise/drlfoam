""" Example training script.
"""
import sys
import pickle
import argparse
import torch as pt

from glob import glob
from time import time
from os.path import join
from torch import manual_seed
from shutil import copytree, rmtree
from os import makedirs, chdir, environ, system

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.agent import PPOAgent
from drlfoam.environment import RotatingCylinder2D
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig

from examples.get_number_of_probes import get_number_of_probes
from drlfoam.environment.env_model_rotating_cylinder_new_training_routine import *
from drlfoam.environment.correct_env_model_error import train_correction_models, predict_traj_for_model_error


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    print("Reward mean/min/max: ", sum(rt)/len(rt), min(rt), max(rt))
    print("Mean action mean/min/max: ", sum(at_mean) /
          len(at_mean), min(at_mean), max(at_mean))
    print("Std. action mean/min/max: ", sum(at_std) /
          len(at_std), min(at_std), max(at_std))


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
    ag.add_argument("-f", "--finish", required=False, default=8.0, type=float,
                    help="End time of the simulations.")
    ag.add_argument("-t", "--timeout", required=False, default=1e15, type=int,
                    help="Maximum allowed runtime of a single simulation in seconds.")
    ag.add_argument("-s", "--seed", required=False, default=0, type=int,
                    help="seed value for torch")
    args = ag.parse_args()
    return args


def main(args):
    # settings
    training_path = args.output
    episodes = args.iter
    buffer_size = args.buffer
    n_runners = args.runners
    end_time = args.finish
    executer = args.environment
    timeout = args.timeout

    # ensure reproducibility
    manual_seed(args.seed)

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # get number of probes defined in the control dict and init env. correctly
    n_probes = get_number_of_probes(os.getcwd())

    # make a copy of the base environment
    copytree(join(BASE_PATH, "openfoam", "test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D(n_probes=n_probes)
    env.path = join(training_path, "base")

    # if debug active -> add execution of bashrc to Allrun scripts, because otherwise the path to openFOAM is not set
    if hasattr(args, "debug"):
        args.set_openfoam_bashrc(path=env.path)
        n_input_time_steps = args.n_input_time_steps
        debug = args.debug
    else:
        n_input_time_steps = 30
        debug = False

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        config = SlurmConfig(
            n_tasks=2, n_nodes=1, partition="standard", time="00:30:00",
            modules=["singularity/latest", "mpi/openmpi/4.1.1/gcc"]
        )
        """
        # for AWS
        config = SlurmConfig(n_tasks=2, n_nodes=1, partition="c6i", time="00:30:00", modules=["openmpi/4.1.1"],
                             commands=["source /fsx/OpenFOAM/OpenFOAM-v2206/etc/bashrc",
                                       "source /fsx/drlfoam/setup-env"])
        """
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # execute Allrun.pre script and set new end_time
    buffer.prepare()
    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = end_time
    buffer.reset()

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # len_traj = length of the trajectory, assuming constant sample rate of 100 Hz (default value)
    # NOTE: at Re != 100, the parameter len_traj needs to be adjusted accordingly since the simulation is only run to
    # the same dimensionless time but here the pysical time is required, e.g. 5*int(...) for Re = 500
    len_traj, obs_cfd, n_models = 5 * int(100 * round(end_time - buffer.base_env.start_time, 1)), [], 5

    # corr_traj = flag for using additional models to correct the MB-trajectories based on MF-trajectories
    corr_traj = False

    # begin training
    start_time = time()
    for e in range(episodes):
        print(f"Start of episode {e}")

        # for debugging -> if episode of crash reached: pause execution in order to set breakpoints (so debugger can run
        # without breakpoints / supervisions up to this point)
        if debug:
            if e == args.crashed_in_e:
                _ = input(f"reached episode {e} (= episode where training crashes) - set breakpoints!")

        # every 5th episode sample from CFD
        if e == 0 or e % 5 == 0:
            # save path of CFD episodes -> models should be trained with all CFD data available
            obs_cfd.append("".join([training_path + f"/observations_{e}.pkl"]))

            # set episode for save_trajectory() method, because n_fills is now updated only every 5th episode
            if e != 0:
                buffer._n_fills = e

            buffer.fill()
            states, actions, rewards = buffer.observations

            # in 1st episode: CFD data is used to train environment models for 1st time
            if e == 0:
                cl_p_models, cd_models, l, obs = wrapper_train_env_model_ensemble(training_path, obs_cfd, len_traj,
                                                                                  env.n_states, buffer_size, n_models,
                                                                                  n_time_steps=n_input_time_steps)

            # ever 5th episode: models are loaded and re-trained based on CFD data of the current & last CFD episode
            else:
                cl_p_models, cd_models, l, obs = wrapper_train_env_model_ensemble(training_path, obs_cfd, len_traj,
                                                                                  env.n_states, buffer_size, n_models,
                                                                                  load=True,
                                                                                  n_time_steps=n_input_time_steps)

            # train the models for correcting the cl- and cd trajectories, until ~e = 40, training runs stable without
            # correction, then it gets unstable. When correcting, the training is very stable, but rewards are not
            # increasing that much, so combine these two approaches
            if e >= 35 and corr_traj:
                min_max = {"cd": obs["min_max_cd"], "cl": obs["min_max_cl"], "states": obs["min_max_states"],
                           "actions": obs["min_max_actions"]}
                corr_out = predict_traj_for_model_error(cl_p_models, cd_models, training_path, obs_cfd[-1],
                                                        n_input_time_steps, env.n_states, len_traj, min_max)

                # in case there are no / not enough trajectories in the current episode, don't update the models
                if corr_out[0] and len(corr_out[0]) >= 2:
                    corr_model_cd, corr_model_cl, corr_model_p = train_correction_models(corr_out[0], corr_out[1],
                                                                                         training_path, buffer_size,
                                                                                         len_traj, min_max)
            else:
                corr_model_cd, corr_model_cl, corr_model_p = None, None, None

            # save train- and validation losses of the environment models in N_models > 1 (1st model runs different
            # amounts of epochs, ...)
            if n_models == 1:
                pass
            else:
                losses = {"train_loss_cl_p": l[:, 0, 0, :], "train_loss_cd": l[:, 0, 1, :],
                          "val_loss_cl_p": l[:, 1, 0, :], "val_loss_cd": l[:, 1, 1, :]}
                save_trajectories(training_path, e, losses, name="/env_model_loss_")

            # all observations are saved in obs_resorted, so reset buffer
            buffer.reset()

        # fill buffer with trajectories generated by the environment models
        else:
            if e > 35 and corr_traj:
                corr = True
            else:
                corr = False
            # generate trajectories from initial states using policy from previous episode, fill model buffer with them
            predicted_traj = fill_buffer_from_models(cl_p_models, cd_models, e, training_path,
                                                     observation=obs, n_probes=env.n_states,
                                                     n_input=n_input_time_steps, len_traj=len_traj,
                                                     buffer_size=buffer_size, corr_cd=corr_model_cd,
                                                     corr_cl=corr_model_cl, corr_p=corr_model_p, correct_traj=corr)

            # if len(predicted_traj) < buffer size -> discard trajectories from models and go back to CFD
            if len(predicted_traj) < buffer_size:
                buffer._n_fills = e
                buffer.fill()
                states, actions, rewards = buffer.observations
                obs_cfd.append("".join([training_path + f"/observations_{e}.pkl"]))

                # re-train environment models to avoid failed trajectories in the next episode
                cl_p_models, cd_models, l, obs = wrapper_train_env_model_ensemble(training_path, obs_cfd, len_traj,
                                                                                  env.n_states, buffer_size, n_models,
                                                                                  load=True, e_re_train=100,
                                                                                  e_re_train_cd=100,
                                                                                  n_time_steps=n_input_time_steps)

            else:
                # save the generated trajectories, for now without model buffer instance
                save_trajectories(training_path, e, predicted_traj)

                # states, actions and rewards required for PPO-training, they are already re-scaled when generated
                states = [predicted_traj[traj]["states"] for traj in range(buffer_size)]
                actions = [predicted_traj[traj]["actions"] for traj in range(buffer_size)]
                rewards = [predicted_traj[traj]["rewards"] for traj in range(buffer_size)]

        # in case no trajectories in CFD converged, use trajectories of the last CFD episodes to train policy network
        if not actions and e >= 5:
            try:
                n_traj = obs["actions"].size()[1]
                traj_n = pt.randint(0, n_traj, size=(buffer_size,))

                # actions and states stored in obs are scaled to interval [0 ,1], so they need to be re-scaled
                actions = [denormalize_data(obs["actions"][:, t.item()], obs["min_max_actions"]) for t in traj_n]
                states = [denormalize_data(obs["states"][:, :, t], obs["min_max_states"]) for t in traj_n]

                # rewards are not scaled to [0, 1] when loading the data since they are not used for env. models
                rewards = [obs["rewards"][:, t.item()] for t in traj_n]

            # if we don't have any trajectories generated within the last 3 CFD episodes, it doesn't make sense to
            # continue with the training
            except IndexError as e:
                print(f"[run_training.py]: {e}, could not find any valid trajectories from the last 3 CFD episodes!"
                      "\nAborting training.")
                exit(0)

        # continue with original PPO-training routine
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save(join(training_path, f"policy_{e}.pkl"),
                   join(training_path, f"value_{e}.pkl"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        buffer.reset()
    print(f"Training time (s): {time() - start_time}")

    # save training statistics
    with open(join(training_path, "training_history.pkl"), "wb") as f:
        pickle.dump(agent.history, f, protocol=pickle.HIGHEST_PROTOCOL)


class RunTrainingInDebugger:
    """
    class for providing arguments when running script in IDE (e.g. for debugging). The ~/.bashrc is not executed when
    not running the training from terminal, therefore the environment variables need to be set manually in the Allrun
    scripts
    """

    def __init__(self, episodes: int = 2, runners: int = 2, buffer: int = 2, finish: float = 5.0,
                 n_input_time_steps: int = 30, seed: int = 0, timeout: int = 1e15, crashed_in_e: int = 5,
                 out_dir: str = "examples/TEST_for_debugging"):
        self.command = ". /usr/lib/openfoam/openfoam2206/etc/bashrc"
        self.output = out_dir
        self.iter = episodes
        self.runners = runners
        self.buffer = buffer
        self.finish = finish
        self.environment = "local"
        self.debug = True
        self.n_input_time_steps = n_input_time_steps
        self.seed = seed
        self.timeout = timeout
        self.crashed_in_e = crashed_in_e

    def set_openfoam_bashrc(self, path: str):
        system(f"sed -i '5i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun.pre")
        system(f"sed -i '4i # source bashrc for openFOAM for debugging purposes\\n{self.command}' {path}/Allrun")


if __name__ == "__main__":
    # option for running the training in IDE, e.g. in debugger
    DEBUG = True

    if not DEBUG:
        main(parseArguments())
        exit(0)

    else:
        # for debugging purposes, set environment variables for the current directory
        environ["DRL_BASE"] = "/media/janis/Daten/Studienarbeit/drlfoam/"
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

        # test MB-DRL on local machine
        d_args = RunTrainingInDebugger(episodes=80, runners=4, buffer=4, finish=5, n_input_time_steps=30, seed=0,
                                       out_dir="examples/TEST/",
                                       # out_dir="examples/e80_r8_b8_f6_Nprobes24_corrModels/seed0/",
                                       crashed_in_e=90)
        assert d_args.finish > 4, "finish time needs to be > 4s, (the first 4sec are uncontrolled)"
        assert d_args.buffer >= 4, f"buffer needs to >= 4 in order to split trajectories for training and sampling" \
                                   f" initial states"

        # run PPO training
        main(d_args)

        # clean up afterwards
        for dirs in [d for d in glob(d_args.output + "/copy_*")]:
            rmtree(dirs)
        rmtree(d_args.output + "/base")

        try:
            rmtree(d_args.output + "/cd_model")
            rmtree(d_args.output + "/cl_p_model")
            rmtree(d_args.output + "/cl_error_model")
            rmtree(d_args.output + "/cd_error_model")
        except FileNotFoundError:
            print("no directories for environment models found.")
