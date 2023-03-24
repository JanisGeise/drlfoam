""" Example training script.
"""
import sys
import argparse

from glob import glob
from time import time
from os.path import join
from torch import manual_seed
from shutil import copytree, rmtree
from os import makedirs, chdir, environ, system, getcwd

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.agent import PPOAgent
from drlfoam.environment import RotatingCylinder2D
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig

from examples.get_number_of_probes import get_number_of_probes
from drlfoam.environment.env_model_rotating_cylinder import *


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
    ag.add_argument("-c", "--checkpoint", required=False, default="", type=str,
                    help="Load training state from checkpoint file.")
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
    checkpoint_file = args.checkpoint

    # ensure reproducibility
    manual_seed(args.seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(args.seed)

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # get number of probes defined in the control dict and init env. correctly
    n_probes = get_number_of_probes(getcwd())

    # make a copy of the base environment
    copytree(join(BASE_PATH, "openfoam", "test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D(n_probes=n_probes)
    env.path = join(training_path, "base")

    # if debug active -> add execution of bashrc to Allrun scripts, because otherwise the path to openFOAM is not set
    if hasattr(args, "debug"):
        args.set_openfoam_bashrc(path=env.path)
        env_model = SetupEnvironmentModel(n_input_time_steps=args.n_input_time_steps, path=training_path)
    else:
        env_model = SetupEnvironmentModel(path=training_path)

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

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # load checkpoint if provided
    if checkpoint_file:
        print(f"Loading checkpoint from file {checkpoint_file}")
        agent.load_state(join(training_path, checkpoint_file))
        starting_episode = agent.history["episode"][-1] + 1
        buffer._n_fills = starting_episode
    else:
        starting_episode = 0
        buffer.prepare()

    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = end_time
    buffer.reset()
    env_model.last_cfd = starting_episode

    # begin training
    env_model.start_training = time()
    for e in range(starting_episode, episodes):
        print(f"Start of episode {e}")

        # if only 1 model is used, switch every 4th episode to CFD, else determine switching based on model performance
        if env_model.n_models == 1:
            switch = (e % 4 == 0)
        else:
            switch = env_model.determine_switching(e)

        if e == starting_episode or switch:
            # save path of current CFD episode
            env_model.append_cfd_obs(e)

            # update n_fills
            if e != starting_episode:
                buffer._n_fills = e

            env_model.start_timer()
            buffer.fill()
            env_model.time_cfd_episode()
            states, actions, rewards = buffer.observations

            # set the correct trajectory length
            env_model.len_traj = actions[0].size()[0]

            # in 1st episode: CFD data is used to train environment models for 1st time
            env_model.start_timer()
            if e == starting_episode:
                cl_p_models, cd_models, l, obs = wrapper_train_env_model_ensemble(training_path, env_model.obs_cfd,
                                                                                  env_model.len_traj, env.n_states,
                                                                                  buffer_size, env_model.n_models,
                                                                                  n_time_steps=env_model.t_input,
                                                                                  env=executer)

            # ever CFD episode: models are loaded and re-trained based on CFD data of the current & last CFD episode
            else:
                cl_p_models, cd_models, l, obs = wrapper_train_env_model_ensemble(training_path, env_model.obs_cfd,
                                                                                  env_model.len_traj, env.n_states,
                                                                                  buffer_size, env_model.n_models,
                                                                                  load=True, env=executer,
                                                                                  n_time_steps=env_model.t_input)
            env_model.time_model_training()

            # save train- and validation losses of the environment models
            env_model.save_losses(e, l)

            # reset buffer, policy loss and set the current episode as last CFD episode
            buffer.reset()
            env_model.reset(e)

        # fill buffer with trajectories generated by the environment models
        else:
            # generate trajectories from initial states using policy from previous episode, fill model buffer with them
            env_model.start_timer()
            predicted_traj, current_policy_loss = fill_buffer_from_models(cl_p_models, cd_models, e, training_path,
                                                                          observation=obs, n_probes=env.n_states,
                                                                          n_input=env_model.t_input,
                                                                          len_traj=env_model.len_traj,
                                                                          buffer_size=buffer_size, agent=agent)
            env_model.time_mb_episode()
            env_model.policy_loss.append(current_policy_loss)

            # if len(predicted_traj) < buffer size -> discard trajectories from models and go back to CFD
            if len(predicted_traj) < buffer_size:
                buffer._n_fills = e
                env_model.start_timer()
                buffer.fill()
                env_model.time_cfd_episode()
                states, actions, rewards = buffer.observations
                env_model.append_cfd_obs(e)

                # re-train environment models to avoid failed trajectories in the next episode
                env_model.start_timer()
                cl_p_models, cd_models, l, obs = wrapper_train_env_model_ensemble(training_path, env_model.obs_cfd,
                                                                                  env_model.len_traj, env.n_states,
                                                                                  buffer_size, env_model.n_models,
                                                                                  n_time_steps=env_model.t_input,
                                                                                  load=True)
                env_model.time_model_training()
            else:
                # save the model-generated trajectories
                env_model.save(e, predicted_traj)

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
        env_model.start_timer()
        agent.update(states, actions, rewards)
        env_model.time_ppo_update()
        agent.save_state(join(training_path, f"checkpoint.pt"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))
        buffer.reset()
    env_model.print_info()
    print(f"Training time (s): {time() - env_model.start_training}")


class RunTrainingInDebugger:
    """
    class for providing arguments when running script in IDE (e.g. for debugging). The ~/.bashrc is not executed when
    not running the training from terminal, therefore the environment variables need to be set manually in the Allrun
    scripts
    """

    def __init__(self, episodes: int = 2, runners: int = 2, buffer: int = 2, finish: float = 5.0,
                 n_input_time_steps: int = 30, seed: int = 0, timeout: int = 1e15, out_dir: str = "examples/TEST"):
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
        self.checkpoint = False

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

        # test MB-DRL on local machine
        d_args = RunTrainingInDebugger(episodes=10, runners=4, buffer=4, finish=5, n_input_time_steps=30, seed=0)
        assert d_args.finish > 4, "finish time needs to be > 4s, (the first 4sec are uncontrolled)"

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
