"""
implements a Dataloader class for the MB-DRL training, which is responsible for loading and filtering the CFD data
"""
import logging
import torch as pt

from typing import Tuple

from ...constants import DEFAULT_TENSOR_TYPE

logger = logging.getLogger(__name__)
pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class TrajectoryLoader:
    def __init__(self, n_probes: int, n_actions: int, buffer_size: int, cfd_episode_no: int, trajectory_length: int,
                 load_n_episodes: int = 2):
        """
        loads, filters and merges the data from CFD

        :param n_probes: number of probes placed in the flow field
        :param n_actions: number of actions
        :param buffer_size: buffer size
        :param cfd_episode_no: episodes executed in CFD
        :param trajectory_length: length of the trajectory
        """
        self._n_probes = n_probes
        self._n_actions = n_actions
        self._buffer_size = buffer_size
        self._cfd_episode_no = cfd_episode_no
        self._load_n_episodes = load_n_episodes
        self._trajectory_length = trajectory_length
        self._data = {}
        self._files = None
        self.keys = ["states", "rewards", "actions", "alpha", "beta", "cx", "cy"]

    def load(self, file_names: list) -> dict:
        """
        executes the loading, filtering, merging and scaling of the trajectories from CFD

        :param file_names: paths to the observation files executed in CFD
        :return:data from CFD scaled to an interval of [0, 1] along with the corresponding min.- and max.-values
        """
        self._files = file_names
        self._load_trajectories()
        self._check_trajectories()
        self._merge_trajectories()
        return self._data

    def _load_trajectories(self, files: list = None) -> None:
        """
        load the trajectory data from the observations_*.pt files

        :return: None
        """
        # load the trajectories of the specified episodes
        self._files = files if files is not None else self._files[-self._load_n_episodes:]
        observations = [pt.load(open(file, "rb")) for file in self._files]

        # merge all trajectories, for training the models, it doesn't matter from which episodes the data is
        shape, n_col = (self._trajectory_length, len(observations) * len(observations[0]), 1), 0

        for key in observations[0][0].keys():
            # same data structure for states, actions, rewards, cx, cy, alpha & beta
            if key == "states":
                self._data[key] = pt.zeros((shape[0], shape[1], self._n_probes))
            elif key == "actions":
                self._data[key] = pt.zeros((shape[0], shape[1], self._n_actions))
            else:
                self._data[key] = pt.zeros(shape)

        # loop over all CFD episodes
        for o, obs in enumerate(observations):
            # loop over all trajectories within the buffer of each CFD episode
            for t, tr in enumerate(obs):
                # omit failed or partly converged trajectories
                if not bool(tr) or tr["rewards"].size(0) < self._trajectory_length:
                    logger.warning(f"Trajectory no. {t} in episode {o} didn't converge.")
                    continue
                else:
                    for key in self._data:
                        # make sure the trajectory is as long as specified in the setup
                        if key != "states" and (key != "actions" or self._n_actions == 1):
                            self._data[key][:, n_col, :] = tr[key][:self._trajectory_length].unsqueeze(-1)
                        else:
                            self._data[key][:, n_col, :] = tr[key][:self._trajectory_length, :]
                n_col += 1

    def _check_trajectories(self) -> None:
        """
        check for diverged or invalid trajectories

        :return: None
        """
        # delete the allocated columns of the failed trajectories
        self._mask = self._data["states"].abs().sum(dim=0).sum(dim=1).bool()

        # if the buffer of the current CFD episode is empty, load trajectories from N+1 CFD episodes ago
        if self._mask.sum() < self._buffer_size and self._cfd_episode_no >= self._load_n_episodes+1:
            self._load_trajectories([self._files[-self._load_n_episodes-1:-1]])

            # and finally, update the masks to remove non-converged trajectories
            self._mask = self._data["states"].abs().sum(dim=0).sum(dim=1).bool()

        # if we don't have any trajectories generated within the last 3 CFD episodes, it doesn't make sense to
        # continue with the training
        if self._data["rewards"][:, self._mask].size(1) == 0:
            logger.critical(f"Could not find any valid trajectories within the last {self._load_n_episodes + 1} CFD "
                            f"episodes!\nAborting training.")
            exit(0)

    def _merge_trajectories(self) -> None:
        """
        merges all trajectories to single tensor, in case multiple actuation is used

        :return: None
        """
        for key in self.keys:
            # merge values for cx_* or cy_* into single tensor
            if key != "states" and key not in self._data.keys():
                self._data[key] = pt.cat([self._data[k] for k in self._data.keys() if k.startswith(f"{key}_")], dim=-1)

        # scale to an interval of [0, 1] (except alpha and beta), use list() to avoid runtimeError when deleting the
        # keys
        for key in list(self._data.keys()):
            # delete all keys that are no longer required
            if key.endswith("_a") or key.endswith("_b") or key.endswith("_c"):
                self._data.pop(key)
                continue

            # scale to an interval of [0, 1]
            elif key != "alpha" and key != "beta":
                self._data[key], self._data[f"min_max_{key}"] = self._scale(self._data[key][:, self._mask, :])

            # alpha and beta don't need to be re-scaled since they are always in [0, 1]
            else:
                self._data[key] = self._data[key][:, self._mask]

    def reset(self) -> None:
        """
        reset the data and files properties

        :return: None
        """
        self._data = {}
        self._files = None

    @property
    def data(self) -> dict:
        """
        get the loaded CFD data

        :return: loaded CFD data as dict
        """
        return self._data

    @staticmethod
    def _scale(x: pt.Tensor) -> Tuple[pt.Tensor, list]:
        """
        scale data to the interval [0, 1] using a min-max-normalization

        :param x: data to scale
        :return: tensor with normalized data and corresponding (global) min- and max-values used for normalization
        """
        # x_i_normalized = (x_i - x_min) / (x_max - x_min)
        return (x - x.min()) / (x.max() - x.min()), [x.min(), x.max()]


if __name__ == "__main__":
    pass
