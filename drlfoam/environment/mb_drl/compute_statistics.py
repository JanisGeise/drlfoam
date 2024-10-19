"""
time everything and compute statistics
"""
from time import time
from torch import tensor, std, mean, max, min


class ComputeStatistics:
    def __init__(self):
        """
        implements a class to time the different parts of the training
        """
        self.start_training = None
        self._start_time = None
        self._time_cfd = []
        self._time_model_train = []
        self._time_prediction = []
        self._time_ppo = []

    def start_timer(self) -> None:
        self._start_time = time()

    def time_cfd_episode(self) -> None:
        self._time_cfd.append(time() - self._start_time)

    def time_model_training(self) -> None:
        self._time_model_train.append(time() - self._start_time)

    def time_mb_episode(self) -> None:
        self._time_prediction.append(time() - self._start_time)

    def time_ppo_update(self) -> None:
        self._time_ppo.append(time() - self._start_time)

    def compute_statistics(self, param) -> list:
        return [round(mean(tensor(param)).item(), 2), round(std(tensor(param)).item(), 2),
                round(min(tensor(param)).item(), 2), round(max(tensor(param)).item(), 2),
                round(sum(param) / (time() - self.start_training) * 100, 2)]

    def print_info(self) -> None:
        """
        display the results of the timings for the overall training routine

        :return: None
        """
        cfd = self.compute_statistics(self._time_cfd)
        model = self.compute_statistics(self._time_model_train)
        predict = self.compute_statistics(self._time_prediction)
        ppo = self.compute_statistics(self._time_ppo)

        # don't use logging here, because this is printed after the training is completed
        print(f"time per CFD episode:\n\tmean: {cfd[0]}s\n\tstd: {cfd[1]}s\n\tmin: {cfd[2]}s\n\tmax: {cfd[3]}s\n\t"
              f"= {cfd[4]} % of total training time")
        print(f"time per model training:\n\tmean: {model[0]}s\n\tstd: {model[1]}s\n\tmin: {model[2]}s\n\tmax:"
              f" {model[3]}s\n\t= {model[4]} % of total training time")
        print(f"time per MB-episode:\n\tmean: {predict[0]}s\n\tstd: {predict[1]}s\n\tmin: {predict[2]}s\n\tmax:"
              f" {predict[3]}s\n\t= {predict[4]} % of total training time")
        print(f"time per update of PPO-agent:\n\tmean: {ppo[0]}s\n\tstd: {ppo[1]}s\n\tmin: {ppo[2]}s\n\tmax:"
              f" {ppo[3]}s\n\t= {ppo[4]} % of total training time")
        print(f"other: {round(100 - cfd[4] - model[4] - predict[4] - ppo[4], 2)} % of total training time")


if __name__ == "__main__":
    pass
