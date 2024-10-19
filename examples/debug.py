"""
implements a class for executing the training inside an IDE, e.g., for debugging
"""
import sys

from glob import glob
from os.path import join
from shutil import rmtree
from os import system, environ, chdir


class DebugTraining:
    """
    Class for providing arguments when running script in the IDE (e.g., for debugging). The ~/.bashrc is not executed,
    when not running the training from the terminal, therefore the environment variables need to be set manually in the
    Allrun scripts
    """
    def __init__(self, episodes: int = 2, runners: int = 2, buffer: int = 2, finish: float = 5.0,
                 n_input_time_steps: int = 30, seed: int = 0, timeout: int = 1e15, out_dir: str = "examples/TEST",
                 simulation: str = "rotatingCylinder2D", n_models: int = 5,
                 drlfoam_path: str = "/media/janis/Daten/Promotion_TUD/Projects/drlfoam/"):
        self.command = ". /usr/lib/openfoam/openfoam2206/etc/bashrc"
        self.output = out_dir
        self.iter = episodes
        self.runners = runners
        self.buffer = buffer
        self.finish = finish
        self.environment = "local"
        self.debug = True
        self.n_input_time_steps = n_input_time_steps
        self.manualSeed = seed
        self.timeout = timeout
        self.checkpoint = ""
        self.simulation = simulation
        self.nModels = n_models

        # for debugging purposes, set environment variables for the current directory
        environ["DRL_BASE"] = drlfoam_path
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

    def set_openfoam_bashrc(self, training_path: str) -> None:
        # check if the path to bashrc was already added
        with open(f"{training_path}/Allrun.pre", "r") as f:
            check = [True for line in f.readlines() if line.startswith("# source bashrc")]

        # if not then add
        if not check:
            system(f"sed -i '5i # source bashrc for OpenFOAM \\n{self.command}' {training_path}/Allrun.pre")
            system(f"sed -i '4i # source bashrc for OpenFOAM \\n{self.command}' {training_path}/Allrun")

    def clean_directory(self) -> None:
        # clean up afterward
        [rmtree(dirs) for dirs in [d for d in glob(join(self.output, "copy_*"))]]

        try:
            rmtree(join(self.output, "env_model"))
        except FileNotFoundError:
            print("no directories for environment models found.")


if __name__ == "__main__":
    pass
