# Force

## About
Force is a library for deep reinforcement learning (RL) research, built on [PyTorch](https://pytorch.org/) and [Gymnasium](https://gymnasium.farama.org//).
It is under active development.
Features at present:
 * Readable, modular implementations of various deep RL algorithms
 * Composable configuration management system that exposes all hyperparameters to be specified by files and command-line arguments
 * A browser-based GUI for viewing experiment info, including some basic filtering and plotting
 * Launch jobs on the [Slurm](https://slurm.schedmd.com/overview.html) scheduler, with easy parallelization over hyperparameters

The name *Force* was originally derived from the word *reinforcement*, but, in a fun coincidence, it is also related to the name of [my PhD advisor](https://ai.stanford.edu/~tengyuma/) via [Newton's second law of motion](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton's_second_law).

## Installation
Clone the repository, run the following commands from the root directory of the repository:
```
pip install -r requirements.txt
pip install -e .
```

## Usage
An example of how to use the library can be found in `scripts/train.py`.
For anything more complicated, you can easily define your own logic by subclassing `Experiment`.

The script can be called like so:
```
python scripts/train.py --root-dir ROOT_DIR --domain DOMAIN --config CONFIG_PATH
```
where the all-caps variables are substituted appropriately:
 * `ROOT_DIR` should specify a directory where experiment logs will be written.
(Each experiment run will create a subdirectory therein.)
 * `DOMAIN` refers to the task being solved.
 * `CONFIG_PATH` is a path to a JSON file specifying the configuration.

Multiple config files can be used by repeating the `--config` (abbr. `-c`) flag.

To override specific entries in the config, use the `--set` (abbr. `-s`) flag, which takes two arguments, the key and the value.
The key name may contain dots to denote nesting, for example `-s agent.init_alpha 0.5` if using a SAC agent.

You can optionally set a specific random seed by passing `--seed SEED`.
Otherwise, a seed will be randomly chosen.

## Viewing experiments
The GUI uses a client-server architecture because the experiment logs typically live on a remote machine.
To launch the server, simply point it to the directory:
```
python force/workflow/result_server.py -d DIRECTORY
```

You can optionally set a specific port using the `--port` (`-p`) flag.

From the client side, you can view various info about each experiment and access the config and log files.
URL query strings can be used to specify a quantity to plot (e.g. `?plot=eval/return_mean`) and to filter by domain, algorithm, recency, or job status.