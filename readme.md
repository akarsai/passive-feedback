# Passive feedback control for nonlinear systems

This repository contains the code to the paper

[T. Breiten and A. Karsai, Passive feedback control for nonlinear systems, arXiv preprint 2502.04987, (2025).](https://arxiv.org/abs/2502.04987)

## Reproducing our results
The first step is to install Python with version `>=3.12.5`.
We recommend using a virtual environment for this.
Using [pyenv](https://github.com/pyenv/pyenv), the steps are as follows:

```bash
# this assumes pyenv is available in the environment
pyenv install --list | grep " 3\.[19]" # get all available versions, only starting with 3.1x or 3.9x
pyenv install 3.12.5 # choose 3.12.5 for example
pyenv virtualenv 3.12.5 phcon # creates environment 'phcon' with version 3.12.5
```

The next step is to clone this repository, install the necessary requirements located in `requirements.txt`, and set the `PYTHONPATH` variable accordingly.
```bash
cd ~ # switch to home directory
git clone https://github.com/akarsai/passive-feedback.git
cd passive-feedback
pip install --upgrade pip # update pip
pip install -r requirements.txt # install requirements
export PYTHONPATH="${PYTHONPATH}:~/passive-feedback" # add folder to pythonpath
```

Now, we can run the scripts `tests/test_controller_performance.py` and `tests/test_discrete_gradient.py` to reproduce the figures in the paper.
The generated plots will be put in the directory `results` as `.pgf` and `.png` files.
```bash
# both of these commands should take < 5 min
python tests/test_controller_performance.py
python tests/test_discrete_gradient.py
```



## Some hints
- Throughout the codebase, all functions depending on time are vectorized in time. This means that, e.g. `eta(z)` must be well-defined for arrays `z` with the shape `z.shape == (number_of_timepoints, space_dimension)`. The time index is always at position `0`.
- Since the implementation uses the algorithmic differentiation capabilities of JAX, the implementations of all functions need to be written in a JAX-compatible fashion. The provided examples should be a good starting point.
- In case of questions, feel free to reach out.