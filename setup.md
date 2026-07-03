---
title: Setup
---

## Expected background

This course sets a reasonably high bar for what you are expected to already know, including:

* Python
* Primitive data types: integers, floats, doubles, etc. and their machine representation
* numpy, including [broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
* Some experience with `numba` and its [parallelisation functions](https://numba.pydata.org/numba-doc/dev/user/parallel.html)

Additionally, the examples given here are aimed towards people coming from a physics, astronomy, or engineering background. It will help if you are familiar with:

* Basic mathematical sum notation, e.g. for [discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
* Complex numbers (and, for example, [Euler's formula](https://en.wikipedia.org/wiki/Euler's_formula))

## Setup

It is expected that you will be working on ozstar for this workshop, but if you have a CUDA based GPU you can work on your local machine (not recommended).

To work on ozstar you will need an account (sign up here), and you will need to have joined the group `oz983`.
Making an account and joining the group takes time as there is a human approval process required for each.

### Installing the environment on a local machine

You will need a computer with an NVIDIA GPU and [CUDA toolkit](https://developer.nvidia.com/cuda/toolkit) installed. (CuPy _does_ have some support AMD GPUs, but `numba-cuda` does not.)

We recommend installing the requisite packages using [`uv`](https://docs.astral.sh/uv/) (although `pip` should work too):

```shell
$ uv init gpu-tutorial

$ cd gpu-tutorial

$ uv add numpy \
       numba \
       numba-cuda \
       cupy \
       matplotlib \
       pytest

$ source .venv/bin/activate
```

### Installing the environment on HPC

Installing the environment on a HPC like [Ngarrgu Tindebeek](https://supercomputing.swin.edu.au/ngarrgu-tindebeek) will require loading the `gcc` and `CUDA` modules and installing [`uv`](https://docs.astral.sh/uv/). 

Installing `uv` should require no change to the Linux installation instructions as the bash script that runs should install `uv` to your home directory on the login node of the HPC.
Using `uv` to install the environment on Ngarrgu Tindebeek:

```shell
$ module load gcc/13.3.0 cuda/12.8.0
# Note the uv will manage the python version you're using, no need to load the python module
$ uv init gpu-tutorial

$ cd gpu-tutorial

$ uv add numpy \
       numba \
       numba-cuda \
       cupy \
       matplotlib \
       pytest

$ source .venv/bin/activate
```

### Running Python on a HPC

Running the code in this tutorial, will require you to interact with SLURM on Ngarrgu Tindebeek, you can do this via `sinteractive` or via `sbatch`. Starting an interactive session with Ngarrgu Tindebeek will require you to run this command (or some variation of it):

```shell
# You may want to make the time longer or shorter depending on what you need
$ sinteractive --job-name="gpu-tutorial" --partition=milan-gpu --gres=gpu:1 --mem 16G --cpus-per-task=8 --time=04:00:00
```

Once you're in the interactive shell, you'll need to reload the modules and the environment before running any code:

```shell
$ module load gcc/13.3.0 cuda/12.8.0
$ source .venv/bin/activate

# From here you can run code as you need
$ python some_python_file_here.py
# Or start an interactive python session
$ python
```

Alternatively, you can also use `sbatch` to do things in a less interactive way. The same interactive command but via an `sbatch` script

```bash
#!/bin/bash 
#SBATCH --job-name=gpu-tutorial 
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G 
#SBATCH --time=00:15:00 

module load gcc/13.3.0 cuda/12.8.0
source .venv/bin/activate

python some_python_file_here.py
```

Then run the script on the cluster via `sbatch`:

```shell
$ sbatch run_python_script.sh
```

### Using Jupyter Notebooks on a HPC for the Tutorial

If you wish to use jupyter notebooks with Ngarrgu Tindebeek, you can do so by first adding `jupyter` to the uv environment via:

```shell
$ uv add jupyter
```

Then either inside an interactive or running job, run jupyter notebook via this command (it's easier to do this inside an `sbatch` replacing the python command above with the jupyter command below):

```shell
# It's important to include the --no-browser and --ip 0.0.0.0 arguments
# The ip being 0.0.0.0 is a broadcast address, which will allow you to access the notebook kernel from the login node via the name of the node.
$ jupyter notebook --no-browser --ip 0.0.0.0
```

If this is inside an sbatch, a slurm output file would have been produced (`slurm-<job_id>.output`), which will contain the URL for the running jupyter notebook. It should look something like this:

```
http://<node-name>:<port>/tree?token=somelongtokenhere
```

From there, you have two main options to interact with the running notebook.

If you are using an IDE like VSCode or PyCharm (which have support for running jupyter notebooks) connected to the login node via SSH, then follow the instructions for connecting to existing jupyter notebooks to connect to the running notebook, the instructions for adding existing/external servers can be found in the links for [VSCode](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management#_existing-jupyter-server) and [PyCharm](https://www.jetbrains.com/help/pycharm/configuring-jupyter-notebook.html#setup-configured-server). **For the external URL, it's critical you use the URL that contains the name of the node, not `localhost`**. 

If you wish to use jupyter notebook in the browser, then you will need to reconnect to the login node with an altered SSH command to forward the notebook ports from the login node to your local machine, using the `<port>` and the `<node-name>` from the url above:

```shell
$ ssh user@nt.swin.edu.au -L <port>:<node-name>:<port>
```

Once the connection is established use that url found inside the `slurm-<job-id>.output` above but replace `<node-name>` with `localhost` on a browser inside your browser on your local machine.

::: information
As the [Swinburne OzStar documentation](https://supercomputing.swin.edu.au/docs/2-ozstar/Notebooks.html#working-with-jupyter-notebooks), running notebooks should generally be used for learning, development, testing or simple analysis purposes. Computationally intensive work should be submitted as a batch job to the cluster, rather than a notebook running on the cluster. In general when working with a HPC, it's best to create python scripts and run them via the `sbatch`.
:::

### Using pip instead

The above instructions will work using `pip` and standard `venv` instead, replace the `uv` commands with the corresponding `pip` commands. Using `pip` will also mean you need to load the `python` module on a HPC environment, for `Ngarrgu Tindebeek` utilise the `python/3.13.1` module. In general setting up the environment will look like this:

```shell
$ module load gcc/14.2.0 cuda/12.8.0 python/3.13.1
$ mkdir gpu-tutorial
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install numpy \
       numba \
       numba-cuda \
       cupy \
       matplotlib \
       pytest
```

The pip install for `cupy` will take some time to complete as it will rebuild the wheel on the login node, if you compare `pip` to `uv` in this case you will _really_ notice just how _fast_ `uv` is.

### Imports

We will frequently omit imports from the code snippets listed throughout this tutorial. The following imports should be assumed:

```python
# Python imports
import math

# Package imports
import cupy
import cupyx
import matplotlib.pyplot as plt
from numba import cuda, njit, prange
import numpy as np
```