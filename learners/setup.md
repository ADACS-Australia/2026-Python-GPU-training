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

It is expected that you will be working on [OzStar] for this workshop, but if you have a CUDA based GPU you can work on your local machine (not recommended).

To work on ozstar you will need an [account](https://supercomputing.swin.edu.au/account-management/), and you will need to have joined the group `oz983`.

Making an account and joining the group takes time as there is a human approval process required for each, so please make sure that you do this before attending the workshop.


### Connecting to Ozstar

Log into ozstar using ssh via:

```bash
ssh <USERNAME>@ozstar.swin.edu.au
```

This should connect you to either `farnarkle1` or `farnarkle2` as these are the login nodes.

See the documentation [here](https://supercomputing.swin.edu.au/docs/1-getting_started/Access.html) for tips on how to setup easy access via ssh.




### Installing the environment on HPC

Installing the environment on a HPC like [OzStar] will require loading the `gcc` and `CUDA` modules and installing [`uv`](https://docs.astral.sh/uv/). 

Installing `uv` should require no change to the Linux installation instructions as the bash script that runs should install `uv` to your home directory on the login node of the HPC.
Using `uv` to install the environment on OzStar:

```bash
# Work in your home directory
cd ~
# Load the required software modules
module load gcc/13.3.0 cuda/12.8.0 python/3.12.3
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Now make a new directory and environment for us to work in
uv init gpu-tutorial
cd gpu-tutorial
# Note that the following may take up to 1/2 hour to complete (thanks cupy!)
uv add numpy \
       numba \
       numba-cuda \
       cupy \
       matplotlib \
       pytest
# Activate this new environment
source .venv/bin/activate
# Test that everything is working
python -c 'import numpy, numba, numba.cuda, cupy, pytest, matplotlib'
```

If you can run all of the above without generating errors then you are ready to go.
If you find errors then either email the organisers of the workshop, or be sure to arrive early on the first day to get in-person help.


### Running Python on a HPC

Running the code in this tutorial, will require you to interact with SLURM on OzStar, you can do this via `sinteractive` (recommended) or via `sbatch` (optional).

::: tab

### sinteractive

Start an interactive session with OzStar:

```bash
# You may want to make the time longer or shorter depending on what you need
sinteractive --gres=gpu:1 --mem 16G --cpus-per-task=8 --time=04:00:00 --reservation=curtin_training 
```
You will then see something like the following

```output
srun: job 14313053 queued and waiting for resources
srun: job 14313053 has been allocated resources
[phancock@john26 gpu-tutorial]$ 
```

There may be a delay between the "waiting" and "allocated" stage, though with the reservation in place this may only a be a few seconds.
You will know that your job has been allocated because your command prompt should now be `[<user>@<node> <working directory>]`, where `<node>` is **not** `farnarkle[12]`.

Once you're in the interactive shell, you'll need to reload the modules and the environment before running any code:

```bash
module load gcc/13.3.0 cuda/12.8.0 python/3.12.3
cd ~/gpu-tutorial
source .venv/bin/activate
```

### sbatch

The usual way to run scripts on an HPC is to submit a job script which contains all the information about the job you are running, and how it should run.

In a file called `run_python_script.sh` you would include:

```bash
#!/bin/bash 
#SBATCH --job-name=myjob 
#SBATCH --out=myjob-%j.out
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G 
#SBATCH --time=00:15:00 
#SBATCH --reservation=curtin_training

module load gcc/13.3.0 cuda/12.8.0 python/3.12.3
cd ~/gpu-tutorial
source .venv/bin/activate

python some_python_file_here.py
```

Then submit this script to the job scheduler using `sbatch`:

```shell
$ sbatch run_python_script.sh
```

When the job runs it will put any output (STDERR + STDOUT) into the file `myjob-<jobid>.out` in the current directory.
This file will include a short summary from the job scheduler about the resources used.

:::

### Editing your files on an HPC

There are many options available for the edit/run/fix loop that we will be engaged in during this workshop.
Depending on your prefered editor (and patience) we recommend the following:


::: tab

### VSCode

1. [Download](https://code.visualstudio.com/download?_exp_download=fb315fc982) and install VSCode on your local machine.
1. Install the extension [remote-ssh](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh), plus whatever other extensions you love.
1. Press the `><` looking button on the bottom left of your window.
1. Choose "Connect to host"
       1. If you set up your `.ssh/config` file you should see "ozstar" as an option
       1. Otherwise choose "Add new SSH host" and enter `ssh <username>@ozstar.swin.edu.au` and then your password if/when requested.
1. Choose File -> Open workspace from Folder"
1. Enter `/home/<username>>/gpu-tutorial/`
1. You should see the files on the remote system in your file explorer tab
1. Open the built in terminal (default is `<Ctrl>+~`)
1. This terminal is already logged into ozstar so you can use the `sinteractive` commands above to join an interactive session
1. From here, you can edit/save files in VSCode, and run them in the terminal without having to swap between them.

### vim/emacs/nano

The workflow here is:

1. ssh into ozstar
1. sinteractive to get a node with a gpu
1. use vim/emacs/nano to create/edit files
1. save file, quit editor
1. run file using `python myfile.py`
1. observe success or errors
1. if errors GOTO 3
1. success

This works just fine for short scripts but the continually open/edit/save/run loop can become tedious, especially when you are making typo level errors.



### Other

Whatever your prefered editor is.
We'll try to help if you have problems, but consider this a self supported option!

:::

## Imports

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


## AI Declaration

With the exception of the following, no AI or LLM tools were used to prepare these notes in any capacity (including planning, writing, or otherwise):

* The shared memory optimisation animation (using Gemini 3).
* Final proof and code checking (using Qwen 3.6)