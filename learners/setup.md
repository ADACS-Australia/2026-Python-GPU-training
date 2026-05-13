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

You will need a computer with an NVIDIA GPU and [CUDA toolkit](https://developer.nvidia.com/cuda/toolkit) installed. (CuPy _does_ have some support AMD GPUs, but `numba-cuda` does not.)

We recommend installing the requisite packages using [`uv`](https://docs.astral.sh/uv/) (although `pip` should work too):

```shell
$ uv init gpu-tutorial

$ cd gpu-tutorial

$ uv add numpy \
       numba \
       numba-cuda \
       cupy \
       matplotlib
```

We will frequently omit imports from the code snippets listed throughout this tutorial. The following imports should be assumed:

```python
# Python imports
import math

# Package imports
import cupy
import matplotlib.pyplot as plt
from numba import cuda, njit, prange
import numpy as np
```