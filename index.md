---
site: sandpaper::sandpaper_site
---

## Introduction

This is an advanced course in GPU programming using Python. You will learn about the types of problems that are suitable for parallelisation on the GPU and how to reformulate a problem using data parallelism. We will show you how to use the high-level GPU API provided by CuPy and then how to write your own, highly optimised kernels.

This course can be separated into two days:

* **Day 1:**

   * GPU fundamentals
   * High-level GPU programming with CuPy
   * Writing your first kernel

* **Day 2**

   * Kernel optimisation, including:
      * Floating point precision
      * Shared memory optimisations
      * Thread coarsening
      * Explicit FMA instructions

### Why use GPU acceleration?

It is increasingly common for science to make use of massive datasets—such as high-resolution sky surveys, complex N-body simulations, or signal processing over terabytes of data. Processing these vast datasets can be computationally prohibitive on standard CPUs. However, it is our good fortune that many of these tasks are trivially parallelisable, meaning that the work can be progressed by many processing cores at once.

GPU acceleration is the next step beyond CPU parallelisation, and allows you to leverage literally thousands of parallel cores to accelerate these tasks. GPUs are built specifically with high multi-threaded workloads in mind, and are efficient at these kinds of tasks in a way that CPUs are simply not. GPU programming is not easy, but with a little care (and experimentation) it is possible to achieve speedups of several orders of magnitude in processing time.

Or, let Mythbusters convince you:

<video width="100%" controls>
   <source src="https://github.com/ADACS-Australia/2026-Python-GPU-training/releases/download/mythbusters/gpu-mythbusters.webm" type="video/webm">
</video>

Source: [Nvidia on Youtube](https://web.archive.org/web/20241001024753/https://www.youtube.com/watch?v=-P28LKWTzrI)

### Why Python?

This course is taught using Python and a number of libraries that allow writing CUDA kernels directly using Python. Python has been chosen to make this course as accessible as possible, which allows us to focus on the underlying concepts without getting snagged on unfamiliar syntax. Under the hood, the Python kernels are [compiled](https://en.wikipedia.org/wiki/Just-in-time_compilation) to CUDA kernels and run just as fast as if we had written them in C.

And rest assured: the concepts learned here can be applied wholesale to CUDA programming in C or C++ (or Julia, Rust, ...).

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

## AI Declaration

With the exception of the following, no AI or LLM tools were used to prepare these notes in any capacity (including planning, writing, or otherwise):

* The shared memory optimisation animation (using Gemini 3).
* Code proofing and final feedback (Claude 4.6)