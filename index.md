---
site: sandpaper::sandpaper_site
---

## Introduction

This is an advanced course in GPU programming using Python. You will learn about the types of problems that are suitable for parallelisation on the GPU and how to reformulate a problem using data parallelism. We will show you how to use the high-level GPU API provided by CuPy and then how to write your own, highly optimised kernels.

This course can be separated into two days:

::: tab

### **Day 1**

* [An intro to parallelism](parallelism.md)
* [GPU fundamentals](gpu-deepdive.md)
* [High-level GPU programming with CuPy](CuPy.md)
* [Writing your first kernel](first-kernels.md)

### **Day 2**

* [Kernel optimisation](optimisation.md), including:
   * Floating point precision
   * Shared memory optimisations
   * Thread coarsening
   * Explicit FMA instructions

:::

**Where**: The workshop will be held at Swinburne university in the [Engineering building](https://maps.app.goo.gl/vyjQiu5cBgsyp5T9A). 

**When**: The workshop will run on the 21st-22nd of July, 10am - 4pm each day. Arrive as early as 9am if you need help setting up your computer/environment.


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