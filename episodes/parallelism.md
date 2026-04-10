---
title: "An introduction to parallelism"
---

Serial programming is by far the simplest programming model and is the easiest to reason about.

Serial programming should always be preferred unless performance needs necessitate otherwise.

## Serial programming

## Task Parallelism

Task parallelism divides up the work into different types of jobs. Each task functions autonomously and relies on different communication methods between tasks to coordinate the work. Tasks in this model form a complex graph structure, with dependencies between nodes.

**Example:** If we consider the bike example, perhaps one person assembles frames, another constructs tyres and a third puts it all together. This is a bit like how a factory may be organised and is called _pipelining_.

**Example:** If the bikes were made to order. Perhaps each person builds each bike in its entirety, customised towards the order specification. Depending on the type of customisation, builds may take varying amounts of time. This is called a _worker pool_.

Task parallelism requires careful coordination between tasks:

- In pipelining, tasks may operate a different speeds. This means:
   - Data must be passed between tasks in a way that clearly construes ownership: for example, there's no point the assembler trying to install partially constructed tyres.
   - Buffers or queues must be implemented between tasks to absorb the different rates of work
   - If upstream queues are empty or downstream queues are full, tasks must be able to pause
- In worker pools, a system must be designed for optimally allocating and distributing work, as well as collecting the results in the end.

It is also common for pipelining and worker pools to be combined.

The coordination aspect of this model has proven in practice to be a common source of bugs. In response, a host of techniques have been developed to help mitigate, including:

- Synchronisation primitivies: semaphores, mutexes, locks, and conditions
- Atomic values
- Channels and pipes
- Software transactional memory
- Message passing interfaces (e.g. OpenMPI)

## Data Parallelism

Data parallelism applies a computation to data by dividing up the data into small, independent chunks of work. Each thread, however, performs identical work.

Data parallelism is implmented in hardware on the CPU as "single instruction, multiple data" (SIMD) and on the GPU as "single instruction, multiple thread" (SIMT). The "single instruction" is key: the program instruction (perhaps an addition, or a multiplication) is broadcast across multiple pieces of data.

In both cases, the programming model is quite restrictive:

* The computation (or "kernel") must be identical for each data input
* Communication between threads is very limited and, for the most part, each kernel must proceed independently
* Computation across each thread proceeds in lockstep: conditions or branches that result in divergence can be costly

**Example:** To return to the bike example, data parallelism would take as inputs all the bike parts, and each worker would assemble a bike in identical fashion at an identical pace. There would be no possibility for customisation.

GPUs are built around data parallelism and gain many of their speed advantages from the associated restrictive programming model.

## The pitfalls of parallelism

For all the potential speed benefits that we gain from parallelism, it is important to be aware of a number of important downsides to parallelism.

* Performance overhead
* Race conditions
* More complex code

When considering moving code from serial to parallel, weigh up these costs to ensure

### Performance overhead

Parallel code doesn't come for free. In fact, sometimes throwing _more_ threads at a problem can make things slower.

The types of overhead differ between CPU and GPU. When using CPU-backed threads, some of the costs include:

- **Thread spawning**: which involves creating a thread and allocating it some initial memory
- **Context switching:** when there are more threads than cores, the host OS will periodically suspend threads to "fairly" let each thread advance its work
- **Communication overhead:** all higher level communication methods are built on a toolbox of atomics, locks, semphores and condition variables and these have a non-neglible overhead

On the GPU the costs are different mostly owe to the fact that a CPU (the "host") and the GPU (the "device") are physically distinct. Some of these costs include:

- **Memory transfers:** GPUs have their own memory that is separate from the host and it takes time to transfer back and forth
- **Command latency:** telling the GPU what to do and transferring kernels has some associated latency
- **Synchronisation and communication:** a GPU has its own synchronisation primitives and a limited ability to communicate, both of which have a cost

When designing GPU code you must ensure that the computational benefits dwarf the associated costs of memory transfers and ensure your algorithm uses synchronisation sparingly.

In addition, your algorithm may also be slowed down by **resource contention.** This occurs when different threads all attempt to access a single resource of some kind, perhaps a file on disk, a network resource, or even to access memory. Slow downs due to resource contention can be non-linear.

### Race conditions

With multiple threads operating at the same time, parallelism makes it difficult to _order_ access to anything that might be shared: reading or writing to memory; printing to a terminal; accessing files; and so on. This is because threads make few guarantees about when they execute, and do not guarantee uninterrupted execution either.

In short, parallelism upsets the order of operations. And anything that relies on order must take explicit steps to enforce a desired ordering.

As an example, consider this parallel shift, where we want to move each item in an array down one element (and wrap around):

```python
xs = [1, 2, 3, 4, 5]
for i in prange(len(xs)):  # prange executes each iteration of the loop in parallel
    x = xs[i]
    xs[i - 1] = x  # shift down
```

This is example of a race condition: the code implicitly expects all threads to have read from `xs` before any thread then writes to it. The problem is that there is no guarantee that these operations will occur in lockstep across all the threads. One thread, for example, might write to `xs` before other threads have even begun, or vice versa.

Or consider a data parallel sum:

```python
xs = [1, 2, 4, 6, 2, ...]

x_sum = 0
for i in prange(len(xs)):
    x_sum += xs[i]
```

This is also subject to a data race. Even though the operation `x_sum += xs[i]` _looks_ like a single instruction, it is in fact equivalent to `x_sum = x_sum + xs[i]`. In this form you can see that the variable `x_sum` is first read, the addition is computed, and result is written back. Once again, the race is on to ensure that all threads read first before any writes occur, and without some kind of synchronisation this is not guaranteed.

In later sections, we will discuss the kinds of available synchronisation techniques available to you.

### Code complexity

Serial code is very easy to reason about: first this happens, then this, and finally that. Under the hood, the compiler may perform [all sorts of optimisations](https://en.wikipedia.org/wiki/Optimizing_compiler) (e.g. rearranging the order of instructions or pre-emptively executing a conditional branch) but it does this whilst guaranteeing that these are invisible within the thread.

Parallel code tends to be much more complex:

- Each thread is doing only part of the overall computation
- You can no longer guarantee the ordering of operations: threads may start, suspened, and stop at different times
- Compiler optimisations become visible between threads
- Significant programmer work is required to reason about and handle synchronisation

These complexities mean that bugs are more likely, and you must weigh this against the performance benefits you expect.

We recommend to always write your algorithm in a single threaded, serial form first, and use this to test your parallel code later.


::: challenge

Run the following code: what do you expect and what do you get?

Can you rewrite the code without a race condition? Hint: use `numba.get_num_threads()` and `numba.get_thread_id()`.

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def psum(xs):
    # We use an array of len=1 to avoid numba automatically
    # fixing our race condition
    x_sum = np.zeros(1)
    for i in prange(len(xs)):
        x_sum[0] += xs[i]

    return x_sum[0]

xs = np.ones(1_000_000)
print("Sum:", psum(xs))
```

::: solution

```python
import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id

@njit(parallel=True)
def psum(xs):
    # Give each thread a unique array index
    x_sum = np.zeros(get_num_threads())
    for i in prange(len(xs)):
        x_sum[get_thread_id()] += xs[i]

    # Finally sum the individual thread sums in the main thread
    return x_sum.sum()

xs = np.ones(1_000_000)
print("Sum:", psum(xs))
```

:::