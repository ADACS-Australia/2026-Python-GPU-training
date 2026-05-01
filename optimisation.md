---
title: Optimisation
---

::: questions

- What is a GPU?

:::

::: objectives

- Understand a warp why multiples of 32 (or 64) threads is important
- Understand the hierarchy of GPU memory

:::

## A toy problem

We're going to start with a toy problem: imaging radio interferometry data. In other domains, this is a also known as a non-uniform Fourier transform: we transform from a 3D visibility space to a 2D surface within the imaging space.

An important non-aim: we are not looking to make _algorithmic_ improvements. Real-world imaging software rarely performs a full _direct_ Fourier transform since it is so costly, but those alternative algorithm are more complex and are not our focus here.

The discrete imaging equation is defined as follows:

$$
V(l, m) = \sum V(u, v, w) e^{2 \pi i  (u l + v m + w [n - 1])} \\
\textrm{where} \quad n = \sqrt{1 - l^2 - m^2}
$$

The dimensions $l$ and $m$ span the image domain, whilst the $u, v, w$ coordinates span the (sparse and unevenly sampled) visibility domain. We are prohibited from treating this as a simple 2D fast Fourier transform problem due to two main factors: u, v, w are not even sampled; and the so-called $w$ term is non-neglible.

Image sizes are typically thousands of pixels by thousands of pixels; whilst visibility data is similarly millions of rows in length.

By the end, we hope to be able to image the large radio lobes of Fornax A from the raw visibility data.

![](episodes/fig/fornaxA.png)

## A first implementation

As always, we will approach this first on the CPU where we can ensure we first make things right before attemping a first pass at a kernel.

Let's first set things up. Our data is stored in the file `visibilities.npz` and we will extract each datum and its associated baseline coordinates:

```python
data = np.load("visibilities.npz")
us, vs, ws, vis = data["u"], data["v"], data["w"], data["data"]
```

Next we will set up our imaging grid. We will image a 700 x 700 pixel grid. We will set the scale such that $\Delta l = \Delta m = 10^{-4}$ (this scale is somewhat arbitrary but is chosen to the main radio feature within the image).

```python
lpx, mpx = np.mgrid[-350:350, -350:350]
ls, ms = lpx * 0.0005, mpx * 0.0005
```

Finally, we will precompute the associated values $n' = n - 1 = \sqrt{1 - l^2 - m^2} - 1$:

```python
ndashes = np.sqrt(1 - ls**2 - ms**2) - 1
```

::: challenge

Using numba's `njit` and `prange` functions, attempt to write an CPU-parallel implementation of the direction imaging equation. Ask yourself: what is the unit of parallelisation?

Using matplotlib's `imshow()`, save a figure of the real components of the image.

Note: below we reduce the data by 50x to speed things up. If things take too long for you, try reducing the data even more aggressively.

```python
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def image_numba(us, vs, ws, data, ls, ms, ndashes, img):
    pass
    # TODO

data = np.load("visibilities.npz")
us, vs, ws, data = data["u"], data["v"], data["w"], data["data"]

lpx, mpx = np.mgrid[-350:350, -350:350]
ls, ms = lpx * 0.0005, mpx * 0.0005
ndashes = np.sqrt(1 - ls**2 - ms**2) - 1

# For now, reduce data by 50x since CPU is too slow
us, vs, ws, data = us[::50], vs[::50], ws[::50], data[::50]

img = np.zeros(ls.shape, dtype=complex)
image(us, vs, ws, data, ls, ms, ndashes, img)

plt.imshow(img.real, origin="lower")
plt.savefig("output.png")
```

:::

::: solution

We parallelise over each pixel which, for a 700 x 700 pixel image, gives us just shy of 500,000 threads. And since each pixel is its own independent sum, we don't have to worry about race conditions.

```python
@njit(parallel=True)
def image_numba(us, vs, ws, data, ls, ms, ndashes, img):
    # Parallelise over each output pixel
    for lmpx in prange(ls.size):
        # Convert 1D to 2D pixel index
        lpx, mpx = lmpx // ls.shape[1], lmpx % ls.shape[1]

        # Retrieve l, m and ndash coordinates associated with this pixel
        l, m, ndash = ls[lpx, mpx], ms[lpx, mpx], ndashes[lpx, mpx]

        # Perform the for this one pixel over all of the visibility data
        for u, v, w, datum in zip(us, vs, ws, data):
            phase = 2 * np.pi * (u * l + v * m + w * ndash)
            img[lpx, mpx] += datum * np.exp(1j * phase)
```

Even with only a fifthieth of the data we start to see something resembling the radio lobes:

![](episodes/fig/fornaxA-fiftieth.png)

:::

::: challenge

Now that we have a working CPU version, the next step is to extract the inner loop as a standalone kernel and configure an associated grid to work on the GPU.

Try implementing a simple GPU kernel implementation using the `@numba.cuda` decorator.

**Note:** As with the DFT function we have worked on earlier, you will need to rewrite the imaginary exponential using the identity $e^{i \phi} = \cos{\phi} + i \sin{\phi}$.

:::

::: solution

The implementation is very similar to that used in Numba's prange loop. Some points of note:

* We have chosen here to retain a linear index for simplicity. Additionally, in the case where the image dimensions don't match an even multiple of the threads per block, the 1D grid minimises the amount of threads that "overflow".
* We have rewritten the imaginary exponential as a combination of `cos()` and `sin()`.

```python
@cuda.jit
def kernel1(us, vs, ws, data, ls, ms, ndashes, img):
    lmpx = cuda.grid(1)

    if lmpx < img.size:
        lpx, mpx = divmod(lmpx, ls.shape[1])

        # Retrieve l, m and ndash coordinates associated with this pixel
        l, m, ndash = ls[lpx, mpx], ms[lpx, mpx], ndashes[lpx, mpx]

        # Perform the for this one pixel over all of the visibility data
        for u, v, w, datum in zip(us, vs, ws, data):
            phase = 2 * np.pi * (u * l + v * m + w * ndash)
            img[lpx, mpx] += datum * complex(math.cos(phase), math.sin(phase))
```

On my machine, this is fast enough to allow us to image the full set of data giving a lovely image of Fornax A:

![](episodes/fig/fornaxA-fulldata.png)

:::

## Profiling with NSIGHT Compute

Up till now we have used simple timers to measure performance. NVIDIA provides us with an alternative, extremely detailed set of benchmarking tools:

* **NSIGHT Compute:** Provides detailed analysis of a kernel, including its register usage, FMA instruction counts, memory accesses, and so on. This is most useful when optimising a specific kernel.
* **NSIGHT Systems:** A higher-level profiler that can show a timeline of memory transfers, kernel runtimes, streams, and the interleaved dependencies between these different GPU processes. This is most useful to understand the full lifecycle of a program and to help prioritise optimisation work.

In this section, we will focus our work on NSIGHT Compute since we are discussing kernel optimisation. However, in a real-world project always begin with NSIGHT Compute: don't start optimising any specific component until you understand both its contribution to the overall runtime and can estimate the overall effect of its improvement.

NSIGHT Compute is run as a two part process:

1. First, the the kernels of interest are run on the machine wrapped by the `ncu` profiler and save the output data to a file. This can be performed either directly on the command line or via the NSIGHT Compute GUI.
2. You run NSIGHT Compute locally, import the profile file, and use the GUI to explore different metrics of your kernel.

The profiler `ncu` should be included as part of the NVIDIA toolkit installation. However, you will need to NSIGHT Compute locally, which you can [download here.](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)

To run the profiler directly on the command line, enter something like the following:

```
/usr/local/cuda/bin/ncu -o profile -f --kernel-name 'regex:^kernel6' --set detailed -- uv run python timeit-demo.py
```

Here, we output (`-o`) to a file called `profile` (`-f` overwrites if it already exists), we profile the kernel(s) whose names match the regex `^kernel6` (kernel names will always start with the python function nane), and we collect the metrics that are part of the `detailed` set.

Other metric sets are available and can be listed with `ncu --list-sets`. Another helpful set is `roofline`.

The profiling can take some time: it will run our kernel with the full data multiple times, collecting a subset of the metrics each time. The larger the number of metrics you measure, the slower the whole process will take.

Once complete, we load the `profile.mcu-rep` file into the NSIGHT Compute GUI, either by using its remote file functionality or by first transferring the profile data locally.

[ TODO: ]

Things to look for:

* Am I using float64 values by mistake?
* Am I compute bound, memory bound, or neither?
* What is the ratio of FMA to total floating point instructions?

## Optimisation 1: Minimise writing to global memory

We've already touched on this optimisation strategy before, but it's important enough to repeat here: reading and writing to _global_ memory is expensive and currently our kernel currently writes to global memory on every single inner loop. Always prefer local or shared memory over global memory.

::: challenge

Rewrite the kernel to use a local accumulator variable inside the innermost loop and write out to global memory just once.

:::

::: solution

```python
@cuda.jit
def kernel2(us, vs, ws, data, ls, ms, ndashes, img):
    lmpx = cuda.grid(1)

    if lmpx < img.size:
        lpx, mpx = divmod(lmpx, ls.shape[1])

        # Retrieve l, m and ndash coordinates associated with this pixel
        l, m, ndash = ls[lpx, mpx], ms[lpx, mpx], ndashes[lpx, mpx]

        # Perform sum over all of the visibility data for just one pixel
        pixel = complex(0)
        for u, v, w, datum in zip(us, vs, ws, data):
            phase = 2 * np.pi * (u * l + v * m + w * ndash)
            pixel += datum * complex(math.cos(phase), math.sin(phase))

        img[lpx, mpx] = pixel
```

:::

## Optimisation 2: Floating point precision

You may be aware that floats come in different sizes, which correlate to different precisions. In C, the standard options are `float` and `double`, which correspond to 32 and 64 bit precision. CUDA provides a range of lower precision types such as a 16 bit float (and even some non-standard numerical types that make different trade offs between mantissa and exponent).

Unfortunately, Python hides the types of values and you may not be used to thinking about the precision of your calculations.

On the CPU, floating point performance usualy doesn't depend on the precision. That is, the performance ratio of 32 and 64 bit floats is 1:1 (with the important previso that 32 bit floats benefit from their smaller memory footprint). On the GPU, however, the performance ratio can vary enormously, and differs from one GPU to the next. For example, some AMD GPUs have a 1:1 performance ratio which is great for high precision numerical calculations; NVIDIA GPUs, on the other hand, strongly prefer lower precision floats, and this imbalance has increased considerably as their GPUs have been optimised towards AI workloads. Their recent B200 GPU, for example has a performance ratio of 1:2:32 for 64/32/16 bit computations.

**The upshot is: always compute at the minimal precision required.**

To do this for our kernels, there are a few things we need to do:

* **Initialise all arrays by explicitly** passing the optional `dtype` keyword argument and setting to `np.float64`, `np.float32`, `np.float16` (or their complex equivalents, `np.complex128` and `np.complex64`).
* **Cast existing arrays using the `astype()` method** e.g. `arr_f32 = arr_f64.astype(np.float32)`.
* **Initialize (or cast) all intermediate variables or constants with explicit types.** For example:

   * Create an accumator variable with a typed constructor: `acc = np.complex32(0)`
   * Cast Pi to the appropriate precision _before_ use in computation: `np.float32(np.pi)`

* **Use typed mathematical functions** from [`cuda.libdevice`](https://nvidia.github.io/numba-cuda/reference/libdevice.html) to compute at the desired precision (e.g. using `cuda.libdevice.sinf()` rather than `math.sin()`).
* **Understand the rules for type promotion.** Types will be quietly promoted to higher precision when combined with other higher precision types. For example:

   * If you use an integer as part of a calculation with a float, the integer will be first cast to the float type.
   * If you combine floats of mixed precision, the lower precision floats will be first cast to the higher precision.

   As a result, you must be careful of high precision types sneaking in and poisoning your computation to a higher precision than necessary.

::: challenge

For our usecase, we deem 32 bit precision to be acceptable. Modify the input arrays and kernel to use 32 bit floats and 64 bit complex numbers (i.e. 32 bits for each of the real and imaginary components).

Some helpful hints:

* The functions `math.cos()` and `math.sin()` promote inputs to 64 bit floats and output the result as 64 bit floats. To force 32 bit computation, you can use `numba.cuda.libdevice.cosf()` and `numba.cuda.libdevice.sinf()`.
* The functions `np.complex64` and `np.complex128` return an error when passed two arguments inside a kernel. Use `np.complex64(complex(real, imag))` for now. (Open bug report: [https://github.com/NVIDIA/numba-cuda/issues/865](https://github.com/NVIDIA/numba-cuda/issues/865).)

:::

::: solution

```python
@cuda.jit
def kernel3(us, vs, ws, data, ls, ms, ndashes, img):
    lmpx = cuda.grid(1)

    if lmpx < img.size:
        lpx, mpx = divmod(lmpx, ls.shape[1])

        # Retrieve l, m and ndash coordinates associated with this pixel
        l, m, ndash = ls[lpx, mpx], ms[lpx, mpx], ndashes[lpx, mpx]

        # Perform sum over all of the visibility data for just one pixel
        pixel = np.complex64(0)
        for u, v, w, datum in zip(us, vs, ws, data):
            phase = 2 * np.float32(np.pi) * (u * l + v * m + w * ndash)
            pixel += datum * np.complex64(
                complex(cuda.libdevice.cosf(phase), cuda.libdevice.sinf(phase))
            )

        img[lpx, mpx] = pixel
```

:::

## Use specialised mathematical functions

CUDA provides a [wide range of mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html) at different precisions. These functions are heavily optimised and will use hardware implementations where possible. Numba exposes a subset of these which are documented [here](https://nvidia.github.io/numba-cuda/reference/libdevice.html) and [here](https://nvidia.github.io/numba-cuda/reference/kernel.html#intrinsic-attributes-and-functions).

In fact, we've already made use of these specialised function by using the `sinf` and `cosf` functions.

Depending on your use case, you might use these functions to:

* Control the precision of the function
* Take advantage of specialised functions that might be more efficient for your use case
* Use _fathmath_ functions

`fastmath` functions are worth commenting on slightly further. Fastmath functions take fewer instructions to complete and are therefore faster, but in exchange they trade both precision and correctness. The NVIDIA programming guide, for example, specifies that `fast_sinf()` has an increased error tolerance of $2^{-21.41}$ in the range $-\pi$ to $\pi$, and that this error is larger elsewhere. Fastmath operations also typically break floating point guarantees around associativity and usually assume strictly finite values (NaN and Inf values may be handled incorrectly).

::: challenge

Scan through the list of available mathematical functions that are part of [libdevice](https://nvidia.github.io/numba-cuda/reference/libdevice.html). Rewrite the calls to `sinf` and `cosf` with more efficient operations.

:::

::: solution

There are a number of candiates which stand out as options here:

* `sinpi()` and `cospi()` compute the phase by multiplying it by $\pi$. That's one less multiplication operation that can have a small effect on the speed of our kernel.
* `fast_sin()` and `fast_cos()` are two among a number of similar `fast_` prefixed functions that use their respective fastmath implmentation.

Looking further however, we also spot a family of functions that _co-compute_ both the `sin` and `cos` components of the phase, sharing the result of intermediate computations for both. They are:

* `sincosf()`
* `sincospif()`
* `fast_sincosf()`

After some experimentation, `fast_sincosf()` produces the fastest results and retains the accuracy needed for our use case:

```python
@cuda.jit
def kernel4(us, vs, ws, data, ls, ms, ndashes, img):
    lmpx = cuda.grid(1)

    if lmpx < img.size:
        lpx, mpx = divmod(lmpx, ls.shape[1])

        # Retrieve l, m and ndash coordinates associated with this pixel
        l, m, ndash = ls[lpx, mpx], ms[lpx, mpx], ndashes[lpx, mpx]

        # Perform sum over all of the visibility data for just one pixel
        pixel = np.complex64(0)
        for u, v, w, datum in zip(us, vs, ws, data):
            phase = 2 * np.float32(np.pi) * (u * l + v * m + w * ndash)
            sin, cos = cuda.libdevice.sincosf(phase)
            pixel += datum * np.complex64(complex(cos, sin))

        img[lpx, mpx] = pixel
```

:::

## Minimise reading from global memory

We've previously used a local accumulation variable to minimise _writes_ to global memory. In this section, however, we're going to use shared memory to minimise _reads_ from global memory.

**Our problem is that each kernel needs to read through the entirety of each of the `us`, `vs`, `ws` and `data` arrays.** There's a lot of redundancy here: since each thread is fully independent, they are each reading the same data, from start to end. But what if we could somehow share the burden of these global memory reads amongst the threads?

A standard pattern in kernel design is for each thread in a threadblock to cooperate by using shared memory as an intermediate cache. Remember that shared memory is much faster than global memory, and is visible from each thread within a thread block. In doing so, global reads reduce by a factor equal to threads per block. For a threadblock with 256 threads, this will reduce global memory reads by a factor of 256.

Suppose there are 256 threads per block. Then the pattern proceeds as follows:

1. The threadblock allocates a shared memory array with size 256. Each thread can read and write to this array, and can see the writes of each sibling thread within the threadblock.
2. The threadblock reads in the first 256 index values of the global memory, with each thread reading a _single, unique_ value based on its thread ID. For example, thread 45 within the thread block reads the 45th value of the global array.
3. Each thread caches its retrieved datum into a unique location in shared memory, using its thread ID as the array index. For example, thread 45 writes to the 45th index of the shared array.
4. Finally, the threads cycle through each value in shared memory and performs its associated computation.

The cycle is then repeated: the _next_ 256 values from global memory are written into shared memory, and so on, until the global memory is exhausted.

In pseduo-code:

```python
# Create shared memory array
xs_shared = cuda.shared.array(256, dtype=np.float32)

tid = cuda.threadIdx.x. # threadId _within_ the threadblock
N = len(xs_global)

# Step through the array in blocks of 256
for offset in range(0, N, 256):
    # Write all 256 values from global memory into
    if offset + tid < N:
        # Each thread in the threadblock is responsible for reading a unique global memory address
        # and writing to a unique shared memory address based on its thread ID.
        xs_shared[tid] = xs_global[offset + tid]
    else:
        # Unless N is a multiple of 256, you will need to handle the remainder
        xs_shared[tid] = 0

    # Ensure the cache is fully populated before any individual thread tries to read from it
    cuda.syncthreads()

    for x in xs_shared:
        # Compute...
        pass

    # Wait for each thread to finish computation before the cache is repopulated
    cuda.syncthreads()
```

Some important comments:

* We create shared memory by calling `cuda.shared.array(...)` in each thread. But this syntax is deceptive: it _looks_ like each thread creates its own shared array, but this allocation is actually just performed just once at the threadblock level. Each thread shares the same shared memory array.
* The `cuda.threadIdx.x` is the thread ID _within_ the thread block, ranging in this case from 0 to 255. This is not the global thread ID which we get from `cuda.grid(1)`.
* The thread ID gurantees that each thread reads from and writes to a unique address in global and shared memory.
* Since threads within are threadblock are not guarateed to operate in lockstep (only at the warp level is this true), we need to call the synchronisation function `cuda.synchthreads()`. This function acts a gate that stops threads within a given threadblock from progressing until all threads have reached this point. This guarantees two things: that the cache is fully populated before we attempt to read from it; and later, that the cache is not updated until all threads have completed reading from it.
* We have to handle the case where `N` is not a multiple of 256. In the example above, on the final batch we pad the shared array with zeros on the assumption that zero is idempotent under our kernel. Other options for handling this remainder exist, such as:

   ```
   for i in range(256, min(N - offset)):
       x = xs_shared[i]
       # Compute...
    ```

::: note

### Coalesced memory reads

The GPU memory system performs well when a threadblock coordinates to read from a _contiguous_ region of memory. In particular, if each thread in a threadblock reads the _next_ entry in an array, the GPU can turn all these little reads into a single, unified read instruction. When this happens, it is called _memory coalescing._

On the other hand, if each thread reads from somewhere seemingly random, or if each thread's memory read is strided (i.e. there are gaps between each read), then the GPU can't combine these into a single, unified read instruction. Those hundreds of reads will be queued and processed one by one. Obviously, this is not good for memory performance.

In the above example, we read from memory using the indexing `offset + tid` which guaratees _memory coalescing_.

:::

::: challenge

Modify the kernel to mimimise global memory reads by caching the global values from `us`, `vs`, `ws`, and `data`.

Start by creating shared memory arrays at the top of the kernel:

```python
NTHREADS = 256
uvw_cache = cuda.shared.array((NTHREADS, 3), dtype=np.float32)
data_cache = cuda.shared.array(NTHREADS, dtype=np.complex64)
```
Then proceed to complete the remaining TODOs.

**Beware:** We must ensure that each thread in a threadblock participates in cache generation. For example, the condition `if lmpx < img.size` must no longer result in some threads prematurely terminating. Instead, these "remainder" threads must still continue to play their role in populating and refreshing their associated index in the caches (to be used by the other threads in its threadblock) even though they ultimately do not write out to global memory.

```python
```python
@cuda.jit
def kernel6(us, vs, ws, data, ls, ms, ndashes, img):
    # TODO:
    # Add the shared memory caches here

    lmpx = cuda.grid(1)

    # TODO:
    # 1. Initialize l, m, ndash
    # 2. Conditionally set l, m, ndash if lmpx < img.size
    # 3. Don't return early!

    # Initialise pixel accumulator variable
    pixel = np.complex64(0)

    # Extract input data in batches of NTHREADS items
    N = data.size
    for offset in range(0, N, NTHREADS):
        # TODO:
        # 1. Fetch data and populate the caches
        # 2. Handle the remainder by setting cache entries to 0.

        # Wait for cache to be populated
        cuda.syncthreads()

        # Iterate over cache
        for (u, v, w), datum in zip(uvw_cache, data_cache):
            phase = 2 * np.float32(np.pi) * (u * l + v * m + w * ndash)
            sin, cos = cuda.libdevice.fast_sincosf(phase)
            pixel += datum * np.complex64(complex(cos, sin))

        # Don't start updating cache until all threads are done
        cuda.syncthreads()

    if lmpx < img.size:
        img[lpx, mpx] = pixel
```

:::

::: solution

```python
@cuda.jit
def kernel6(us, vs, ws, data, ls, ms, ndashes, img):
    NTHREADS = 256
    uvw_cache = cuda.shared.array((NTHREADS, 3), dtype=np.float32)
    data_cache = cuda.shared.array(NTHREADS, dtype=np.complex64)

    lmpx = cuda.grid(1)

    l, m, ndash = np.float32(0), np.float32(0), np.float32(0)
    if lmpx < img.size:
        lpx, mpx = divmod(lmpx, ls.shape[1])

        # Retrieve l, m and ndash coordinates associated with this pixel
        l, m, ndash = ls[lpx, mpx], ms[lpx, mpx], ndashes[lpx, mpx]

    # Initialise pixel accumulator variable
    pixel = np.complex64(0)

    # Extract input data in batches of NTHREADS items
    N = data.size
    for offset in range(0, N, NTHREADS):
        # Fetch data and populate cache
        i = offset + cuda.threadIdx.x
        if i < N:
            uvw_cache[cuda.threadIdx.x] = us[i], vs[i], ws[i]
            data_cache[cuda.threadIdx.x] = data[i]
        else:
            uvw_cache[cuda.threadIdx.x] = 0, 0, 0
            data_cache[cuda.threadIdx.x] = 0

        # Wait for cache to be populated
        cuda.syncthreads()

        # Iterate over cache
        for (u, v, w), datum in zip(uvw_cache, data_cache):
            phase = 2 * np.float32(np.pi) * (u * l + v * m + w * ndash)
            sin, cos = cuda.libdevice.fast_sincosf(phase)
            pixel += datum * np.complex64(complex(cos, sin))

        # Don't start updating cache until all threads are done
        cuda.syncthreads()

    if lmpx < img.size:
        img[lpx, mpx] = pixel
```

:::

## Thread coarsening

## The compiler blackbox: experiment with the hot loop

## Force FMA instructions

In my experience, and especially in the context of complex numbers, it is worth experimenting with explicit FMA instructions. In my opinion, this is a failing of the CUDA compiler.