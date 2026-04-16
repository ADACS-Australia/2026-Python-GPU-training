---
title: "CuPy: A Numpy-like GPU experience"
---

::: questions

- What is a GPU?

:::

::: objectives

- Understand a warp why multiples of 32 (or 64) threads is important
- Understand the hierarchy of GPU memory

:::

## CuPy as a drop-in replacement for Numpy

CuPy is a library that provides a multidimensional array type very similar to Numpy's `ndarray`.

## CPU arrays versus GPU arrays

Numpy's and CuPy's arrays are both multidimensional arrays, and they both expose similar APIs.

They differ in _where_ they reside. The CPU (host) and the GPU (device) each have their own independent memory, which is something we will cover in much more detail later. Objects stored in host memory can't be "seen" by the GPU without first transferring the data across to the device, and vice versa.

Numpy arrays exist in host memory and CuPy arrays exist on the GPU. To perform operations involving multiple arrays you must ensure that each array is within the same portion of memory: if all the arrays are on the host, the computation will be performed by the CPU; if they are all on the device, the computation will be peformed by the GPU.

It's up to you to handle transferring arrays back and forth between the host and device.

In `Numpy` we can create a numerical array from a range of iterables:

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5]). # from a list
arr2 = np.array(range(1, 1000, step=2))  # from a range
arr3 = np.array([x**2 for x in range(1, 100)])  # from a list comprehension
```

These arrays all reside in host (CPU) memory.

To create a CuPy GPU array we can do something very similar and merely need to swap out `np` for `cupy`:

```python
import cupy
import numpy as np

arr1_d = cupy.array([1, 2, 3, 4, 5])  # from a list
arr2_d = cupy.array(range(1, 1000, step=2))  # from a range
arr3_d = cupy.array([x**2 for x in range(1, 100)])  # from a list comprehension

# Or transfer across an existing numpy array:
arr4 = np.random.normal(shape=1_000_000)
arr4_d = cupy.array(arr4)  # this is a copy!

# And to return back to the host:
arr3 = cupy.asnumpy(arr3_d)
```

Note the convention of appending `_d` to the names of arrays that are on the GPU **d**evice. This is purely convention but, in the absence of types, this can help you keep track of where an array is located.

(There's also `cupy.asarray(otherarray)`: this will create a GPU array _only if_ `otherarray` isn't already on the GPU device. It's handy function to ensure an array is on the GPU but will avoid a copy if it isn't needed.)


## An aside: on benchmarking

### Python's `timeit`

In Python, the easiest way to benchmark some code is to use the built-in module `timeit`.

For example:

```python
import timeit

import numpy as np

a = np.random.normal(size=100_000_000)
b = np.random.normal(size=100_000_000)

def my_computation(a, b):
    a += b

timer = timeit.Timer(lambda: my_computation(a, b))

# Call `do_some_work()` just once, and repeat this 10 times
elapsed_times = timer.repeat(repeat=10, number=1)
print(min(elapsed_times))
```

The first argument to `Timer()` must be a callable that takes no arguments; we use a lambda function to capture and pass in the arguments. The method `repeat()` will return a set of times for each iteration. In most cases, you want to take the _minimum_ value from these repeats.

Now we can try running this same code on the GPU using CuPy:

```python
import timeit

import cupy

a = cupy.random.normal(size=100_000_000)
b = cupy.random.normal(size=100_000_000)

def my_computation(a, b):
    a += b

timer = timeit.Timer(lambda: my_computation(a, b))

# Call `do_some_work()` just once, and repeat this 10 times
elapsed_times = timer.repeat(repeat=10, number=1)
print(min(elapsed_times))
```

Notice all we did to run this on the GPU was to swap the `np` namespace calls to `cupy`.

The time looks great. On my machine, the Numpy computation takes 0.44 ms whilst the CuPy takes 0.005 ms.

But there's a problem with this benchmarking. The CuPy code is being executed asynchronously: we are sending the computation to the GPU and then immediately continuing without waiting for the result. Since the CPU and the GPU are separate devices this is a great way to keep sending work across to the GPU, but it hides the true time of the computation.

One solution here is to force the code to behave synchronously. And we can do that by manually inserting a synchronisation call at the tail of our computation function:

```python
def my_computation(a, b):
    a += b

    # Wait here until the GPU is finished
    cupy.cuda.get_current_stream().synchronize()
```

Now if we measure the timings, the CuPy version on my machine takes 1.8 ms. That's still a great time but its 3 orders of magnitude slower than we first reported!

### Using `cupyx.benchmark()`

Another way to avoid these issues is to use the benchmarking function provided by CuPy:

```python
import cupy
import cupyx

a = cupy.random.normal(size=100_000_000)
b = cupy.random.normal(size=100_000_000)

def my_computation(a, b):
    a += b

results = cupyx.profiler.benchmark(lambda: my_computation(a, b), n_repeat=10)
print(results)
```

This function inserts event timers onto the GPU device and is able to measure two different times: the CPU time and the GPU time. If you run this, you'll notice that the CPU time is much less than the GPU time, since the asynchronous computation returned control to the CPU almost immediately, whilst the GPU continued to execute in the background.

For me, benchmarking shows 0.008 ms recorded by the CPU, and 1.7 ms on the GPU — both very similar to what we measured earlier before and after we added the explicit synchronisation.

::: exercise

Modify the previous benchmark in the following way:

* Create the vectors `a` and `b` as numpy arrays first.
* Transfer these to GPU arrays _inside_ the `my_computation` function.

How does this affect the benchmark? Why?

:::

::: solution

```python
import cupy
import cupyx
import numpy as np

a = np.random.normal(size=100_000_000)
b = np.random.normal(size=100_000_000)

def my_computation(a, b):
    a_d = cupy.array(a)
    b_d = cupy.array(b)
    a_d += b_d

results = cupyx.profiler.benchmark(lambda: my_computation(a, b), n_repeat=10)
print(results)
```

The benchmark is significantly longer in this version due to the time taken for memory to be copied from host to device.

This is an important consideration to make in all your work with GPUs: even if the computation itself is faster, you must be careful that setup costs like memory transfers don't eclipse you're savings.

:::


## Broadcasting

Let's use our new powers to speed up some computations. Almost all of the standard numpy functions are available to you, including the [broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) you will be familiar with.

Let's take a look at a couple of examples.

### Discrete Fourier transform

First we consider the [discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform). Recall that this is defined as:

$X_k = \sum_n^N x_n  e^{-2 i \pi \frac{k n}{N}}$

We can write this as a broadcasting operation across 2 dimensions (k by n), followed by a sum along the n'th dimension:

```python
import cupy
import cupyx
import numpy as np

def DFT(xs):
    # This function returns either np or cupy module depending on array type
    # which lets us write device-agnostic code.
    xp = cupy.get_array_module(xs)

    N = len(xs)
    ks = xp.arange(0, N)
    ns = xp.arange(0, N)

    phases = ks[:, None] * ns[None, :] / N
    return xp.sum(
        xs[None, :] * np.exp(-2j * np.pi * phases),
        axis=1
    )

# Create a compex valued input where both real and imaginary components
# are normally distributed.
xs = np.random.normal(size=1000) + 1j * np.random.normal(size=1000)
xs_d = cupy.array(xs)

# Sanity check! Does our version agree with numpy's own library?
np.testing.assert_allclose(DFT(xs), np.fft.fft(xs))

# CPU benchmark
result = cupyx.profiler.benchmark(lambda: DFT(xs), n_repeat=10)
print(result)

# GPU benchmark
result = cupyx.profiler.benchmark(lambda: DFT(xs_d), n_repeat=100)
print(result)
```

This is a function with elementwise multiplication, complex eponentiation, scalar multiplication and summation. When our inputs are GPU arrays it happens entirely on the GPU. Run this on your own machine and take note of the runtime comparison between the CPU and GPU versions.

Also take note of the function `cupy.get_array_module(arr)`: this is a useful helper function to write device-agnostic code.

### Matrix multuplication

Try benchmarking on your machine the following where we implement matrix multiplication as a pair of broadcasting and sum operations:

```python
import cupy
import cupyx
import numpy as np

def matmul(A, B):
    # Matrix multiplication using broadcasting
    return (A[:, :, None] * B[None, :, :]).sum(axis=1)

A = np.random.normal(size=(100, 1000))
B = np.random.normal(size=(1000, 100))
A_d = cupy.array(A)
B_d = cupy.array(B)

# Sanity check
cupy.testing.assert_allclose(matmul(A, B), A @ B)

# CPU benchmark
result = cupyx.profiler.benchmark(lambda: matmul(A, B), n_repeat=10)
print(result)

# GPU benchmark
result = cupyx.profiler.benchmark(lambda: matmul(A_d, B_d), n_repeat=10)
print(result)
```
On my machine, the GPU result comes out clearly ahead by a factor of over 60! (Bonus points: Can you spot a danger in using this broadcasting algorithm for matrix multiplication? Hint: What happens if the matrices get larger?)

### Fusing operations

Just like numpy, a series of array operations are applied to CuPy arrays sequentially. This means that each operator (e.g. an addition, a scaler multiplication, perhaps a trigonometric function) is applied as a separate kernel, which must in turn read and write through the entirety of the arrays each time. This kernel dispatch overhead and the associated memory churn can be a considerable performance penalty.

CuPy offers the ability to fuse _simple_ operators into a single operation by using a simple function decorator. In practice this means the operators are applied as a single kernel.

Try running the following on your own machine and see how the speeds compare:

```python
import cupy
import cupyx

xs = cupy.random.normal(size=100_000)
ys = cupy.random.normal(size=100_000)

def sequential(xs, ys):
    return (2 * xs + 3) * cupy.sin(ys)**4

@cupy.fuse
def fused(xs, ys):
    return (2 * xs + 3) * cupy.sin(ys)**4

result = cupyx.profiler.benchmark(lambda: sequential(xs, ys), n_repeat=100)
print(result)

result = cupyx.profiler.benchmark(lambda: fused(xs, ys), n_repeat=100)
print(result)
```

On my machine I see an approximately threefold speed improvement but, as always, benchmark and profile your code.

## Linear Algebra

CUDA includes an extensive linear algebra library that is highly optimised, and CuPy's `linalg` routines provide an accessible interface to this library. If you can rewrite your problem succinctly as a series of linear algebra operations this will almost always be faster than a custom kernel.

Consider the example of a large matrix multiplication on both CPU and GPU:

```python
import numpy as np
import cupy
import cupyx

A = np.random.normal(size=(10_000, 10_000))
B = np.random.normal(size=(10_000, 10_000))

A_d = cupy.array(A)
B_d = cupy.array(B)

print(
    cupyx.profiler.benchmark(lambda: A @ B, n_repeat=10)
)

print(
    cupyx.profiler.benchmark(lambda: A_d @ B_d, n_repeat=10)
)
```

Note that here we've used the matrix multiplication operator, `@`, which is shorthand for either `np.linalg.matmul` or `cupy.linalg.matmul` depending on the array type.

The speed-up is huge: on my own hardware, we observe 16.6 s versus just 107 ms.

Similarly, we can perform matrix inversion or decomposition just as we would using numpy:

```python
import cupy

# Let's solve for x in: Ax = y
A = cupy.random.normal(size=(1000, 1000))
y = cupy.random.normal(size=1000)

Ainv = cupy.linalg.inv(A)
x0 = Ainv @ y

# solve() uses a decomposition algorithm that is more
# numerically stable than the inverse matrix
x1 = cupy.linalg.solve(A, y)

# Check that both methods return that same solution
cupy.testing.assert_allclose(x0, x1)

# Perform QR decomposition
Q, R = cupy.linalg.qr(A)
```

There's also the `einsum()` method which is not a linear algebra method but which is very powerful if you can write your equations using [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation), e.g.:

```python
import cupy

A = cupy.random.normal(size=(1000, 10_000))
B = cupy.random.normal(size=(10_000, 1000))

# Einstein notation for matrix multiplication
C = cupy.einsum("ij, jk -> ik", A, B)

cupy.testing.assert_allclose(C, A @ B)
```

::: challenge

Compute the [outer product](https://en.wikipedia.org/wiki/Outer_product) of two large vectors, `x` and `y`, using both CuPy built-in `cupy.outer` routine and using `einsum()`. Check that the different methods give the same result.

```python
import cupy

x = cupy.random.normal(shape=10_000)
y = cupy.random.normal(shape=10_000)

# To do...
```

:::

::: solution

```python
import cupy

x = cupy.random.normal(size=10_000)
y = cupy.random.normal(size=10_000)

A0 = cupy.outer(x, y)

A1 = cupy.einsum("i, j -> ij", x, y)

cupy.testing.assert_allclose(A0, A1)
```

:::

## FFT

The Fourier transform is a staple of signal processing, and the _fast_ Fourier transform (FFT) algorithm is already a massive improvement on the naive sum.

CUDA provides highly optimised libraries for performing FFTs and CuPy provides a high level wrapper that mirrors the numpy routines. In addition, it also exposes some lower-level routines that might be essential to getting good performance.

In numpy, for example, we can do the following FFT:

```python
import numpy as np

# Create a complex valued 4 x 1024 x 1024 array where real
# and imaginary components are both normally distributed
a = np.random.normal(size=(4, 1024, 1024)) + 1j * np.random.normal(size=(4, 1024, 1024))

# Perform a 2D fft for each of the 4 1024 x 1024 matrices
A = np.fft.fftn(a, axes=(1, 2))
```

Performing this on the GPU involves the same steps as before: ensure the arrays reside in GPU memory and replace numpy with CuPY prefixed methods:

```python
import cupy

# Create a complex valued 4 x 1024 x 1024 array where real
# and imaginary components are both normally distributed
a = cupy.random.normal(size=(4, 1024, 1024)) + 1j * cupy.random.normal(size=(4, 1024, 1024))

# Perform a 2D fft for each of the 4 1024 x 1024 matrices
A = cupy.fft.fftn(a, axes=(1, 2))
```

Try benchmarking the results: is it faster?

When you run a FFT the CUDA library actually does two things:

* It creates a plan: depending on your input data, the axes you care about, etc. it needs to work out how best to do this. This will almost always involve reserving some memory to use as a working space.
* It then executes a plan.

If you're executing the same FFT multiple times you might want to save on the overhead of creating the plan. CuPy lets you do this:

```python
import cupy
import cupyx

# Create a complex valued 4 x 1024 x 1024 array where real
# and imaginary components are both normally distributed
a = cupy.random.normal(size=(4, 1024, 1024)) + 1j * cupy.random.normal(size=(4, 1024, 1024))

# Create a plan
plan = cupyx.scipy.fft.get_fft_plan(a, axes=(1, 2))

# Plan acts as a context manager
with plan:
    A = cupy.fft.fftn(a, axes=(1, 2))
```

(Although in my testing, this doesn't provide a speed-up, and possibly isn't working as expected in the current versions.)
