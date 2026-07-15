---
title: 'Reference'
---

## Glossary

**Amdahl's Law:** A formula that quantifies the maximum speedup achievable by parallelising a program, given the fraction of the program that can be parallelised and the number of parallel execution units. It shows that the serial portion of a program places a hard ceiling on speedup, no matter how many cores are added.

**Asynchronous Execution:** When the CPU dispatches work to the GPU and immediately continues without waiting for the GPU to finish. This is the default behaviour for CuPy operations and means that naive benchmarks (e.g., using Python's `timeit`) can report misleadingly fast times. Use `cupyx.profiler.benchmark` or explicit synchronisation to measure true GPU runtime.

**Atomic Operation:** A limited set of operations (such as `+`, `max`, `min`) that are guaranteed to complete as a single, indivisible step even when multiple threads attempt them simultaneously. In CUDA, functions like `cuda.atomic.add()` prevent race conditions when threads compete to update the same memory location. They are correct but can be slow because they force serialisation.

**Bank Conflict:** A performance penalty that occurs when multiple threads in a warp access different addresses in the same memory bank of shared memory. Shared memory is divided into 32 banks on modern NVIDIA GPUs; if two threads in a warp request addresses that map to the same bank, the requests are serialised. The optimal access pattern ensures each thread in a warp reads from a unique bank (addresses differing by less than 32 bits).

**Binary Reduction:** A pattern for combining the elements of an array into a single value (such as a sum) by pairing adjacent elements, halving the number of elements at each step. Within a thread block, a binary reduction over shared memory cuts the work in half on each iteration: first thread `i` adds thread `i + 128`, then `i + 64`, and so on, until only thread 0 holds the block's total.

**Broadcasting:** Rules that control how arrays of different shapes are matched for element-wise operations. NumPy and CuPy automatically expand dimensions of length 1 so that arrays align. For example, adding a vector of shape `(1, N)` to a matrix of shape `(M, N)` adds the vector to every row. Broadcasting is powerful but can create large intermediate arrays that exceed GPU memory.

**Coalesced Memory Access:** A memory access pattern in which threads within a warp read from or write to contiguous (adjacent) addresses in global memory. When accesses are coalesced, the GPU combines them into a single, wide memory transaction. When accesses are random or strided, each thread's request is handled separately, dramatically reducing memory throughput. See also *Memory Coalescing*.

**Compile Time:** The moment when source code is translated into machine code, before the program runs. Values known at compile time (such as constants written literally in the code) allow the compiler to perform optimisations like loop unrolling. Values only known at *runtime* (such as function arguments) cannot be used for these optimisations.

**Compute-Bound:** A kernel is compute-bound when its runtime is limited by the rate at which the GPU can perform arithmetic operations, not by the rate at which it can move data. Compute-bound kernels benefit from faster arithmetic units, FMA instructions, and lower-precision floats. The opposite is *memory-bound*.

**Context Switching:** An overhead that occurs when there are more threads than available cores: the operating system periodically suspends threads to let each one advance its work. Each switch incurs a cost (saving and restoring thread state). On the GPU, this is handled far more efficiently through *latency hiding*, where the SM switches between warps at the hardware level with minimal overhead.

**CUDA:** Compute Unified Device Architecture — NVIDIA's proprietary parallel computing platform and programming model for general-purpose computation on GPUs. CUDA provides the API, compiler (nvcc), and runtime library that underlie all GPU kernels in this workshop. Python libraries like Numba and CuPy compile down to CUDA code.

**CUDA Core:** A single processing unit within a streaming multiprocessor (SM). CUDA cores are simpler and slower than CPU cores (typically running at a few hundred MHz) but exist in thousands on a modern GPU. They always execute in groups of 32 threads (a *warp*) in lockstep under the SIMT model.

**CUDA Stream:** A queue of operations executed in submission order on the GPU. By default, all operations are submitted to the *default stream* and run sequentially. Multiple streams allow operations to run concurrently — for example, overlapping a memory transfer on one stream with a computation on another. Within a stream, ordering is guaranteed; between streams, operations may be interleaved. Often referred to simply as a *stream*.

**`cuda.syncthreads()`:** A synchronisation barrier that pauses all threads in a thread block until every thread has reached the call. Essential when threads share data via *shared memory*: it ensures the entire block has finished writing to shared memory before any thread reads from it. Must not be placed inside a conditional branch, as threads that skip the call will cause a deadlock.

**CuPy:** A NumPy-compatible GPU array library. CuPy provides its own `ndarray` type that lives in GPU memory and mirrors almost all NumPy functions (math, linear algebra, FFTs, etc.) under the same names. CuPy is the recommended first approach for GPU acceleration before writing custom kernels.

**Data Parallelism:** A form of parallelism in which the *same* computation is applied to many independent pieces of data. Each thread performs identical work on different inputs (e.g., filtering every pixel in an image). GPUs are built around data parallelism. The alternative is *task parallelism*, where different threads perform different types of work.

**Data Race (Race Condition):** A bug that occurs when two or more threads access the same memory location concurrently and at least one is writing, with no synchronisation between them. The result is non-deterministic: it varies from run to run. An example is multiple threads executing `accumulator += value` simultaneously — the read, add, and write steps interleave unpredictably.

**Device:** The GPU, as seen from the CUDA programming model. The device has its own memory, its own compute units, and cannot directly access memory on the host (CPU). Data must be explicitly transferred between host and device.

**Fastmath:** Optimised mathematical functions (such as `fast_sinf()`, `fast_cosf()`) that use fewer instructions and are therefore faster than their standard counterparts. The trade-off is reduced precision and relaxed guarantees: for example, `fast_sinf()` has a larger error tolerance, may not preserve associativity, and may handle `NaN` or `Inf` values incorrectly.

**Fused Multiply-Add (FMA):** A single hardware instruction that computes `a * b + c` in one step. FMA instructions improve throughput (approaching 2× for the arithmetic portion of a kernel) and are the basis of advertised GPU floating-point performance numbers. In Numba CUDA kernels, use `cuda.fma(a, b, c)` to emit FMA instructions explicitly, as the compiler does not always insert them automatically.

**Global Memory:** The largest and slowest tier of GPU memory (tens to hundreds of gigabytes). All input and output data lives in global memory. It is visible to all threads on the GPU and persists for the lifetime of the allocation. Accessing global memory takes hundreds of cycles, so kernel optimisation focuses heavily on reducing global memory traffic.

**Grid:** The complete set of threads launched for a single kernel invocation. A grid is composed of one or more *thread blocks*. The grid dimensions (number of blocks and threads per block) are chosen by the programmer when launching a kernel, written in Numba as `kernel[nblocks, nthreads](...)`.

**Grid-Stride Loop:** A loop pattern inside a kernel that ensures every element of an array is processed regardless of the grid size. Instead of an `if` bounds check, the kernel uses a `for` loop starting at `cuda.grid(1)`, stepping by `cuda.gridsize(1)`, and ending at the array length. This works correctly whether the grid is smaller or larger than the data.

**GPGPU:** General-Purpose computation on Graphics Processing Units — the use of a GPU for computation beyond graphics rendering. CUDA (NVIDIA) and HIP/ROCm (AMD) are the dominant GPGPU platforms.

**High Bandwidth Memory (HBM):** A type of GPU memory technology that stacks memory chips vertically and connects them to the GPU die through an extremely wide bus. This delivers vastly higher bandwidth than conventional GDDR memory. Modern datacentre GPUs such as the NVIDIA B200 use HBM3e, providing 192 GB of global memory with 8 TB/s bandwidth.

**Host:** The CPU, as seen from the CUDA programming model. The host controls the device, allocates and frees GPU memory, launches kernels, and transfers data. Python code running CuPy or Numba executes on the host.

**Hot Loop:** The innermost loop of a kernel — the code executed most frequently. Small changes in the hot loop (rearranging operations, pre-computing constants, choosing different math functions) have outsized effects on overall performance because they are repeated millions or billions of times.

**JIT (Just-in-Time) Compilation:** Compilation that happens at runtime, the first time a decorated function is called. Numba inspects the argument types, infers the types of all variables in the function, and compiles the function to machine code. The compiled version is cached, so subsequent calls with the same argument types skip compilation. GPU kernels decorated with `@cuda.jit` are compiled to CUDA code in the same way.

**Kernel:** The function body that runs on the GPU. A kernel is called thousands or millions of times, once per thread, and each invocation processes a different piece of data identified by the thread's index. Kernels are decorated with `@cuda.jit` (Numba) and do not return values — outputs are written to arrays passed as arguments.

**Latency Hiding:** The technique by which a GPU avoids sitting idle while waiting for slow operations (such as global memory accesses). When one *warp* stalls, the SM immediately switches to another warp that is ready to run. So long as each SM has a large enough pool of warps, the cores remain busy every cycle. Latency hiding is why high *occupancy* is desirable. Also see *Little's Law*.

**Latency vs. Throughput:** The fundamental trade-off that defines GPU design. *Latency* is the time to complete a single task; *throughput* is the total amount of work completed per unit time. A single GPU core has much higher latency (is slower) than a CPU core, but a GPU achieves vastly higher throughput by running tens of thousands of threads simultaneously. GPU programming is about maximising throughput, not minimising individual-task latency.

**L1 / L2 Cache:** Automatic, hardware-managed caches that sit between global memory and the SM. The L1 cache is per-SM (hundreds of KB); the L2 cache is shared across all SMs (tens of MB). They speed up repeated or spatially local memory accesses without programmer intervention. They are faster than global memory but slower than shared memory.

**Libdevice:** Numba's interface to CUDA's specialised mathematical functions. `cuda.libdevice` provides functions like `sinf()`, `cosf()`, `fast_sincosf()`, and `sincospif()` that compute at specific precisions and use hardware-optimised implementations. These are preferred over Python's standard `math` module inside kernels, because they respect the precision you choose and are often faster.

**Local Memory:** Private per-thread storage used for scalar variables and fixed-size arrays created inside a kernel (e.g., `cuda.local.array`). Logically it is "local" to each thread, but when the SM runs out of physical registers, local variables *spill* into global memory — a severe performance penalty. Minimising the number of local variables per thread helps avoid register spills.

**Little's Law:** A theorem from queueing theory: in a stable system, the average number of items in a queue equals the arrival rate multiplied by the average time each item spends in the system. Applied to latency hiding, it means that with sufficient concurrent warps queued on an SM, even a high-latency operation (like a 400-cycle global memory read) takes only about one cycle *on average* per thread — because the SM keeps other warps busy while any one warp waits.

**MapReduce:** A two-stage parallel pattern: first a *map* transforms each element of an array independently, then a *reduction* combines the transformed elements into a single value (e.g., sum, maximum). For example, computing the mean of `x**2` maps each element to its square, then reduces by summing. On the GPU, the reduce stage is the challenging part, implemented via *parallel reduction*.

**Memory-Bound:** A kernel is memory-bound when its runtime is limited by the rate at which data can be moved to and from memory, not by the rate of computation. Memory-bound kernels benefit from reducing memory traffic (using shared memory, coalescing accesses, thread coarsening) rather than from faster arithmetic. The opposite is *compute-bound*.

**Memory Coalescing:** The process by which the GPU memory controller combines many individual memory requests from a warp into a single, wide memory transaction. See *Coalesced Memory Access* for the access pattern that enables this.

**Moore's Law:** The observation that the number of transistors on a microchip doubles approximately every two years. Until the early 2000s, this translated into faster single-core CPU clock speeds. Physical limits (power, heat) stopped clock-speed growth, and the benefit of Moore's Law now comes from more cores — both on CPUs and especially on GPUs.

**NSIGHT Compute / NSIGHT Systems:** NVIDIA's profiling tools. **NSIGHT Compute** analyses individual kernels: register usage, instruction counts, cache hits, memory accesses, and roofline plots. **NSIGHT Systems** provides a timeline view of the whole program: memory transfers, kernel launches, streams, and dependencies. Both help identify whether a kernel is compute-bound or memory-bound.

**Numba:** A Python library that compiles annotated Python functions to fast, machine-level code using JIT compilation. On the CPU, `@njit` produces near-C speed; on the GPU, `@cuda.jit` produces CUDA kernels. Numba supports a subset of Python (loops, conditionals, NumPy array access, math functions) but not most object-oriented features or dynamic typing.

**Operation Fusion:** The process of combining multiple array operations into a single pass over the data, eliminating intermediate arrays. In CuPy, the `@cupy.fuse` decorator achieves this: without fusion, an expression like `ys = cupy.cos(2 * (xs - 1))` executes as three separate kernel launches (subtraction, multiplication, cosine), each reading and writing the full array; with fusion, all three operations are applied in one pass. Fusion is experimental and works best with simple, element-wise operations.

**Occupancy:** The fraction of GPU resources (SMs, thread blocks, warps) actively utilised during kernel execution. High occupancy means each SM has many warps queued and ready, which improves *latency hiding*. Occupancy is affected by threads per block, number of blocks, register usage per thread, and shared memory usage per block.

**Parallel Reduction:** A two-stage strategy for combining many values into one (e.g., summing an array) on the GPU: first, each thread block performs a local *reduction* using *shared memory* and a *binary reduction* pattern; second, thread 0 of each block uses an *atomic operation* to add the block's partial result into global memory. This minimises the number of (slow) atomic operations — one per block rather than one per thread.

**PCIe / NVLink:** The physical links that connect the GPU to the host CPU and to other GPUs. **PCIe** (Peripheral Component Interconnect Express) is the standard bus for host-to-device communication; **NVLink** is NVIDIA's proprietary high-speed interconnect that offers significantly higher bandwidth than PCIe. Both add latency and cost to memory transfers between host and device.

**prange:** Numba's *parallel range*, used with `@njit(parallel=True)` to distribute loop iterations across CPU threads. `prange` is the CPU equivalent of a GPU kernel's grid: each iteration is independent, and the runtime decides how to assign iterations to threads. Iterations are not guaranteed to execute in order.

**PTX / SASS:** The intermediate and final compilation layers of a CUDA kernel. **PTX** (Parallel Thread Execution) is a virtual assembly language — an intermediate representation that is architecture-independent. **SASS** is the actual machine code for a specific GPU architecture. When Numba compiles a `@cuda.jit` kernel, it produces PTX first, which the CUDA driver then compiles to SASS at runtime. These layers are opaque to the programmer, making it difficult to predict exactly what optimisations the compiler will apply.

**Register Spill:** When a thread uses so many registers that the SM cannot hold them all, the compiler silently moves some registers into global memory (called *local memory*). Because global memory is hundreds of times slower than registers, register spills are extremely costly for performance. They are avoided by keeping per-thread register usage low (fewer local variables, smaller local arrays, lower thread coarsening factors).

**Reduction:** An operation that combines many values into a single result, such as summing an array, finding a maximum, or computing a product. A reduction is one of the most common parallel patterns and also one of the trickiest to implement efficiently on the GPU due to potential race conditions. The idiomatic GPU solution is *parallel reduction*, which uses shared memory to reduce within a block before combining block results atomically.

**Roofline Model:** A visual performance model that plots achievable kernel performance against two limits: peak computational throughput (the "roof") and memory bandwidth (the "slope"). Kernels fall either on the compute-bound side (performance limited by arithmetic capacity) or the memory-bound side (performance limited by data movement). The roofline helps identify whether optimisation effort should target arithmetic intensity or memory traffic. Available via NSIGHT Compute.

**Registers:** The fastest tier of GPU storage — per-thread private variables stored inside the SM, with approximately 1-cycle latency. The register pool is shared by all threads running on an SM (tens of thousands of 32-bit registers per SM); using too many registers per thread reduces the number of concurrent threads and hurts occupancy.

**Shared Memory:** A small, fast pool of memory (hundreds of KB per SM) visible to all threads within a *thread block*. Shared memory is manually managed by the programmer via `cuda.shared.array()` and lives only for the duration of the block. It is used for caching data from global memory, communicating between threads, and implementing parallel reductions. Access is much faster than global memory (~20 cycles vs. ~400 cycles).

**SIMD:** Single Instruction, Multiple Data — a CPU parallelism model where a single instruction operates on a vector of data (e.g., AVX instructions process 4 or 8 floats in one register). The GPU equivalent is *SIMT*.

**SIMT:** Single Instruction, Multiple Thread — the GPU execution model. Threads are grouped into *warps* of 32 threads that execute the same instruction in lockstep. Each thread operates on its own data. If threads within a warp take different branches (*warp divergence*), the branches are executed sequentially with inactive threads masked. The GPU analogue of the CPU concept of *SIMD*.

**Streaming Multiprocessor (SM):** A cluster of CUDA cores, shared resources (shared memory, registers, L1 cache, FPUs), and a warp scheduler, all grouped together on the GPU die. An SM is the fundamental scheduling unit: thread blocks are assigned to a single SM for their lifetime. A modern GPU contains dozens to hundreds of SMs.

**Stream:** See *CUDA Stream*.

**Task Parallelism:** A form of parallelism in which different threads perform different types of work (e.g., one thread assembles frames, another builds tyres). Task parallelism requires coordination (locks, semaphores, message passing) and is the primary model for CPU threading. GPUs use *data parallelism* instead.

**Tensor Cores:** Specialised hardware units on modern NVIDIA GPUs designed for high-throughput matrix operations, particularly matrix multiplication and accumulation. Tensor cores operate at lower precisions (e.g., 16-bit or 8-bit) and deliver orders-of-magnitude more throughput than standard CUDA cores for compatible workloads. They are the primary driver of AI performance on modern GPUs but are less directly accessible from Python kernel code than the general-purpose CUDA cores.

**Thread:** The fundamental unit of parallelism on the GPU. A kernel is launched as millions of threads, each executing the same code but on different data. Threads are identified by a unique grid index obtained via `cuda.grid(1)`.

**Thread Block:** A group of threads (typically 128–1024) that execute on a single SM and can cooperate via *shared memory* and `cuda.syncthreads()`. Thread blocks are the scheduling unit of the GPU — the scheduler assigns blocks to SMs, not individual threads. Threads in different blocks cannot share memory or synchronise directly.

**Thread Coarsening:** An optimisation that reduces the number of thread blocks (lower parallelism) while increasing the amount of work each thread does (higher *computational intensity*). Each thread processes multiple output elements using local arrays, amortising the cost of global memory reads and per-thread initialisation across more computation. The coarsening factor is typically small (2–8) and must be a compile-time constant.

**Type Promotion:** The rule that when values of different precision are combined in an expression, the lower-precision value is silently cast to the higher precision before the operation. For example, combining a `float32` with a `float64` promotes the `float32` to `float64`. Inside kernels, unintended promotion can cause operations to run at higher precision (and lower speed) than intended.

**Warp:** A group of 32 threads (NVIDIA) or 64 threads (AMD) that execute in lockstep under the SIMT model. Warps are the unit of dispatch from an SM to its cores. When a warp stalls (e.g., waiting for memory), the SM switches to another warp — this is the mechanism behind *latency hiding*.

**Warp Divergence:** When threads within a warp take different branches of a conditional (e.g., some threads enter an `if` block and others do not). Because a warp executes in lockstep, both branches are run sequentially: the first branch executes with threads that should not participate masked (no-op), then the second branch executes with the complementary threads masked. Warp divergence wastes processor cycles and should be minimised.

**Warp Shuffle (`cuda.shfl_sync`):** A warp-level intrinsic that moves data directly between the registers of threads within a warp, without going through shared memory. Used for communication patterns like broadcasting a value, reduction, or daisy-chaining data around warp members. Faster than shared memory for small data exchanges but limited to warp scope (32 threads).
