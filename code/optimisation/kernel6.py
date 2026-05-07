import math

import cupy
import cupyx
from numba import cuda
import numpy as np


@cuda.jit
def kernel(us, vs, ws, data, ls, ms, ndashes, img):
    NTHREADS = 256
    uvw_cache = cuda.shared.array((NTHREADS, 3), dtype=np.float32)
    data_cache = cuda.shared.array(NTHREADS, dtype=np.complex64)

    lmpx = cuda.grid(1)

    l, m, ndash = np.float32(0), np.float32(0), np.float32(0)
    if lmpx < img.size:
        lpx, mpx = divmod(lmpx, ls.shape[1])

        # Retrieve l, m and ndash coordinates associated with this pixel
        l = 2 * np.float32(np.pi) * ls[lpx, mpx]
        m = 2 * np.float32(np.pi) * ms[lpx, mpx]
        ndash = 2 * np.float32(np.pi) * ndashes[lpx, mpx]

    # Perform sum over all of the visibility data for just one pixel
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
        for j in range(NTHREADS):
            phase = (
                uvw_cache[j, 0] * l +
                uvw_cache[j, 1] * m +
                uvw_cache[j, 2] * ndash
            )
            sin, cos = cuda.libdevice.fast_sincosf(phase)
            pixel += data_cache[j] * complex(cos, sin)

        # Don't start updating cache until all threads are done
        cuda.syncthreads()

    if lmpx < img.size:
        img[lpx, mpx] = pixel


def benchmark():
    # Load the visibility data
    data = np.load("visibilities.npz")
    us, vs, ws, data = data["u"], data["v"], data["w"], data["data"]

    # Initialise the image grid and the corresponding l, m and ndash coordinates
    lpx, mpx = np.mgrid[-350:350, -350:350]
    ls, ms = lpx * 0.0005, mpx * 0.0005
    ndashes = np.sqrt(1 - ls**2 - ms**2) - 1

    # Transfer data to the GPU
    img_d = cupy.zeros(ls.shape, dtype=np.complex64)
    data_d = cupy.array(data, dtype=np.complex64)
    us_d, vs_d, ws_d, ls_d, ms_d, ndashes_d = map(
        lambda x: cupy.array(x, dtype=np.float32), [us, vs, ws, ls, ms, ndashes]
    )

    nthreads = 256
    nblocks = math.ceil(img_d.size / nthreads)

    result = cupyx.profiler.benchmark(
        lambda: kernel[nblocks, nthreads](
            us_d, vs_d, ws_d, data_d, ls_d, ms_d, ndashes_d, img_d
        ),
        n_repeat=1,
        n_warmup=1,
    )

    return "Shared memory", cupy.asnumpy(img_d), result
