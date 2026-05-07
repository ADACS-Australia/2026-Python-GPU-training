import math

import cupy
import cupyx
from numba import cuda
import numpy as np


@cuda.jit
def kernel(us, vs, ws, data, ls, ms, ndashes, img):
    THREAD_COARSEN = 3
    lmns = cuda.local.array((THREAD_COARSEN, 3), np.float32)
    pixels = cuda.local.array((THREAD_COARSEN, 2), np.float32)

    for i in range(THREAD_COARSEN):
        lmpx = cuda.grid(1) + i * cuda.gridsize(1)
        if lmpx < img.size:
            lpx, mpx = divmod(lmpx, ls.shape[1])

            # Retrieve l, m and ndash coordinates associated with this pixel
            lmns[i, 0] = 2 * np.float32(np.pi) * ls[lpx, mpx]
            lmns[i, 1] = 2 * np.float32(np.pi) * ms[lpx, mpx]
            lmns[i, 2] = 2 * np.float32(np.pi) * ndashes[lpx, mpx]

            # Perform sum over all of the visibility data for just one pixel
            pixels[i, 0] = 0
            pixels[i, 1] = 0

    WARPSIZE = cuda.warpsize
    warpid = cuda.threadIdx.x % WARPSIZE
    nextwarpid = (cuda.threadIdx.x + 1) % WARPSIZE

    # Extract input data in batches of WARPSIZE items
    N = data.size
    for offset in range(0, N, WARPSIZE):
        # Each warp member loads its respective visibility data
        i = offset + warpid
        if i < N:
            u, v, w = us[i], vs[i], ws[i]
            datum = data[i]
        else:
            u, v, w, = np.float32(0), np.float32(0), np.float32(0)
            datum = np.complex64(0)

        # Iterate over warp cache
        for _ in range(WARPSIZE):
            for i in range(THREAD_COARSEN):
                phase = cuda.fma(
                    u,
                    lmns[i, 0],
                    cuda.fma(v, lmns[i, 1], w * lmns[i, 2]),
                )
                sin, cos = cuda.libdevice.fast_sincosf(phase)
                pixels[i, 0] = cuda.fma(datum.real, cos, pixels[i, 0])
                pixels[i, 0] = cuda.fma(datum.imag, -sin, pixels[i, 0])
                pixels[i, 1] = cuda.fma(datum.real, sin, pixels[i, 1])
                pixels[i, 1] = cuda.fma(datum.imag, cos, pixels[i, 1])

            # Daisychain the values around members of the warp
            u = cuda.shfl_sync(0xffffff, u, nextwarpid)
            v = cuda.shfl_sync(0xffffff, v, nextwarpid)
            w = cuda.shfl_sync(0xffffff, w, nextwarpid)

            datum = complex(
                cuda.shfl_sync(0xffffff, datum.real, nextwarpid),
                cuda.shfl_sync(0xffffff, datum.imag, nextwarpid)
            )

    for i in range(THREAD_COARSEN):
        lmpx = cuda.grid(1) + i * cuda.gridsize(1)
        if lmpx < img.size:
            lpx, mpx = divmod(lmpx, ls.shape[1])
            img[lpx, mpx] = complex(pixels[i, 0], pixels[i, 1])


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

    nthreads = 512
    nblocks = math.ceil(img_d.size / nthreads / 3)

    result = cupyx.profiler.benchmark(
        lambda: kernel[nblocks, nthreads](
            us_d, vs_d, ws_d, data_d, ls_d, ms_d, ndashes_d, img_d
        ),
        n_repeat=6,
        n_warmup=1,
    )

    return "Warp shuffle", cupy.asnumpy(img_d), result
