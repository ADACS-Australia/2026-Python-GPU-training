import math

import cupy
import cupyx
from numba import cuda
import numpy as np


@cuda.jit
def kernel(us, vs, ws, data, ls, ms, ndashes, img):
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


def benchmark():
    # Load the visibility data
    data = np.load("visibilities.npz")
    us, vs, ws, data = data["u"], data["v"], data["w"], data["data"]

    # Initialise the image grid and the corresponding l, m and ndash coordinates
    lpx, mpx = np.mgrid[-350:350, -350:350]
    ls, ms = lpx * 0.0005, mpx * 0.0005
    ndashes = np.sqrt(1 - ls**2 - ms**2) - 1

    # Transfer data to the GPU
    img_d = cupy.zeros(ls.shape, dtype=complex)
    us_d, vs_d, ws_d, data_d, ls_d, ms_d, ndashes_d = map(
        cupy.array, [us, vs, ws, data, ls, ms, ndashes]
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

    return "Local accumulator", cupy.asnumpy(img_d), result
