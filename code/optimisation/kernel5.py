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
        l = 2 * np.float32(np.pi) * ls[lpx, mpx]
        m = 2 * np.float32(np.pi) * ms[lpx, mpx]
        ndash = 2 * np.float32(np.pi) * ndashes[lpx, mpx]

        # Perform sum over all of the visibility data for just one pixel
        pixel = np.complex64(0)
        for u, v, w, datum in zip(us, vs, ws, data):
            phase = u * l + v * m + w * ndash
            sin, cos = cuda.libdevice.fast_sincosf(phase)
            pixel += datum * complex(cos, sin)

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

    return "Hot loop fix", cupy.asnumpy(img_d), result
