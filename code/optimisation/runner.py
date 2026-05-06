import numpy as np
import matplotlib.pyplot as plt

import kernel1
import kernel2
import kernel3
import kernel4
import kernel5
import kernel6
import kernel7
import kernel8
import kernel9
import kernel10

names = []
times = []
img = None

for i, kernel in enumerate([
    kernel1,
    kernel2,
    kernel3,
    kernel4,
    kernel5,
    kernel6,
    kernel7,
    kernel8,
    kernel9,
    kernel10,
]):
    name, _img, result = kernel.benchmark()

    print(f"{name}: {result.gpu_times.min() * 1000:.0f} ms")

    # Check for relative error against first image
    img = _img if img is None else img
    print(
        "Relative error: ", (np.abs(img - _img) / np.abs(img)).max()
    )

    # Create plot showing relative speedup at each optimisation step
    names.append(name)
    times.append(result.gpu_times.min())

    fig, ax = plt.subplots()
    ax.bar(names, times[0] / np.array(times))
    ax.set_ylim(ymin=0)
    ax.set_ylabel("Speed up (vs. initial kernel)")
    ax.grid(axis="y")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(f"benchmark-kernel{i + 1}.png")
