"""Refinement engine.

Each Metropolis move launches one or two short parallel loops, so thread
launch overhead matters. OpenMP is preferred (about 5-20 us per launch versus
40-90 us for TBB on a 32-thread laptop), and the thread count defaults to the
number of physical cores, since OpenMP threads spin-wait and contend when
every hyperthread is used. Set ``NUMBA_THREADING_LAYER`` or
``NUMBA_NUM_THREADS`` to override.
"""

import os

import numba


def physical_cores():
    """Number of physical CPU cores (half the logical CPUs if unknown)."""

    try:
        cores = set()
        physical = core = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    physical = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and core is not None:
                    cores.add((physical, core))
                    physical = core = None
        if core is not None:
            cores.add((physical, core))
        if cores:
            return len(cores)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


if "NUMBA_THREADING_LAYER" not in os.environ:
    numba.config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]

if "NUMBA_NUM_THREADS" not in os.environ:
    numba.set_num_threads(min(numba.config.NUMBA_NUM_THREADS, physical_cores()))
