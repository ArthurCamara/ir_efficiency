import os
from typing import List

import numpy as np


def g_path(*argv: str):
    """short hand for creating a new path properly
    args:
        argv: vector of strings to join into a path"""
    return os.path.join(*argv)


def get_free_gpus(n_gpus: int = 2, forced_used: List[int] = []):
    """Searches how many free GPUs you have, and make only n_gpus available
    Quite useful if multiple users are running the same machine,
    so you can automatically pick how many GPUs you need, without juglng GPU IDS
    If you NEED to use any GPU, use forced_use.
    """

    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >/tmp/gpus")
    memory_available = np.array([int(x.split()[2]) for x in open("/tmp/gpus", "r").readlines()])
    good_gpus = list((np.array(memory_available) > 3000).nonzero()[0])
    if len(forced_used) > 0:
        good_gpus = [g for g in good_gpus if g in forced_used]
    if len(good_gpus) == 0:
        raise IndexError("Could not find any free GPUS! Try again later.")
    good_gpus = good_gpus[:n_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, good_gpus))
    return good_gpus


def rawcount(filename: os.PathLike) -> int:
    """Fast count of number of lines on a file.
    Got it from https://stackoverflow.com/a/27518377/1052440"""
    f = open(filename, "rb")
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b"\n")
        buf = read_f(buf_size)

    return lines
