import multiprocessing as mp
import os
from itertools import product

kinds = ["uniform", "normal"]
num_steps = [4, 16, 64, 128, 256]


def spawn_worker(args):
    kind, num_step = args
    command = f"~/miniconda3/bin/python3 mnist_init.py --kind {kind} --num_steps {num_step}"
    os.system(command)


with mp.Pool(mp.cpu_count()) as p:
    p.map(spawn_worker, product(kinds, num_steps))


def spawn_worker_vanilla(args):
    # kind, num_step = args
    num_step = args
    command = f"~/miniconda3/bin/python3 mnist_init.py --vanilla 1 --num_steps {num_step}"
    os.system(command)


with mp.Pool(mp.cpu_count()) as p:
    p.map(spawn_worker_vanilla, num_steps)
