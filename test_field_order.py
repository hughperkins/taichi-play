import argparse
import numpy as np
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument('--swap-order', action='store_true', help='Swap the order of the fields')
args = parser.parse_args()

ti.init(arch=ti.cpu, offline_cache_file_path='/tmp/foo')

@ti.kernel
def foo():
    x[None] += 1

if args.swap_order:
    x = ti.field(int, shape=())
    y = ti.field(int, shape=())
else:
    y = ti.field(int, shape=())
    x = ti.field(int, shape=())

foo()
