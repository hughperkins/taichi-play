import numpy as np
import taichi as ti
from p8650 import lbm_solver   # adjust import path if needed

ti.init(arch=ti.cpu)

# 1) construct + init
solver = lbm_solver(name="sym_test")
solver.init()

# 2) take one step
solver.UpdateTemp()

# 3) pull back to NumPy
A = solver.old_temperature.to_numpy()   # shape (nx,ny)

# 4) check 180° symmetry
#    e.g. corners (0,8) vs (300,292)
pairs = [((0,8),(300,292)),
         ((300,8),(0,292)),
         ((0,0),(300,300))]

for (i,j),(ii,jj) in pairs:
    print(f"A[{i},{j}] = {A[i,j]:.6f}, A[{ii},{jj}] = {A[ii,jj]:.6f}")

# or assert equality across the whole field:
D = np.abs(A - A[::-1, ::-1])
print("max sym‑diff:", D.max())
assert D.max() == 0.0, "symmetry broken!"