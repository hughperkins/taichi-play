import taichi as ti

ti.init(arch=ti.gpu)

m = ti.Matrix(n=10, m=10, dt=float)
print('m', m)
