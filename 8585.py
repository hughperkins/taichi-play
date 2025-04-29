import taichi as ti
ti.init(arch=ti.cpu)
N = 20
l=86
triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=l)
@ti.kernel
def fill(triplets: ti.types.ndarray()):
    for i in range(l):
       triplets[i] = ti.Vector([i // N, i % N, i+1], dt=ti.f32)
fill(triplets)
A = ti.linalg.SparseMatrix(n=N, m=N, dtype=ti.f32)
A.build_from_ndarray(triplets)
print(A)
