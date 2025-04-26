import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

V0 = 1.0
A0 = 1.0
ev = 1.0
ea = 1.0

n_vertices = 4
n_faces    = 4
vertices = ti.Vector.field(3, float, n_vertices, needs_grad=True)
faces    = ti.Vector.field(3, int,   n_faces)
forces   = ti.Vector.field(3, float, n_vertices)
U        = ti.field(float, shape=(), needs_grad=True)
vol = ti.field(float, shape=(), needs_grad=True)

vertices.from_numpy(np.array([
    [ 1,  1,  1],
    [-1, -1,  1],
    [-1,  1, -1],
    [ 1, -1, -1]], dtype=np.float32))
faces.from_numpy(np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]], dtype=np.int32))

@ti.kernel
def initialize():
    vol[None] = 0.0
    U[None] = 0.0

@ti.kernel
def compute_energy_1():
    for fi in range(n_faces):
        idxs = faces[fi]
        p0 = vertices[idxs[0]]
        p1 = vertices[idxs[1]]
        p2 = vertices[idxs[2]]

        area = 0.5 * (p1 - p0).cross(p2 - p0).norm()
        U[None] += ea * (area / A0 - 1.0) ** 2

        vol[None] += p0.dot((p1 - p0).cross(p2 - p0)) / 6.0

@ti.kernel
def compute_energy_2():
    U[None] += ev * (vol[None] / V0 - 1.0) ** 2

with ti.ad.Tape(loss=U):
    initialize()
    compute_energy_1()
    compute_energy_2()
for i in range(n_vertices):
    forces[i] = -vertices.grad[i]

print("Forces:", forces.to_numpy())
