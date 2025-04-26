import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)

@ti.kernel
def add(mat: ti.math.mat3, v2: ti.math.vec3) -> ti.math.mat3:
    mat[:, 0] += v2
    return mat

mat = ti.math.mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)
print('mat', mat)
vec = ti.math.vec3(1, 2, 3)
print('vec', vec)
print(add(mat, vec))
