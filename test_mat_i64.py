import taichi as ti
ti.init(arch=ti.arm64)

m = ti.types.matrix(9, 1, float)

@ti.kernel
def get_element(mat: m, i: ti.i32) -> float:
        return mat[0, i]

m[0] = 123
m[4294967296] = 256
print('m[0]', m[0])
print('m[4294967296]', m[4294967296])
