import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i64)

vec4 = ti.math.vec3
mat32 = ti.types.matrix(3, 2, float)

@ti.kernel
def get_element(mat: mat32, i: ti.i32) -> float:
        return mat[0, i]
        # return mat[ti.i32(0), i]

m=mat32([[11,21], [31,41], [51,61]])
print(get_element(m, 0))
