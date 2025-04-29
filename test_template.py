import taichi as ti
import numpy as np

# ti.init(arch=ti.cpu)
ti.init(arch=ti.arm64, print_ir=True, debug=True)
# ti.init(arch=ti.metal)

# field_type = ti.types.vector(n=3, dtype=float)

int_field = ti.field(int, shape=())
int_field[None] = 5

# foo(a=5)
# foo(a=int_field)

# my_vec = field_type.field(shape=(5))
# print('my_vec', my_vec)

@ti.kernel
def foo(a: ti.template()):
    # print('foo', a[None])
    a[None] = 123456

foo(int_field)
print('int_field', int_field[None])

foo(int_field)
print('int_field', int_field[None])
