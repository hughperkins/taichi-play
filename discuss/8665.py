import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True)

T_part = ti.types.struct(pos=ti.math.vec2, obj_id=int)
S_part = ti.root.dynamic(ti.i, 50, chunk_size=3)
parts = T_part.field()
S_part.place(parts)

@ti.kernel
def loop_parts():
    print("loop parts")
    for i in parts:
        print('i', i)
    print('')

parts[2] = T_part(ti.math.vec2(1, 2), 3)
parts[3] = T_part(ti.math.vec2(1, 2), 3)

loop_parts()

parts[13] = T_part(ti.math.vec2(1, 2), 3)

loop_parts()
