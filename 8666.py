import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

vec4i = ti.types.vector(4, ti.i32)

@ti.dataclass
class MyObject:
    id: int
    particles_id: vec4i
    

my_object = MyObject()
my_object.particles_id = [1,2,3,4]
print('my_object', my_object)
