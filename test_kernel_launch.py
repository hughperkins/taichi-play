import numpy as np
import taichi as ti
import genesis as gs

ti.init(arch=ti.cpu, offline_cache=False)

@ti.kernel
def foo():
    ...
print("after foo declaration")

@ti.kernel
def bar():
    ...
print("after bar declaration")

nd1i = ti.ndarray(int, shape=(10))
nd1 = ti.ndarray(float, shape=(10))
nd2 = ti.ndarray(float, shape=(10, 7))
rigid_solver = gs.engine.solvers.RigidSolver()
print('')

print("dir(foo)", dir(foo), type(foo), foo)
foo.background_compile()
print("after background compile foo")
print('')

foo()
print("after call foo")
print('')

bar()
print("after call bar")
print('')

rigid_solver._kernel_init_joint_fields(nd1i, nd2, nd1i, nd1i, nd1i, nd1i, nd2)
rigid_solver._kernel_forward_kinematics_links_geoms(nd1i)
