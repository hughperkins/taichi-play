import taichi as ti

ti.init()
a = ti.Matrix([[True, True], [False, False]])
b = ti.Matrix([[True, False], [True, False]])

print("a & b:\n", a & b)
print("a | b:\n", a | b)
