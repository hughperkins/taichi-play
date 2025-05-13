import taichi as ti

ti.init(arch=ti.arm64)

x = ti.field(float, shape=())
print('x', x, type(x))

print('about to call x[None]')
y = x[None]
print(type(y))
print(y, type(y))
print(dir(y))
# print(dir(y.snode))

# @ti.kernel
# def foo():
# 	print(type(x[None]))

# foo()
