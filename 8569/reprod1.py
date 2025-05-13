import taichi as ti

ti.init(arch=ti.arm64, debug=True)

fields = []
for i in range(20):
    fields.append(ti.field(float, shape=()))
    ti.sync()

@ti.kernel
def foo():
    fields[19][None] = 1

foo()
foo()
