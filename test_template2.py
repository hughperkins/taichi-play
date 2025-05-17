from typing import TypeAlias, cast, Type
import taichi as ti
ti.init(arch=ti.cpu)

a = 1

ti_template = cast(Type, ti.template())

TiTemplate: TypeAlias = ti.template()

@ti.kernel
def test():
    print(a)

@ti.kernel
def test_template(a: ti.template(), b: ti_template):
    print(a, b)

test()
test_template(a, 3)

a = 2
test()
test_template(a, 5)  # should print 2 5, and now does :)
