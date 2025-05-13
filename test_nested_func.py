import taichi as ti

ti.init(arch=ti.cpu, offline_cache=False, advanced_optimization=True, print_ir=True)

@ti.func
def foo():
    print("foo func")

@ti.kernel
def bar():
    print("bar kernel")
    foo()

@ti.kernel
def car():
    print("car kernel")
    foo()

print('')
bar()
print('')
bar()
print('')
car()
