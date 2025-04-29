import taichi as ti

ti.init(debug=True, arch=ti.arm64)

def iterate():
    x = ti.field(ti.i32)
    print('created x')
    y = ti.field(ti.f32)
    print('created y')
    fb = ti.FieldsBuilder()
    print('create fb')
    fb.dense(ti.ij, 8).place(x)
    print('placed x')
    fb.pointer(ti.ij, 8).dense(ti.ij, 4).place(y)
    print('placed y')

    # After this line, `x` and `y` are placed. No more fields can be placed
    # into `fb`.
    #
    # The tree looks like the following:
    # (implicit root)
    #  |
    #  +-- dense +-- place(x)
    #  |
    #  +-- pointer +-- dense +-- place(y)
    fb.finalize()
    print('finalized')

for it in range(3):
    print("=================")
    print("it", it)
    iterate()
    print()

@ti.kernel
def foo(p1: ti.template()):
    p1[None] += 2

x = ti.field(ti.i32, shape=())
print('x', x.snode.snode_tree_id)
foo(x)
print('x', x.snode.snode_tree_id)

y = ti.field(ti.i32, shape=())
print('y', y.snode.snode_tree_id)
foo(y)
print('y', y.snode.snode_tree_id)

foo(x)
print('x', x.snode.snode_tree_id)
foo(y)
print('y', y.snode.snode_tree_id)

@ti.kernel
def foo2(p1: ti.template(), p2: ti.template()): ...
foo2(x, y)
print('x', x.snode.snode_tree_id)
print('y', y.snode.snode_tree_id)
