import taichi as ti

# ti.init(arch=ti.metal)
ti.init(arch=ti.arm64, debug=True)

def iterate():
    x = ti.field(ti.i32)
    print('created x')
    y = ti.field(ti.f32)
    print('created y')
    fb = ti.FieldsBuilder()
    print('create fb')
    fb.dense(ti.ij, 8).place(x)
    print('placed x')
    fb.dense(ti.ij, 8).dense(ti.ij, 4).place(y)
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

@ti.kernel
def bar(): ...

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

print('ti.root.finalized', ti.root.finalized)

@ti.kernel
def foo2(p1: ti.template(), p2: ti.template()): ...
print('ti.root.finalized', ti.root.finalized)
foo2(x, y)
print('ti.root.finalized', ti.root.finalized)
print('x', x.snode.snode_tree_id)
print('ti.root.finalized', ti.root.finalized)
print('y', y.snode.snode_tree_id)
# print("ti.foo", ti.lang.impl.foo)
print("ti.root", ti.lang.impl.root == ti.root)

print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
x = ti.field(ti.i32)
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
ti.root.place(x)
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
y = ti.field(ti.i32)
ti.root.place(y)
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
# ti.root.finalize()
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())

z = ti.field(ti.i32)
ti.root.place(z)
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
print('x', x.snode.snode_tree_id, 'y', y.snode.snode_tree_id, 'z', z.snode.snode_tree_id)
foo(x)
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
print('x', x.snode.snode_tree_id, 'y', y.snode.snode_tree_id, 'z', z.snode.snode_tree_id)
print('type(ti.root)', type(ti.root))

i = ti.field(ti.i32, shape=())
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
print('i snode tree id ', i.snode.snode_tree_id)
# print('ti.tools.', ti.tools.memory_usage())
bar()
print('ti.root.finalized', ti.root.finalized, ti.root._get_children())
print('i snode tree id ', i.snode.snode_tree_id)
# print(dir(ti.profiler.memory_profiler))
