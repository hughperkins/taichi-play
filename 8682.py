import taichi as ti

ti.init(arch=ti.arm64)

global_table = {}

tmp = ti.Vector.field(3, float, shape=(4, 4))
tmp.fill((0.5, 0.5, 0.5))

def init_global_table():
    global_table["debug"] = tmp

init_global_table()

space_blocks = ti.field(ti.i32)
S1 = ti.root.pointer(ti.ijk, (2, 2, 2))
S2 = S1.bitmasked(ti.ijk, (2, 2, 2))
array = S2.dynamic(ti.l, 4, chunk_size=32)
array.place(space_blocks)

@ti.kernel
def shard():
    for i, j, k, l in ti.ndrange(4, 4, 4, 4):
        space_blocks[i, j, k].append(0)

shard()

p = ti.Vector.field(3, dtype=float, shape=(), needs_grad=True)
p[None] = (0.7, 0.7, 0.7)

loss = ti.field(float, shape=(), needs_grad=True)
loss.grad[None] = 1
loss[None] = 0

label = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
label[None] = (0.1, 0.2, 0.5)

def loss_func_py():
    # Alternative: for i in ti.group(x)  # where x is a field
    for _ in range(4):
        x = int(p[None][0] // 1)
        y = int(p[None][1] // 1)
        z = int(p[None][2] // 1)
        for index in range(space_blocks[x, y, z].length()):
            target = global_table["debug"][space_blocks[x, y, z, index], space_blocks[x, y, z, index]]
            loss[None] -= (target.dot(p[None]) - label[None].norm()) * (target.dot(p[None]) - label[None].norm())

@ti.kernel
def loss_func():
    # Alternative: for i in ti.group(x)  # where x is a field
    for _ in range(4):
        x = int(p[None][0] // 1)
        y = int(p[None][1] // 1)
        z = int(p[None][2] // 1)
        for index in range(space_blocks[x, y, z].length()):
            target = global_table["debug"][space_blocks[x, y, z, index], space_blocks[x, y, z, index]]
            loss[None] -= (target.dot(p[None]) - label[None].norm()) * (target.dot(p[None]) - label[None].norm())

'''
# This version does not throw an error, but has different program logic

x = int(p[None][0] // 1)
y = int(p[None][1] // 1)
z = int(p[None][2] // 1)

@ti.kernel
def loss_func():
    for index in range(space_blocks[x, y, z].length()):
        target = global_table["debug"][space_blocks[x, y, z, index], space_blocks[x, y, z, index]]
        loss[None] -= (target.dot(p[None]) - label[None].norm()) * (target.dot(p[None]) - label[None].norm())
'''

loss_func_py()

with ti.ad.Tape(loss=loss):
    loss_func()

print('grad: ', p.grad[None])