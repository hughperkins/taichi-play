import taichi as ti

ti.init(arch=ti.arm64)

tmp = ti.field(float, shape=())

space_blocks = ti.field(ti.i32)
S1 = ti.root.pointer(ti.i, 4)
array = S1.dynamic(ti.j, 4, chunk_size=32)
array.place(space_blocks)

p = ti.field(dtype=float, shape=(), needs_grad=True)
loss = ti.field(float, shape=(), needs_grad=True)

@ti.kernel
def loss_func():
    for _ in range(1):
        x = int(p[None])
        for index in range(space_blocks[x].length()):
            target = tmp[None]
            loss[None] -= target

with ti.ad.Tape(loss=loss):
    loss_func()
