import taichi as ti

ti.init(arch=ti.arm64)

space_blocks = ti.field(ti.i32)
S1 = ti.root.pointer(ti.i, 4)
array = S1.dynamic(ti.j, 4, chunk_size=32)
array.place(space_blocks)

loss = ti.field(float, shape=(), needs_grad=True)

@ti.kernel
def loss_func():
    for _ in range(1):
        x = int(loss[None])
        for index in range(space_blocks[x].length()):
            ...

with ti.ad.Tape(loss=loss):
    loss_func()
