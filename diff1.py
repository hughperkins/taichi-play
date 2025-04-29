import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

x = ti.field(float, (), needs_grad=True)
y = ti.field(float, (), needs_grad=True)

x[None] = 0.9

@ti.kernel
def compute_y():
    y[None] = ti.sin(x[None])

with ti.ad.Tape(y):
    compute_y()
print('x', x[None], 'y', y[None], 'dy/dx', x.grad[None])

n = 1000
x = ti.field(float, (n))
y = ti.field(float, (n))

@ti.kernel
def compute_y(phase: float):
    for i in range(n):
        x[i] = i / n * 2 * ti.math.pi
        y[i] = ti.sin(x[i] + phase)

gui = ti.GUI('sin', res=(800, 400), background_color=0xFFFFFF)
phase = 0.0
while gui.running:
    with ti.ad.Tape(y):
        compute_y(phase)
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    gui_x = x_np / 2 / ti.math.pi
    gui_y = y_np * 0.5 + 0.5
    dy_dx_np = x.grad.to_numpy()
    gui_dy_dx = dy_dx_np * 0.5 + 0.5
    gui.lines(begin=np.column_stack((gui_x[:-1], gui_y[:-1])),
              end=np.column_stack((gui_x[1:], gui_y[1:])),
              color=0x000000, radius=2)
    gui.lines(begin=np.column_stack((gui_x[:-1], gui_dy_dx[:-1])),
              end=np.column_stack((gui_x[1:], gui_dy_dx[1:])),
              color=0xFF0000, radius=2)
    gui.show()
    phase += 0.01
