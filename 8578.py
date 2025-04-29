import taichi as ti

ti.init(arch=ti.gpu, print_ir=True, print_ir_dbg_info = False, offline_cache=False, cache_loop_invariant_global_vars=True)

x = ti.field(float, shape=(3, 5))


@ti.kernel
def repro():
    ti.loop_config(serialize=True)
    for i in range(5):
        x[2, i] = x[2, i] + 1.0
        for j in range(1):
            x[2, i] = x[2, i] - 5.0
            print("x value ", x[2, i])
            for z in range(1):
                idx = 0
                if z == 0:
                    idx = 2
                x_print = x[idx, i]
                print("x value inside ", x_print)
                print("x value inside direct access", x[2, i])

repro()
