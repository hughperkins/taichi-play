import taichi as ti
import taichi.profiler as profiler

ti.init(arch=ti.cpu, debug=True)

profiler.print_memory_profiler_info()
