import taichi as ti

ti.init(arch=ti.cpu) # Use CPU is fine for showing a compile-time error

# Arbitrary data
data = ti.field(dtype=ti.i32, shape=4)
data.fill(1)

result = ti.field(dtype=ti.i32, shape=4)
result.fill(0)

# This is a function which does arbitrary operations using a struct-for loop (ti.ndrange)
@ti.func
def a_struct_for_evaluation_function():
    for i, j in ti.ndrange(4, 4):
        result[i] += data[j]

# This is a TI kernel which calls the function in a simple for loop for a given number of iterations.
@ti.kernel
def perform_optimization(iterations: ti.i32):

    for i in range(iterations):
        a_struct_for_evaluation_function()

# Error occurs when calling this function:
#   Calling the function compiles the taichi kernel, but it won't compile.
#   The error states that `struct_for cannot be nested inside a kernel`.
perform_optimization(1000)