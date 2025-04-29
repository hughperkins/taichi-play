import taichi as ti

ti.init(arch=ti.arm64)

int_field: ti.ScalarField = ti.field(int, shape=())
int_field_23: ti.ScalarField = ti.field(int, shape=(2, 3))

print('int_field', int_field, int_field[None])
print('int_field_23', int_field_23, int_field_23[1, 1])
print('int_field', int_field, int_field.shape, int_field.dtype)

vec_field: ti.MatrixField = ti.Vector.field(3, float, shape=())

vec_field = ti.Vector.field(3, float, shape=())
vec_field_23 = ti.Vector.field(3, float, shape=(2, 3))
print('vec_field', vec_field, vec_field[None], vec_field[None][0])
print('vec_field_23', vec_field_23, vec_field_23[1, 1], vec_field_23[1, 1][0])

struct_field = ti.Struct.field({"a": ti.types.vector(3, float)}, shape=(2, 3))
print('struct_field', struct_field)
print('struct_field.a', struct_field.a)
print('struct_field.a[1, 1]', struct_field.a[1, 1])
print('struct_field.a[1, 1][1]', struct_field.a[1, 1][1])
print('struct_field[1, 1]', struct_field[1, 1])
print('struct_field[1, 1].a', struct_field[1, 1].a)
print('struct_field[1, 1].a[0]', struct_field[1, 1].a[0])

struct_type = ti.types.struct(a=ti.types.vector(3, float), b=ti.types.vector(2, int))
print('struct_type', struct_type)

struct_field2 = struct_type.field(shape=(2, 3))
print('struct_field2', struct_field2)
print('struct_field2[1, 1].b[1]', struct_field2[1, 1].b[1])

my_vec3 = ti.types.vector(3, float)
my_vec_field = my_vec3.field(shape=(1, 1))
print('my_vec_field', my_vec_field)
my_vec = my_vec3(2, 3, 4)
print('my_vec', my_vec)

my_struct = struct_type((3, 4, 5), (7, 8))
print('my_struct', my_struct)

# ==================================

x = ti.field(float)
ti.root.place(x)
print('x', x, x.shape, x.dtype)

x = ti.field(float)
ti.root.dense(ti.i, 3).place(x)
print('x', x, x.shape, x.dtype)

x = ti.field(float)
ti.root.dense(ti.ij, (2, 3)).place(x)
print('x', x, x.shape, x.dtype)

x = ti.field(float)
ti.root.dense(ti.i, 3).dense(ti.j, 4).place(x)
print('x', x, x.shape, x.dtype)

M = 128
N = 128
x = ti.field(float)
ti.root.dense(ti.ij, (M // 8, N // 8)).dense(ti.ij, (8, 8)).place(x)
print('x', x, x.shape, x.dtype)

x = ti.field(float, shape=())
print('x.snode', x.snode)
print(dir(x.snode))
for k in ['_name', '_cell_size_bytes', '_offset_bytes_in_parent_cell', '_id', '_dtype', '_path_from_root']:
    print(k, getattr(x.snode, k))
print(x.snode._path_from_root())

print('')
print('parent')
for k in ['_name', '_cell_size_bytes', '_offset_bytes_in_parent_cell', '_id', '_dtype', '_path_from_root']:
    print(k, getattr(x.parent(), k))

# print('')
# print('parent parent')
# for k in ['_name', '_cell_size_bytes', '_offset_bytes_in_parent_cell', '_id', '_dtype', '_path_from_root']:
#     print(k, getattr(x.parent().parent(), k))

# for k, v in x.snode.__dict__.items():
#     print(k, v)
# print(x.snode._name, x.snode)

# print(ti. .print_memory_statistics())
