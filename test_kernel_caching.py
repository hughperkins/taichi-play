import argparse
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", action="store_true", help="disable offline cache")
parser.add_argument("--part-2", action="store_true", help="disable offline cache")
args = parser.parse_args()

# ti.init(arch=ti.cpu, offline_cache=False)
ti.init(arch=ti.cpu, offline_cache=not args.no_cache, print_ir=False)
# ti.init(arch=ti.cpu)

@ti.data_oriented
class Foo:
    @ti.kernel
    def bar(self, y: ti.template()):
        # x[None] += 1
        y[None] += 2
        # print("bar kernel", x[None])


print('')
print('construct x')
x = ti.field(ti.i32, shape=())
y1 = ti.field(ti.i32, shape=())

print('')
print("construct foo")
foo = Foo()

print('')
print("run foo.bar(y1)")
foo.bar(y=y1)

print('')
print("run foo.bar(y1)")
foo.bar(y=y1)

if args.part_2:
    print('')
    print('construct x')
    x = ti.field(ti.i32, shape=())
    y2 = ti.field(ti.i32, shape=())

    print('')
    print("construct foo")
    foo = Foo()

    print('')
    print("run foo.bar(y1)")
    foo.bar(y=y1)

    print('')
    print("run foo.bar(y2)")
    foo.bar(y=y2)

    y3 = ti.field(ti.i32, shape=())
    print('')
    print("run foo.bar(y3)")
    print('y3.tree_root', y3.snode._snode_tree_id, y3.snode._id)
    foo.bar(y=y3)
    print('y3.tree_root', y3.snode._snode_tree_id, y3.snode._id)
    print("y1", y1[None])
    print("y2", y2[None])
    print("y3", y3[None])
