import argparse
import os


template = """
import taichi as ti
ti.init(arch=ti.cpu, offline_cache=False)
"""


def run(args: argparse.Namespace) -> None:
    source = template
    kernel_body = """
@ti.kernel
def k1():
    f0()
"""
    func_declarations = []
    func_annot = "@ti.real_func" if args.real_func else "@ti.func"
    for n in range(args.n):
         func_body = f"""{func_annot}
def f{n}():
    print(f"f{n}")
"""
         is_last = n == args.n - 1
         if not is_last:
             func_body += f"    f{n+1}()\n"
         func_declarations.append(func_body)

    launch_kernel = "k1()"

    source += kernel_body + "\n"
    source += "\n".join(func_declarations)
    source += launch_kernel + "\n"
    with open(args.out_file, 'w') as f:
        f.write(source)
    os.system(f"cat {args.out_file}")
    if args.run:
        os.system(f"python {args.out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', type=int, default=3)
    parser.add_argument('-o', '--out-file', type=str, default='/tmp/long.py')
    parser.add_argument('--real-func', action="store_true")
    parser.add_argument("-r", "--run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
