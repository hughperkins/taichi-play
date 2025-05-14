import argparse
import ast
import json
import time
from os.path import expanduser as expand


def ast_to_dict(node: ast.mod):
    if isinstance(node, ast.AST):
        fields = {k: ast_to_dict(v) for k, v in ast.iter_fields(node)}
        return {
            "type": node.__class__.__name__,
            "fields": fields,
            # Optional: Store line/col info
            "lineno": getattr(node, "lineno", None),
            "col_offset": getattr(node, "col_offset", None),
        }
    elif isinstance(node, list):
        return [ast_to_dict(x) for x in node]
    else:
        return node  # Basic types (str, int, None, etc.)


def run(args: argparse.Namespace) -> None:
    with open(args.in_module_filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    start = time.time()
    ast_str = ast.dump(tree, indent=2)
    with open(args.out_ast_txt, "w") as f:
        f.write(ast_str)
    elapsed_txt_time = time.time() - start
    print("text dump time", elapsed_txt_time)

    start = time.time()
    json_str = json.dumps(ast_to_dict(tree), indent=2)
    with open(args.out_ast_json, "w") as f:
        f.write(json_str)
    elapsed_json_time = time.time() - start
    print("json dump time", elapsed_json_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-module-filepath", default=expand("~/git/Genesis/genesis/engine/solvers/rigid/collider_decomp.py"))
    parser.add_argument("--out-ast-txt", default="/tmp/ast.txt")
    parser.add_argument("--out-ast-json", default="/tmp/ast.json")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
