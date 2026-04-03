# # only.py
# # Requirements:
# #   pip install tree-sitter tree-sitter-python

# from tree_sitter import Language, Parser, Node
# import tree_sitter_python as tspython

# def analyze_python_file(filepath: str):
#     """
#     Parse Python file and extract function information (name, line, decorators, docstring presence)
#     Works with tree-sitter >= 0.22
#     """
#     # ─── Setup parser ────────────────────────────────────────
#     PY_LANGUAGE = Language(tspython.language())
#     parser = Parser(PY_LANGUAGE)

#     # ─── Read file (binary is safer for tree-sitter) ─────────
#     try:
#         with open(filepath, "rb") as f:
#             source_bytes = f.read()
#     except Exception as e:
#         print(f"Error reading file {filepath}: {e}")
#         return []

#     # ─── Parse ───────────────────────────────────────────────
#     try:
#         tree = parser.parse(source_bytes)
#     except Exception as e:
#         print(f"Error parsing {filepath}: {e}")
#         return []

#     root: Node = tree.root_node
#     source = source_bytes.decode("utf-8", errors="replace")  # for text extraction

#     functions = []

#     def collect_function_info(node: Node):
#         if node.type != "function_definition":
#             return

#         name_node = node.child_by_field_name("name")
#         if not name_node:
#             return

#         func_name = source[name_node.start_byte : name_node.end_byte]

#         # Decorators (look upward)
#         decorators = []
#         current = node.prev_sibling
#         while current and current.type == "decorator":
#             dec_text = source[current.start_byte : current.end_byte].strip()
#             decorators.append(dec_text)
#             current = current.prev_sibling

#         # Docstring detection (very basic – first string literal in body)
#         docstring = None
#         body_node = node.child_by_field_name("body")
#         if body_node and len(body_node.children) > 0:
#             first = body_node.children[0]
#             if first.type == "expression_statement":
#                 expr = first.children[0] if len(first.children) > 0 else None
#                 if expr and expr.type in ("string", "concatenated_string"):
#                     doc_text = source[expr.start_byte : expr.end_byte].strip()
#                     # Clean common triple-quote styles
#                     if doc_text.startswith(('"""', "'''")) and doc_text.endswith(('"""', "'''")):
#                         doc_text = doc_text[3:-3].strip()
#                     docstring = doc_text

#         functions.append({
#             "name": func_name,
#             "line": node.start_point[0] + 1,
#             "decorators": decorators,
#             "has_docstring": bool(docstring),
#             "docstring_preview": (docstring or "").split("\n", 1)[0][:80] if docstring else None
#         })

#     # Traverse top-level + check inside classes/functions for nested defs
#     def traverse(node: Node):
#         collect_function_info(node)
#         for child in node.children:
#             traverse(child)

#     traverse(root)

#     return functions


# # ────────────────────────────────────────────────
# #                  Main / Demo
# # ────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) > 1:
#         target_file = sys.argv[1]
#     else:
#         target_file = "test_symbols/utils.py"   # ← change if needed

#     print(f"Analyzing: {target_file}\n")

#     funcs = analyze_python_file(target_file)

#     if not funcs:
#         print("No functions found or parsing failed.")
#     else:
#         for f in sorted(funcs, key=lambda x: x["line"]):
#             print(f"Line {f['line']:4}  {f['name']}")
#             for d in f['decorators']:
#                 print(f"      @ {d}")
#             if f['has_docstring']:
#                 preview = f['docstring_preview']
#                 print(f"      \"\"\"{preview}{'...' if len(preview) > 75 else ''}\"\"\"")
#             print()

# only.py
# pip install tree-sitter tree-sitter-python

from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
from typing import List, Dict, Optional


def analyze_python_file(filepath: str) -> List[Dict]:
    """
    Parse Python file and extract detailed function information:
    name, async, parameters (with defaults & annotations), return type,
    decorators, docstring first line, line range
    """
    # ─── Setup ───────────────────────────────────────────────
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    try:
        with open(filepath, "rb") as f:
            source_bytes = f.read()
    except Exception as e:
        print(f"Cannot read {filepath}: {e}")
        return []

    try:
        tree = parser.parse(source_bytes)
    except Exception as e:
        print(f"Parse error in {filepath}: {e}")
        return []

    root: Node = tree.root_node
    source = source_bytes.decode("utf-8", errors="replace")

    functions: List[Dict] = []

    def extract_parameters(params_node: Optional[Node]) -> List[Dict]:
        if not params_node:
            return []

        result = []
        for child in params_node.children:
            if child.type == "identifier":
                # positional / kwarg-only without default
                name = source[child.start_byte : child.end_byte]
                result.append({
                    "name": name,
                    "default": None,
                    "annotation": None,
                    "kind": "positional"
                })

            elif child.type == "default_parameter":
                # name = value   or   name: type = value
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = source[name_node.start_byte : name_node.end_byte]

                default_node = child.child_by_field_name("value")
                default_text = (
                    source[default_node.start_byte : default_node.end_byte]
                    if default_node else None
                )

                annot_node = child.child_by_field_name("type")
                annot_text = (
                    source[annot_node.start_byte : annot_node.end_byte].strip()
                    if annot_node else None
                )

                result.append({
                    "name": name,
                    "default": default_text,
                    "annotation": annot_text,
                    "kind": "default"
                })

            elif child.type == "typed_parameter":
                # name: type   (without default)
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = source[name_node.start_byte : name_node.end_byte]

                type_node = child.child_by_field_name("type")
                annot = (
                    source[type_node.start_byte : type_node.end_byte].strip()
                    if type_node else None
                )

                result.append({
                    "name": name,
                    "default": None,
                    "annotation": annot,
                    "kind": "typed"
                })

            elif child.type in ("list_parameter", "dict_parameter"):  # *args, **kwargs
                prefix = "*" if child.type == "list_parameter" else "**"
                name_node = child.child_by_field_name("name")
                name = (
                    prefix + source[name_node.start_byte : name_node.end_byte]
                    if name_node else prefix
                )
                result.append({
                    "name": name,
                    "default": None,
                    "annotation": None,
                    "kind": "vararg" if prefix == "*" else "kwarg"
                })

        return result

    def collect_function_info(node: Node):
        if node.type != "function_definition":
            return

        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        func_name = source[name_node.start_byte : name_node.end_byte]

        # async?
        is_async = bool(node.child_by_field_name("async"))

        # decorators
        decorators = []
        cur = node.prev_sibling
        while cur and cur.type == "decorator":
            dec_text = source[cur.start_byte : cur.end_byte].strip()
            decorators.append(dec_text)
            cur = cur.prev_sibling

        # parameters
        params_node = node.child_by_field_name("parameters")
        params_list = extract_parameters(params_node)

        # return type annotation
        return_annot_node = node.child_by_field_name("return_type")
        return_annotation = (
            source[return_annot_node.start_byte : return_annot_node.end_byte].strip()
            if return_annot_node else None
        )

        # basic docstring first line (optional)
        doc_first = None
        body = node.child_by_field_name("body")
        if body and body.children and body.children[0].type == "expression_statement":
            expr = body.children[0].children[0] if body.children[0].children else None
            if expr and expr.type in ("string", "concatenated_string"):
                raw_doc = source[expr.start_byte : expr.end_byte]
                # rough clean — first meaningful line
                content = raw_doc.strip().strip('"').strip("'").lstrip("rR").strip('"').strip("'")
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                doc_first = lines[0][:100] + "…" if lines and len(lines[0]) > 100 else lines[0] if lines else None

        functions.append({
            "name": func_name,
            "is_async": is_async,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "decorators": decorators,
            "parameters": params_list,
            "return_annotation": return_annotation,
            "docstring_first_line": doc_first,
        })

    # Traverse whole tree (catches nested functions too)
    def traverse(node: Node):
        collect_function_info(node)
        for child in node.children:
            traverse(child)

    traverse(root)

    return functions


# ─── Demo / CLI ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    target = sys.argv[1] if len(sys.argv) > 1 else "test_symbols/utils.py"

    print(f"Analyzing: {target}\n")

    results = analyze_python_file(target)

    if not results:
        print("No functions found or failed to parse.")
    else:
        for f in sorted(results, key=lambda x: x["line_start"]):
            print(f"{'async ' if f['is_async'] else ''}def {f['name']}")
            print(f"  Lines: {f['line_start']}–{f['line_end']}")

            if f['decorators']:
                print("  Decorators:")
                for d in f['decorators']:
                    print(f"    {d}")

            if f['parameters']:
                print("  Parameters:")
                for p in f['parameters']:
                    s = f"    {p['name']}"
                    if p['annotation']:
                        s += f": {p['annotation']}"
                    if p['default'] is not None:
                        s += f" = {p['default']}"
                    print(s)
            else:
                print("  Parameters: ()")

            if f['return_annotation']:
                print(f"  -> {f['return_annotation']}")

            if f['docstring_first_line']:
                print(f"  \"\"\"{f['docstring_first_line']}\"\"\"")

            print()