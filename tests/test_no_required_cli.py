"""Every active script must run with NO command-line arguments.

The convention (see HOW_TO_RUN.md): each script carries its settings in a
``CONFIG`` dict near the top of the file, argparse defaults are read from
that dict, and CLI flags exist only as overrides. This test is the guard
rail: it fails if a script in an active stage (or tools/) grows a
``required=True`` argument, a bare positional, or an argparse parser
without a ``CONFIG`` block.

Pure AST inspection — no script is imported, so no KenLM / LightGBM /
Tesseract is needed. Excluded on purpose: ``archive/`` (frozen code),
``vendor/`` (frozen snapshot), and anything under a ``tests/`` folder
(test modules may take pytest-style arguments).
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE_DIRS = (
    "1_data_preparation",
    "2_training",
    "3_training_evaluation",
    "4_ocr_correction",
    "5_correction_evaluation",
    "tools",
)


def _active_scripts():
    for top in ACTIVE_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, top)):
            dirnames[:] = [d for d in dirnames
                           if d not in ("archive", "tests", "__pycache__")]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def _is_true(node):
    return isinstance(node, ast.Constant) and node.value is True


class TestNoRequiredCli(unittest.TestCase):
    def test_no_required_arguments_anywhere(self):
        problems = []
        for path in _active_scripts():
            rel = os.path.relpath(path, ROOT)
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument"):
                    continue
                kws = {k.arg: k.value for k in node.keywords}
                if _is_true(kws.get("required")):
                    problems.append(f"{rel}:{node.lineno} required=True")
                first = node.args[0] if node.args else None
                if (isinstance(first, ast.Constant)
                        and isinstance(first.value, str)
                        and not first.value.startswith("-")):
                    nargs = kws.get("nargs")
                    optional = "default" in kws or (
                        isinstance(nargs, ast.Constant)
                        and nargs.value in ("?", "*"))
                    if not optional:
                        problems.append(
                            f"{rel}:{node.lineno} positional {first.value!r} "
                            "has no default")
        self.assertEqual(problems, [],
                         "scripts must run with no arguments; "
                         "give the argument a default from CONFIG:\n"
                         + "\n".join(problems))

    def test_every_argparse_script_has_a_config_block(self):
        missing = []
        for path in _active_scripts():
            src = open(path, encoding="utf-8").read()
            if "ArgumentParser" in src and "CONFIG = {" not in src:
                missing.append(os.path.relpath(path, ROOT))
        self.assertEqual(missing, [],
                         "argparse scripts must carry their settings in a "
                         "CONFIG block (see HOW_TO_RUN.md):\n"
                         + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
