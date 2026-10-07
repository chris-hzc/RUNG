"""Patch torchtyping 0.1.4 so that it imports under PyTorch >= 2.0.

torchtyping subclasses ``type(torch.Tensor)`` (and ``torch.Tensor`` itself), which
recent PyTorch versions forbid. Only ``__instancecheck__`` and ``base_cls`` are
needed for type annotations, so plain ``type`` / no tensor base suffices.

Usage:
    python scripts/patch_torchtyping.py            # patch the installed package
    python scripts/patch_torchtyping.py <dir>      # patch a torchtyping copy in <dir>
"""
import importlib.util
import pathlib
import sys

REPLACEMENTS = [
    ("class _TensorTypeMeta(type(torch.Tensor)):", "class _TensorTypeMeta(type):"),
    ("class TensorType(torch.Tensor, TensorTypeMixin):", "class TensorType(TensorTypeMixin):"),
]


def main():
    if len(sys.argv) > 1:
        pkg_dir = pathlib.Path(sys.argv[1])
    else:
        spec = importlib.util.find_spec("torchtyping")
        if spec is None:
            sys.exit("torchtyping is not installed.")
        pkg_dir = pathlib.Path(spec.submodule_search_locations[0])

    target = pkg_dir / "tensor_type.py"
    src = target.read_text()
    for old, new in REPLACEMENTS:
        if new in src:
            continue
        if old not in src:
            sys.exit(f"[!] unexpected content in {target}; please patch manually.")
        src = src.replace(old, new)
    target.write_text(src)
    print(f"[ok] torchtyping patched: {target}")


if __name__ == "__main__":
    main()
