import ast
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
    CppExtension,
)

ROOT = Path(__file__).parent
SRC = ROOT / "rosa_soft" / "csrc"


def version() -> str:
    tree = ast.parse((ROOT / "rosa_soft" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(x, ast.Name) and x.id == "__version__"
            for x in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("missing __version__")


mode = os.environ.get("USE_CUDA", "auto").lower()
if mode not in {"auto", "0", "1"}:
    raise RuntimeError("USE_CUDA must be auto, 0, or 1")
cuda = CUDA_HOME is not None if mode == "auto" else mode == "1"
if cuda and CUDA_HOME is None:
    raise RuntimeError("USE_CUDA=1 but CUDA_HOME was not found")

sources = [SRC / "export.cpp", SRC / "sam.cpp"]
macros = []
args = {"cxx": ["-O3"]}
extension = CppExtension
if cuda:
    extension = CUDAExtension
    macros.append(("ROSA_WITH_CUDA", "1"))
    sources += [
        SRC / "rosa_soft.cpp",
        SRC / "cuda" / "hard.cu",
        SRC / "cuda" / "soft.cu",
        SRC / "cuda" / "soft_fp16.cu",
    ]
    args["nvcc"] = ["-O3", "--use_fast_math", "-Xptxas", "-O3"]

setup(
    name="rosa_soft",
    version=version(),
    description="Hard ROSA routing with a dense soft training gradient",
    packages=find_packages(include=["rosa_soft"]),
    ext_modules=[
        extension(
            "rosa_soft._C",
            [str(x) for x in sources],
            define_macros=macros,
            extra_compile_args=args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.11,<2.12"],
    python_requires=">=3.10",
    extras_require={"test": ["pytest"]},
)
