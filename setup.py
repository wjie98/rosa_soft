import os
import sys

from pathlib import Path
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


library_name = "rosa_soft"


def wants_cuda_build():
    value = os.getenv("USE_CUDA", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_extensions():
    use_cuda = wants_cuda_build()
    if use_cuda and CUDA_HOME is None:
        raise RuntimeError(
            "USE_CUDA=1 but CUDA_HOME was not found. Set CUDA_HOME to a CUDA toolkit "
            "path, or set USE_CUDA=0 to build only the CPU-only extension pieces."
        )
    extension = CUDAExtension if use_cuda else CppExtension

    if sys.platform == "win32":
        cxx_args = ["/O2", "/openmp"]
        extra_link_args = []
    else:
        cxx_args = [
            "-O3",
            "-fopenmp",
            "-fdiagnostics-color=always",
        ]
        extra_link_args = ["-fopenmp"]

    extra_compile_args = {"cxx": cxx_args}
    if use_cuda:
        extra_compile_args["nvcc"] = [
            "-O3",
            "-res-usage",
            "--use_fast_math",
            "-Xptxas", "-O3",
            "--extra-device-vectorization",
        ]

    extensions_dir = Path(__file__).parent / library_name / "csrc"
    if use_cuda:
        sources = list(str(p) for p in extensions_dir.glob("*.cpp"))
    else:
        sources = [
            str(extensions_dir / "export.cpp"),
            str(extensions_dir / "rosa_sam.cpp"),
        ]

    extensions_cuda_dir = extensions_dir / "cuda"
    cuda_sources = list(str(p) for p in extensions_cuda_dir.glob("*.cu"))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.1.0",
    author="Wenjie Huang",
    packages=find_packages(include=[library_name, f"{library_name}.*"]),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    extras_require={
        "build": ["ninja"],
    },
    description="ROSA Operations for PyTorch",
    cmdclass={"build_ext": BuildExtension},
)
