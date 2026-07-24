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

CPU_SOURCES = [
    "export.cpp",
    "rosa_runtime.cpp",
]

CUDA_CPP_SOURCES = [
    "rosa_soft.cpp",
    "rwkv7_albatross.cpp",
    "rwkv7_clampw.cpp",
    "rwkv7_state_clampw.cpp",
    "rwkv7_statepassing_clampw.cpp",
]

CUDA_SOURCES = [
    "cuda/rosa_soft_kernels.cu",
    "cuda/rwkv7_albatross.cu",
    "cuda/rwkv7_clampw.cu",
    "cuda/rwkv7_state_clampw.cu",
    "cuda/rwkv7_statepassing_clampw.cu",
]


def wants_extension_build():
    value = os.getenv("ROSA_BUILD_EXTENSION", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def wants_cuda_build():
    value = os.getenv("USE_CUDA", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_extensions():
    if not wants_extension_build():
        return []
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
    source_names = list(CPU_SOURCES)
    if use_cuda:
        source_names += CUDA_CPP_SOURCES
        source_names += CUDA_SOURCES
    sources = [
        str(extensions_dir / source_name)
        for source_name in source_names
    ]

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
