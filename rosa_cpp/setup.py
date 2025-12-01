import os
import torch

from pathlib import Path
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


library_name = "rosa_cpp"


def get_extensions():
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fopenmp",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3",
        ],
    }

    extensions_dir = Path(__file__).parent / "rosa_cpp" / "csrc"
    sources = list(str(p) for p in extensions_dir.glob("*.cpp"))

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
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Torch ROSA C++ Extension",
    cmdclass={"build_ext": BuildExtension},
)
