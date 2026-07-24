import os
import sys

from pathlib import Path
from typing import Mapping, NamedTuple, Optional

from setuptools import find_packages, setup


library_name = "rosa_soft"
PACKAGE_VERSION = "0.1.0"

CPU_SOURCES = [
    "export.cpp",
    "rosa_runtime.cpp",
]

ROSA_CUDA_SOURCES = [
    "rosa_soft.cpp",
    "cuda/rosa_soft_kernels.cu",
]

RWKV7_CPP_SOURCES = [
    "rwkv7_albatross.cpp",
    "rwkv7_clampw.cpp",
    "rwkv7_state_clampw.cpp",
    "rwkv7_statepassing_clampw.cpp",
]

RWKV7_CUDA_SOURCES = [
    "cuda/rwkv7_albatross.cu",
    "cuda/rwkv7_clampw.cu",
    "cuda/rwkv7_state_clampw.cu",
    "cuda/rwkv7_statepassing_clampw.cu",
]


class BuildConfiguration(NamedTuple):
    build_extension: bool
    cuda_mode: str
    use_cuda: bool
    build_rwkv7: bool
    variant: str


def _binary_setting(
    name: str,
    default: str,
    environ: Mapping[str, str],
) -> bool:
    value = environ.get(name, default).strip().lower()
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _cuda_mode(environ: Mapping[str, str]) -> str:
    value = environ.get("USE_CUDA", "auto").strip().lower()
    if value not in {"auto", "0", "1"}:
        raise RuntimeError(
            f"USE_CUDA must be auto, 0, or 1, got {value!r}"
        )
    return value


def wants_extension_build(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    if environ is None:
        environ = os.environ
    return _binary_setting("ROSA_BUILD_EXTENSION", "1", environ)


def resolve_build_configuration(
    cuda_home: Optional[str],
    environ: Optional[Mapping[str, str]] = None,
) -> BuildConfiguration:
    if environ is None:
        environ = os.environ

    build_extension = wants_extension_build(environ)
    cuda_mode = _cuda_mode(environ)
    build_rwkv7 = _binary_setting("ROSA_BUILD_RWKV7", "0", environ)

    if not build_extension:
        return BuildConfiguration(
            build_extension=False,
            cuda_mode=cuda_mode,
            use_cuda=False,
            build_rwkv7=False,
            variant="reference",
        )

    use_cuda = cuda_home is not None if cuda_mode == "auto" else cuda_mode == "1"
    if cuda_mode == "1" and cuda_home is None:
        raise RuntimeError(
            "USE_CUDA=1 but CUDA_HOME was not found. Install a CUDA toolkit "
            "visible to torch.utils.cpp_extension, or use USE_CUDA=0/auto."
        )
    if build_rwkv7 and not use_cuda:
        raise RuntimeError(
            "ROSA_BUILD_RWKV7=1 requires a CUDA extension build; set "
            "USE_CUDA=1 with a valid CUDA_HOME."
        )

    if build_rwkv7:
        variant = "cuda-rwkv7"
    elif use_cuda:
        variant = "cuda"
    else:
        variant = "cpu-runtime"
    return BuildConfiguration(
        build_extension=True,
        cuda_mode=cuda_mode,
        use_cuda=use_cuda,
        build_rwkv7=build_rwkv7,
        variant=variant,
    )


def source_names_for(config: BuildConfiguration):
    if not config.build_extension:
        return []
    source_names = list(CPU_SOURCES)
    if config.use_cuda:
        source_names += ROSA_CUDA_SOURCES
    if config.build_rwkv7:
        source_names += RWKV7_CPP_SOURCES
        source_names += RWKV7_CUDA_SOURCES
    return source_names


def define_macros_for(config: BuildConfiguration):
    macros = []
    if config.use_cuda:
        macros.append(("ROSA_WITH_CUDA", "1"))
    if config.build_rwkv7:
        macros.append(("ROSA_WITH_RWKV7", "1"))
    return macros


def _load_torch_extension_api():
    try:
        import torch.utils.cpp_extension as cpp_extension
    except ImportError as error:
        raise RuntimeError(
            "Building the rosa_soft extension requires an already installed "
            "PyTorch. Install PyTorch first and invoke pip with "
            "--no-build-isolation so the extension uses that exact PyTorch "
            "ABI. Set ROSA_BUILD_EXTENSION=0 for a reference-only wheel."
        ) from error
    return cpp_extension


def get_extensions(cpp_extension=None):
    if not wants_extension_build():
        return []
    if cpp_extension is None:
        cpp_extension = _load_torch_extension_api()

    config = resolve_build_configuration(cpp_extension.CUDA_HOME)
    extension = (
        cpp_extension.CUDAExtension
        if config.use_cuda
        else cpp_extension.CppExtension
    )

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
    if config.use_cuda:
        extra_compile_args["nvcc"] = [
            "-O3",
            "-res-usage",
            "--use_fast_math",
            "-Xptxas",
            "-O3",
            "--extra-device-vectorization",
        ]

    extensions_dir = Path(__file__).parent / library_name / "csrc"
    sources = [
        str(extensions_dir / source_name)
        for source_name in source_names_for(config)
    ]
    return [
        extension(
            f"{library_name}._C",
            sources,
            define_macros=define_macros_for(config),
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


def setup_package():
    if wants_extension_build():
        cpp_extension = _load_torch_extension_api()
        ext_modules = get_extensions(cpp_extension)
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
    else:
        ext_modules = []
        cmdclass = {}

    setup(
        name=library_name,
        version=PACKAGE_VERSION,
        author="Wenjie Huang",
        packages=find_packages(include=[library_name, f"{library_name}.*"]),
        ext_modules=ext_modules,
        install_requires=["torch"],
        extras_require={
            "build": ["ninja"],
        },
        description="ROSA Operations for PyTorch",
        cmdclass=cmdclass,
    )


if __name__ == "__main__":
    setup_package()
