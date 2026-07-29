import ast
import os
import sys

from pathlib import Path
from typing import Mapping, NamedTuple, Optional

from setuptools import find_packages, setup


library_name = "rosa_soft"
TORCH_REQUIREMENT = "torch>=2.11,<2.12"


def _read_package_version() -> str:
    init_path = Path(__file__).parent / library_name / "__init__.py"
    module = ast.parse(init_path.read_text(encoding="utf-8"), init_path.name)
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        version = ast.literal_eval(statement.value)
        if isinstance(version, str):
            return version
    raise RuntimeError(f"Unable to find __version__ in {init_path}")


PACKAGE_VERSION = _read_package_version()

CPU_SOURCES = [
    "export.cpp",
    "rosa_runtime.cpp",
]

ROSA_CUDA_SOURCES = [
    "rosa_soft.cpp",
    "cuda/rosa_soft_kernels.cu",
]


class BuildConfiguration(NamedTuple):
    build_extension: bool
    use_cuda: bool


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

    if not wants_extension_build(environ):
        return BuildConfiguration(False, False)

    cuda_mode = _cuda_mode(environ)
    use_cuda = cuda_home is not None if cuda_mode == "auto" else cuda_mode == "1"
    if cuda_mode == "1" and cuda_home is None:
        raise RuntimeError(
            "USE_CUDA=1 but CUDA_HOME was not found. Install a CUDA toolkit "
            "visible to torch.utils.cpp_extension, or use USE_CUDA=0/auto."
        )

    return BuildConfiguration(True, use_cuda)


def source_names_for(config: BuildConfiguration):
    if not config.build_extension:
        return []
    source_names = list(CPU_SOURCES)
    if config.use_cuda:
        source_names += ROSA_CUDA_SOURCES
    return source_names


def define_macros_for(config: BuildConfiguration):
    macros = []
    if config.use_cuda:
        macros.append(("ROSA_WITH_CUDA", "1"))
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
        install_requires=[TORCH_REQUIREMENT],
        python_requires=">=3.10",
        extras_require={
            "build": ["ninja"],
            "test": ["numpy", "pytest"],
        },
        description="ROSA Operations for PyTorch",
        cmdclass=cmdclass,
    )


if __name__ == "__main__":
    setup_package()
