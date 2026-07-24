import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "rosa_soft_build_setup",
        ROOT / "setup.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD_SETUP = _load_setup_module()


def test_default_build_is_cpu_without_cuda_home():
    config = BUILD_SETUP.resolve_build_configuration(
        None,
        environ={},
    )
    assert config.cuda_mode == "auto"
    assert config.variant == "cpu-runtime"
    assert not config.use_cuda
    assert not config.build_rwkv7
    assert BUILD_SETUP.source_names_for(config) == [
        "export.cpp",
        "rosa_runtime.cpp",
    ]
    assert BUILD_SETUP.define_macros_for(config) == []


def test_default_cuda_build_has_only_four_rosa_translation_units():
    config = BUILD_SETUP.resolve_build_configuration(
        "/opt/cuda",
        environ={},
    )
    assert config.variant == "cuda"
    assert config.use_cuda
    assert not config.build_rwkv7
    assert BUILD_SETUP.source_names_for(config) == [
        "export.cpp",
        "rosa_runtime.cpp",
        "rosa_soft.cpp",
        "cuda/rosa_soft_kernels.cu",
    ]
    assert BUILD_SETUP.define_macros_for(config) == [
        ("ROSA_WITH_CUDA", "1"),
    ]


def test_rwkv7_is_explicit_and_requires_cuda():
    with pytest.raises(RuntimeError, match="ROSA_BUILD_RWKV7=1 requires"):
        BUILD_SETUP.resolve_build_configuration(
            None,
            environ={
                "USE_CUDA": "0",
                "ROSA_BUILD_RWKV7": "1",
            },
        )

    config = BUILD_SETUP.resolve_build_configuration(
        "/opt/cuda",
        environ={
            "USE_CUDA": "1",
            "ROSA_BUILD_RWKV7": "1",
        },
    )
    sources = BUILD_SETUP.source_names_for(config)
    assert config.variant == "cuda-rwkv7"
    assert len(sources) == 12
    assert set(BUILD_SETUP.RWKV7_CPP_SOURCES).issubset(sources)
    assert set(BUILD_SETUP.RWKV7_CUDA_SOURCES).issubset(sources)
    assert BUILD_SETUP.define_macros_for(config) == [
        ("ROSA_WITH_CUDA", "1"),
        ("ROSA_WITH_RWKV7", "1"),
    ]


def test_explicit_cuda_is_strict_and_values_are_unambiguous():
    with pytest.raises(RuntimeError, match="USE_CUDA=1"):
        BUILD_SETUP.resolve_build_configuration(
            None,
            environ={"USE_CUDA": "1"},
        )
    with pytest.raises(RuntimeError, match="USE_CUDA must be auto, 0, or 1"):
        BUILD_SETUP.resolve_build_configuration(
            None,
            environ={"USE_CUDA": "true"},
        )
    with pytest.raises(RuntimeError, match="ROSA_BUILD_RWKV7 must be 0 or 1"):
        BUILD_SETUP.resolve_build_configuration(
            "/opt/cuda",
            environ={"ROSA_BUILD_RWKV7": "yes"},
        )


def test_reference_build_overrides_native_feature_requests():
    config = BUILD_SETUP.resolve_build_configuration(
        None,
        environ={
            "ROSA_BUILD_EXTENSION": "0",
            "USE_CUDA": "1",
            "ROSA_BUILD_RWKV7": "1",
        },
    )
    assert config.variant == "reference"
    assert not config.build_extension
    assert not config.use_cuda
    assert not config.build_rwkv7
    assert BUILD_SETUP.source_names_for(config) == []


def test_get_extensions_is_configuration_only(monkeypatch):
    captured = {}

    class FakeExtension:
        def __init__(self, name, sources, **kwargs):
            captured["name"] = name
            captured["sources"] = sources
            captured["kwargs"] = kwargs

    fake_api = SimpleNamespace(
        CUDA_HOME="/opt/cuda",
        CUDAExtension=FakeExtension,
        CppExtension=FakeExtension,
    )
    monkeypatch.delenv("ROSA_BUILD_EXTENSION", raising=False)
    monkeypatch.delenv("USE_CUDA", raising=False)
    monkeypatch.delenv("ROSA_BUILD_RWKV7", raising=False)

    extensions = BUILD_SETUP.get_extensions(fake_api)

    assert len(extensions) == 1
    assert captured["name"] == "rosa_soft._C"
    assert len(captured["sources"]) == 4
    assert captured["kwargs"]["define_macros"] == [
        ("ROSA_WITH_CUDA", "1"),
    ]


def test_pep517_contract_requires_installed_torch_abi():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    assert pyproject["build-system"]["requires"] == [
        "setuptools",
        "wheel",
    ]
    build = pyproject["tool"]["rosa-soft"]["build"]
    assert build["requires-no-build-isolation"] is True
    assert build["use-cuda-default"] == "auto"
    assert build["build-rwkv7-default"] is False


def test_export_schema_guards_and_mutation_contract():
    source = (ROOT / "rosa_soft" / "csrc" / "export.cpp").read_text()

    assert "#ifdef ROSA_WITH_CUDA" in source
    assert "#ifdef ROSA_WITH_RWKV7" in source
    assert (
        "Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa"
        in source
    )
    assert (
        "Tensor(a!) y, Tensor(b!) sT, Tensor(c!) s, Tensor(d!) sa"
        in source
    )
    assert (
        "Tensor dy, Tensor dsT, Tensor s, Tensor sa, "
        "Tensor(a!) ds0"
        in source
    )
    assert (
        "Tensor(a!) s0, Tensor r, Tensor w, Tensor k, Tensor v, "
        "Tensor a, Tensor b, Tensor(b!) y, Tensor elapsed_t"
        in source
    )


def test_public_capabilities_and_placeholders_are_stable():
    pytest.importorskip("torch")
    import rosa_soft

    expected_public = {
        "__version__",
        "BUILD_CAPABILITIES",
        "EXTENSION_IMPORT_ERROR",
        "EXTENSION_IMPORT_ERROR_MESSAGE",
        "HAS_COMPILED_EXTENSION",
        "HAS_ROSA_RUNTIME",
        "HAS_ROSA_SOFT_CUDA",
        "HAS_RWKV7_CUDA",
        "RosaRuntime",
        "RosaRuntimeWork",
        "rosa_soft",
        "rosa_soft_reference",
    }
    assert expected_public.issubset(rosa_soft.__all__)
    assert all(hasattr(rosa_soft, name) for name in expected_public)
    assert rosa_soft.__version__ == BUILD_SETUP.PACKAGE_VERSION
    assert isinstance(rosa_soft.BUILD_CAPABILITIES, dict)
    assert (
        rosa_soft.BUILD_CAPABILITIES["version"]
        == rosa_soft.__version__
    )

    if not rosa_soft.HAS_COMPILED_EXTENSION:
        assert rosa_soft.EXTENSION_IMPORT_ERROR is not None
        assert rosa_soft.EXTENSION_IMPORT_ERROR_MESSAGE
    if not rosa_soft.HAS_ROSA_SOFT_CUDA:
        with pytest.raises(
            RuntimeError,
            match="rosa_soft CUDA training operator is unavailable",
        ):
            rosa_soft.rosa_soft(None, None, None)
    if not rosa_soft.HAS_ROSA_RUNTIME:
        with pytest.raises(RuntimeError, match="RosaRuntime is unavailable"):
            rosa_soft.RosaRuntime(1)


def test_cpu_runtime_registers_no_cuda_operator_schemas():
    torch = pytest.importorskip("torch")
    import rosa_soft

    if rosa_soft.BUILD_CAPABILITIES["variant"] != "cpu-runtime":
        pytest.skip("requires the CPU runtime extension variant")

    operators = [
        "rosa_soft::soft_forward",
        "rosa_soft::soft_backward",
        "rosa_soft::rwkv7_clampw_forward",
        "rosa_soft::rwkv7_albatross_forward_w0_fp16_dither",
    ]
    for operator in operators:
        with pytest.raises(RuntimeError):
            torch._C._dispatch_find_schema_or_throw(operator, "")
