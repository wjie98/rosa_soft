import importlib.util
import sys
import tomllib
from dataclasses import FrozenInstanceError, is_dataclass
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
    config = BUILD_SETUP.resolve_build_configuration(None, environ={})

    assert config.build_extension
    assert not config.use_cuda
    assert BUILD_SETUP.source_names_for(config) == [
        "export.cpp",
        "rosa_runtime.cpp",
    ]
    assert BUILD_SETUP.define_macros_for(config) == []


def test_default_cuda_build_contains_only_core_translation_units():
    config = BUILD_SETUP.resolve_build_configuration(
        "/opt/cuda",
        environ={},
    )

    assert config.build_extension
    assert config.use_cuda
    assert BUILD_SETUP.source_names_for(config) == [
        "export.cpp",
        "rosa_runtime.cpp",
        "rosa_soft.cpp",
        "cuda/rosa_soft_kernels.cu",
    ]
    assert BUILD_SETUP.define_macros_for(config) == [
        ("ROSA_WITH_CUDA", "1"),
    ]


def test_packed_symbol_shape_contracts_are_explicit_without_cuda():
    soft_source = (ROOT / "rosa_soft" / "soft.py").read_text()
    native_source = (
        ROOT / "rosa_soft" / "csrc" / "rosa_soft.cpp"
    ).read_text()

    assert (
        "packed_symbol_shape = (\n"
        "        query.shape[0],\n"
        "        query.shape[2],\n"
        "        query.shape[1],\n"
        "    )"
    ) in soft_source

    dense_validation = native_source[
        native_source.index("void check_packed_symbols("):
        native_source.index("void check_surrogate_vjp_inputs(")
    ]
    assert '" must have shape (B, H, T)"' in dense_validation
    assert "packed_symbols.size(1) == query.size(2)" in dense_validation
    assert "packed_symbols.size(2) == query.size(1)" in dense_validation

    varlen_validation = native_source[
        native_source.index("void check_varlen_packed_symbols("):
        native_source.index("}  // namespace")
    ]
    assert '" must have shape (H, N)"' in varlen_validation
    assert (
        "packed_symbols.size(0) == query.size(1) &&\n"
        "          packed_symbols.size(1) == query.size(0)"
    ) in varlen_validation


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


def test_reference_build_skips_native_configuration():
    config = BUILD_SETUP.resolve_build_configuration(
        None,
        environ={
            "ROSA_BUILD_EXTENSION": "0",
            "USE_CUDA": "1",
        },
    )

    assert not config.build_extension
    assert not config.use_cuda
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

    extensions = BUILD_SETUP.get_extensions(fake_api)

    assert len(extensions) == 1
    assert captured["name"] == "rosa_soft._C"
    assert [Path(path).name for path in captured["sources"]] == [
        "export.cpp",
        "rosa_runtime.cpp",
        "rosa_soft.cpp",
        "rosa_soft_kernels.cu",
    ]
    assert captured["kwargs"]["define_macros"] == [
        ("ROSA_WITH_CUDA", "1"),
    ]


def test_setup_metadata_uses_static_package_version_and_python_floor(
    monkeypatch,
):
    captured = {}
    monkeypatch.setenv("ROSA_BUILD_EXTENSION", "0")
    monkeypatch.setattr(
        BUILD_SETUP,
        "setup",
        lambda **kwargs: captured.update(kwargs),
    )

    BUILD_SETUP.setup_package()

    init_source = (ROOT / "rosa_soft" / "__init__.py").read_text()
    assert f'__version__ = "{BUILD_SETUP.PACKAGE_VERSION}"' in init_source
    assert captured["version"] == BUILD_SETUP.PACKAGE_VERSION
    assert captured["python_requires"] == ">=3.10"
    assert captured["install_requires"] == [
        BUILD_SETUP.TORCH_REQUIREMENT
    ]
    assert captured["ext_modules"] == []


def test_pep517_contract_requires_the_installed_torch_abi():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    assert pyproject["build-system"]["requires"] == [
        "setuptools",
        "wheel",
    ]
    assert "torch" not in pyproject["build-system"]["requires"]


def test_core_package_and_build_have_no_rwkv_sources_or_entry_points():
    setup_source = (ROOT / "setup.py").read_text()
    for token in (
        "ROSA_BUILD_RWKV7",
        "ROSA_WITH_RWKV7",
        "RWKV7_CPP_SOURCES",
        "RWKV7_CUDA_SOURCES",
    ):
        assert token not in setup_source

    assert not (ROOT / "rosa_soft" / "ops.py").exists()
    assert not (ROOT / "rosa_soft" / "rwkv7.py").exists()
    assert not list((ROOT / "rosa_soft" / "csrc").glob("rwkv7_*.cpp"))
    assert not list(
        (ROOT / "rosa_soft" / "csrc" / "cuda").glob("rwkv7_*.cu")
    )

    archive = ROOT / "contrib" / "rwkv7_legacy"
    assert (archive / "python" / "ops.py").is_file()
    assert (archive / "python" / "rwkv7.py").is_file()


def test_only_missing_extension_module_selects_reference_mode(monkeypatch):
    pytest.importorskip("torch")
    import rosa_soft

    module_name = "rosa_soft._C"

    def missing_extension(name):
        raise ModuleNotFoundError(
            f"No module named {name!r}",
            name=name,
        )

    monkeypatch.setattr(
        rosa_soft._importlib,
        "import_module",
        missing_extension,
    )
    assert rosa_soft._load_compiled_extension() is None

    def missing_dependency(name):
        del name
        raise ModuleNotFoundError(
            "No module named 'extension_dependency'",
            name="extension_dependency",
        )

    monkeypatch.setattr(
        rosa_soft._importlib,
        "import_module",
        missing_dependency,
    )
    with pytest.raises(ModuleNotFoundError, match="extension_dependency"):
        rosa_soft._load_compiled_extension()

    def broken_extension(name):
        del name
        raise OSError("undefined symbol: incompatible_torch_abi")

    monkeypatch.setattr(
        rosa_soft._importlib,
        "import_module",
        broken_extension,
    )
    with pytest.raises(OSError, match="incompatible_torch_abi"):
        rosa_soft._load_compiled_extension()

    assert module_name == f"{rosa_soft.__name__}._C"


def test_public_capabilities_and_placeholders_are_stable():
    pytest.importorskip("torch")
    import rosa_soft

    assert rosa_soft.__all__ == [
        "__version__",
        "BUILD_CAPABILITIES",
        "RosaRuntime",
        "rosa_soft",
        "rosa_soft_reference",
        "rosa_soft_varlen",
        "rosa_soft_varlen_reference",
    ]
    assert rosa_soft.__version__ == BUILD_SETUP.PACKAGE_VERSION
    assert is_dataclass(rosa_soft.BUILD_CAPABILITIES)
    assert isinstance(
        rosa_soft.BUILD_CAPABILITIES,
        rosa_soft.BuildCapabilities,
    )
    with pytest.raises(FrozenInstanceError):
        rosa_soft.BUILD_CAPABILITIES.variant = "mutated"

    for removed_name in (
        "EXTENSION_IMPORT_ERROR",
        "EXTENSION_IMPORT_ERROR_MESSAGE",
        "HAS_RWKV7_CUDA",
        "HAS_RWKV7_CLAMPW_CUDA",
        "HAS_RWKV7_STATE_CLAMPW_CUDA",
        "HAS_RWKV7_STATE_PASSING_CLAMPW_CUDA",
        "HAS_RWKV7_ALBATROSS_CUDA",
        "HAS_COMPILED_EXTENSION",
        "HAS_ROSA_RUNTIME",
        "HAS_ROSA_SOFT_CUDA",
        "ROSA_SOFT_DEFAULT_MISMATCH_SCALE",
        "ROSA_SOFT_DEFAULT_SCALE",
        "RosaRuntimeWork",
    ):
        assert not hasattr(rosa_soft, removed_name)

    capabilities = rosa_soft.BUILD_CAPABILITIES
    assert capabilities.rosa_soft_cuda <= capabilities.compiled_extension
    assert capabilities.rosa_runtime <= capabilities.compiled_extension
    assert capabilities.variant in {"reference", "cpu-runtime", "cuda"}
    if not capabilities.compiled_extension:
        assert capabilities.variant == "reference"
    if not capabilities.rosa_soft_cuda:
        with pytest.raises(
            RuntimeError,
            match="rosa_soft CUDA training operator is unavailable",
        ):
            rosa_soft.rosa_soft(None, None, None)
        with pytest.raises(
            RuntimeError,
            match="rosa_soft_varlen CUDA training operator is unavailable",
        ):
            rosa_soft.rosa_soft_varlen(None, None, None, None)
    if not capabilities.rosa_runtime:
        with pytest.raises(RuntimeError, match="RosaRuntime is unavailable"):
            rosa_soft.RosaRuntime(1)


def test_partial_cuda_registration_is_rejected():
    pytest.importorskip("torch")
    import rosa_soft

    assert rosa_soft._required_cuda_operators == (
        "hard_forward",
        "hard_forward_varlen",
        "surrogate_vjp_masked",
        "surrogate_vjp_varlen_masked",
    )
    assert not rosa_soft._require_complete_cuda_registration(False, False)
    assert rosa_soft._require_complete_cuda_registration(True, True)
    with pytest.raises(RuntimeError, match="incomplete CUDA operator"):
        rosa_soft._require_complete_cuda_registration(True, False)
    with pytest.raises(RuntimeError, match="incomplete CUDA operator"):
        rosa_soft._require_complete_cuda_registration(False, True)
