import inspect
from pathlib import Path

import pytest
import torch

import rosa_soft


def test_public_surface_is_minimal():
    assert rosa_soft.__all__ == [
        "__version__", "RosaSam", "rosa_hard", "rosa_soft", "rosa_bitflip"
    ]
    for old in (
        "rosa_soft_unbounded",
        "rosa_soft_varlen",
        "rosa_hard_reference",
        "rosa_soft_reference",
    ):
        assert not hasattr(rosa_soft, old)
    assert list(inspect.signature(rosa_soft.rosa_soft).parameters) == [
        "q", "k", "v", "cu_seqlens", "scale", "dropout_p", "mismatch_scale"
    ]
    assert list(inspect.signature(rosa_soft.rosa_bitflip).parameters) == [
        "q", "k", "v", "rows"
    ]


def test_native_surface_is_minimal():
    schemas = {
        schema.name.split("::", 1)[1]
        for schema in torch._C._jit_get_all_schemas()
        if schema.name.startswith("rosa_soft::")
    }
    if not schemas:
        pytest.skip("CPU-only build")
    assert schemas == {"forward", "backward", "bitflip_forward", "bitflip_backward"}


def test_production_sources_do_not_expose_a_suffix_window():
    root = Path(__file__).parents[1] / "rosa_soft"
    text = "\n".join(
        path.read_text()
        for path in root.rglob("*")
        if path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh"}
    )
    assert "max_suffix_length" not in text
    assert "rosa_soft_unbounded" not in text
