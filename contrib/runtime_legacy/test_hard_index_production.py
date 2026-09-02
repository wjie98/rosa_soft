import pytest
import torch

import rosa_soft


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="CUDA RosaSoft extension is unavailable",
)


def _unpack_symbols(symbols, bits, dtype):
    return torch.stack(
        [
            ((symbols.to(torch.int64) >> bit) & 1)
            .to(dtype)
            .mul(2)
            .sub(1)
            for bit in range(bits)
        ],
        dim=-1,
    )


def _packed_case(bits, tokens, pattern, seed, batch_size=1):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch_size, tokens, 4)
    query = torch.randint(
        0,
        1 << bits,
        shape,
        generator=generator,
        dtype=torch.uint8,
    )
    key = torch.randint(
        0,
        1 << bits,
        shape,
        generator=generator,
        dtype=torch.uint8,
    )
    if pattern == "aligned":
        query[:, 1:] = key[:, :-1]
    elif pattern.startswith("periodic"):
        period = int(pattern.removeprefix("periodic"))
        codes = torch.randint(
            0,
            1 << bits,
            (batch_size, period, 4),
            generator=generator,
            dtype=torch.uint8,
        )
        positions = torch.arange(tokens) % period
        query = codes[:, positions].clone()
        key = codes[:, positions].clone()
    elif pattern == "all_match":
        query.zero_()
        key.zero_()
    elif pattern == "all_mismatch":
        query.zero_()
        key.fill_((1 << bits) - 1)
    elif pattern != "random":
        raise ValueError(pattern)
    return query, key


def _position_values(tokens, dtype, batch_size=1):
    positions = torch.arange(tokens, dtype=torch.int64).view(1, tokens, 1)
    positions = positions.expand(batch_size, -1, -1)
    value_heads = []
    for value_head in range(2):
        code = positions ^ (value_head * 0x5A5A)
        value_heads.append(_unpack_symbols(code, 16, dtype))
    return torch.cat(value_heads, dim=2)


def _cpu_expected(query, key, value, bits):
    payload = torch.zeros(
        query.size(0),
        query.size(1),
        2,
        dtype=torch.uint8,
    )
    with rosa_soft.RosaRuntime(
        4,
        2,
        qk_bits=bits,
        payload_bits=1,
    ) as runtime:
        _, matched_ends = runtime.update_packed(query, key, payload)
    routes = torch.where(
        matched_ends >= 0,
        matched_ends + 1,
        torch.zeros_like(matched_ends),
    )
    expanded_value = value.repeat_interleave(2, dim=2).permute(0, 2, 1, 3)
    gather_index = routes.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 16)
    expected = torch.gather(expanded_value, 2, gather_index).permute(0, 2, 1, 3)
    return torch.where(
        matched_ends.unsqueeze(-1) >= 0,
        expected,
        torch.zeros((), dtype=value.dtype),
    )


def _assert_production_matches_cpu(
    *,
    bits,
    tokens=4096,
    pattern="random",
    dtype=torch.float16,
    batch_size=1,
):
    query, key = _packed_case(
        bits,
        tokens,
        pattern,
        seed=bits * 100_000 + tokens,
        batch_size=batch_size,
    )
    value = _position_values(tokens, dtype, batch_size)
    expected = _cpu_expected(query, key, value, bits)
    output, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        _unpack_symbols(query, bits, dtype).cuda(),
        _unpack_symbols(key, bits, dtype).cuda(),
        value.cuda(),
    )

    assert torch.equal(packed_query.cpu(), query.permute(0, 2, 1).int())
    assert torch.equal(packed_key.cpu(), key.permute(0, 2, 1).int())
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("bits", range(1, 9))
def test_production_index_matches_cpu_runtime_for_every_supported_code_width(bits):
    _assert_production_matches_cpu(bits=bits)


@pytest.mark.parametrize(
    "pattern",
    ["random", "aligned", "periodic4", "all_match", "all_mismatch"],
)
def test_production_index_matches_cpu_runtime_across_code_regimes(pattern):
    _assert_production_matches_cpu(
        bits=8,
        pattern=pattern,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_production_index_matches_cpu_runtime_across_dtypes(dtype):
    _assert_production_matches_cpu(
        bits=8,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ("bits", "tokens"),
    [(8, 511), (8, 512), (4, 1023), (4, 1024)],
)
def test_hard_index_dispatch_boundaries_match_cpu_runtime(bits, tokens):
    _assert_production_matches_cpu(bits=bits, tokens=tokens)


@pytest.mark.parametrize("tokens", [4095, 4096])
def test_candidate_trajectory_boundary_matches_cpu_runtime(tokens):
    _assert_production_matches_cpu(bits=8, tokens=tokens)


def test_production_index_latest_tie_and_multi_batch_match_cpu_runtime():
    _assert_production_matches_cpu(
        bits=8,
        pattern="all_match",
        batch_size=2,
    )


def test_production_index_full_suffix_certificate_matches_cpu_runtime():
    _assert_production_matches_cpu(
        bits=8,
        pattern="all_match",
    )


@pytest.mark.parametrize(
    ("bits", "tokens", "pattern"),
    [
        (1, 8192, "random"),
        (4, 8192, "random"),
        (8, 8191, "random"),
        (8, 8192, "random"),
        (8, 8192, "periodic64"),
    ],
)
def test_parallel_occurrence_and_heavy_diagonal_paths_match_cpu_runtime(
    bits,
    tokens,
    pattern,
):
    _assert_production_matches_cpu(
        bits=bits,
        tokens=tokens,
        pattern=pattern,
    )


@pytest.mark.parametrize("bits", [8, 9, 32])
def test_fixed_and_varlen_hard_paths_match_at_long_sequence(bits):
    tokens = 4096
    generator = torch.Generator(device="cuda").manual_seed(9000 + bits)
    query = torch.randn(
        1,
        tokens,
        4,
        bits,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(
        query.shape,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn(
        1,
        tokens,
        2,
        16,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    offsets = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")

    dense = torch.ops.rosa_soft.hard_forward(query, key, value)
    packed = torch.ops.rosa_soft.hard_forward_varlen(
        query[0],
        key[0],
        value[0],
        offsets,
    )

    assert torch.equal(dense[0][0], packed[0])
    assert torch.equal(dense[1][0], packed[1])
    assert torch.equal(dense[2][0], packed[2])
