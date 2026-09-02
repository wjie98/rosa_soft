import pytest
import torch

import rosa_soft
from benchmarks.runtime_pageability import (
    ExactPagedSamModel,
    PageAccessProfiler,
)


pytestmark = pytest.mark.skipif(
    not rosa_soft.BUILD_CAPABILITIES.rosa_runtime,
    reason="RosaRuntime extension is unavailable",
)


def _run_model(query, key, chunk_points=()):
    profiler = PageAccessProfiler(
        [(4096, 16384), (16384, 65536)],
        hot_tail_bytes=4096,
    )
    model = ExactPagedSamModel(profiler)
    routes = []
    points = [0, *chunk_points, len(query)]
    for begin, end in zip(points[:-1], points[1:]):
        routes.extend(model.update(query[i], key[i]) for i in range(begin, end))
    return torch.tensor(routes), model, profiler


@pytest.mark.parametrize("pattern", ["random", "skewed", "periodic"])
@pytest.mark.parametrize("seed", range(4))
def test_page_model_matches_native_unlimited_runtime(pattern, seed):
    tokens = 193
    generator = torch.Generator().manual_seed(seed)
    if pattern == "random":
        query = torch.randint(16, (tokens,), generator=generator).tolist()
        key = torch.randint(16, (tokens,), generator=generator).tolist()
    elif pattern == "skewed":
        query = torch.randint(8, (tokens,), generator=generator)
        key = torch.randint(8, (tokens,), generator=generator)
        query.masked_fill_(torch.rand(tokens, generator=generator) < 0.8, 0)
        key.masked_fill_(torch.rand(tokens, generator=generator) < 0.8, 0)
        query, key = query.tolist(), key.tolist()
    else:
        motif = torch.randint(8, (7,), generator=generator).tolist()
        query = [motif[index % len(motif)] for index in range(tokens)]
        key = query.copy()

    actual, _, _ = _run_model(query, key, (1, 31, 97))
    query_tensor = torch.tensor(query, dtype=torch.uint8).view(1, tokens, 1)
    key_tensor = torch.tensor(key, dtype=torch.uint8).view(1, tokens, 1)
    payload = torch.zeros(1, tokens, 1, dtype=torch.uint8)
    with rosa_soft.RosaRuntime(1, 1, 4, 1) as runtime:
        _, expected = runtime.update_packed(query_tensor, key_tensor, payload)
    assert torch.equal(actual, expected.flatten())


def test_page_cache_settings_do_not_change_routes_or_suffix_horizon():
    tokens = 384
    key = [(index * 29) & 255 for index in range(tokens)]
    query = [17] * 128 + key[:-128]
    small, _, small_profiler = _run_model(query, key)

    profiler = PageAccessProfiler([(4096, 1 << 20)], hot_tail_bytes=1 << 20)
    model = ExactPagedSamModel(profiler)
    large = torch.tensor([model.update(q, k) for q, k in zip(query, key)])

    assert torch.equal(small, large)
    assert small[-1] >= 0
    assert small_profiler.report()["caches"][0]["accesses"] > 0


def test_unary_run_stays_compressed_without_changing_exact_routes():
    tokens = 1024
    routes, model, profiler = _run_model([0] * tokens, [0] * tokens)
    expected = torch.tensor([-1, *range(tokens - 1)])
    assert torch.equal(routes, expected)
    assert model.logical_counts()["states"] == 1
    report = profiler.report()
    assert report["logical_bytes"]["state"] == 16
    assert report["logical_bytes"]["payload"] == tokens
