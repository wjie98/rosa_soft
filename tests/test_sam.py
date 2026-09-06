import pytest
import torch

from rosa_soft import RosaSam, rosa_hard
from tests.oracle import hard, routes


@pytest.mark.parametrize("bits", [1, 2, 8, 32])
def test_sam_matches_naive_random_routes(bits):
    torch.manual_seed(100 + bits)
    q = torch.randn(2, 41, 3, bits)
    k = torch.randn_like(q)
    actual = RosaSam(3, bits).update(q, k)
    assert torch.equal(actual, routes(q, k))


def test_sam_is_stateful_across_chunks():
    torch.manual_seed(4)
    q = torch.randn(2, 53, 2, 5)
    k = torch.randn_like(q)
    full = RosaSam(2, 5).update(q, k)
    sam = RosaSam(2, 5)
    chunked = torch.cat((sam.update(q[:, :19], k[:, :19]), sam.update(q[:, 19:], k[:, 19:])), 1)
    assert torch.equal(chunked, full)


def test_unlimited_suffix_beats_recent_one_symbol_match():
    t, bits = 71, 8
    code = torch.arange(40)
    motif = torch.where(
        ((code[:, None] >> torch.arange(bits)) & 1).bool(), 1.0, -1.0
    )
    q = -torch.ones(1, t, 1, bits)
    k = torch.ones_like(q)
    k[0, :40, 0] = motif
    k[0, 69, 0] = motif[-1]
    q[0, 31:71, 0] = motif
    route = RosaSam(1, bits).update(q, k)
    assert route[0, -1, 0].item() == 39


def test_hard_value_is_successor_and_gqa_is_exact():
    torch.manual_seed(9)
    q = torch.randn(2, 19, 4, 6)
    k = torch.randn_like(q)
    v = torch.randn(2, 19, 2, 7)
    expected, expected_route = hard(q, k, v)
    actual, route = rosa_hard(q, k, v)
    assert torch.equal(route, expected_route)
    assert torch.equal(actual, expected)


def test_packed_sequences_are_isolated():
    torch.manual_seed(11)
    lengths = [7, 0, 5, 9]
    cu = torch.tensor([0, 7, 7, 12, 21], dtype=torch.int32)
    q = torch.randn(21, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn(21, 1, 3)
    actual, route = rosa_hard(q, k, v, cu)
    outputs = []
    routes_expected = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if start == end:
            continue
        y, r = hard(q[start:end][None], k[start:end][None], v[start:end][None])
        outputs.append(y[0])
        routes_expected.append(r[0])
    assert torch.equal(actual, torch.cat(outputs))
    assert torch.equal(route, torch.cat(routes_expected))


def test_sam_rejects_changed_batch_count_without_reset():
    sam = RosaSam(1, 2)
    sam.update(torch.ones(1, 2, 1, 2), torch.ones(1, 2, 1, 2))
    with pytest.raises(RuntimeError, match="sequence count changed"):
        sam.update(torch.ones(2, 1, 1, 2), torch.ones(2, 1, 1, 2))
