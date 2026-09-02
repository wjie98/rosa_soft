import pytest
import torch

from benchmarks.runtime_periodic_state import PeriodicSuffixState


def _diagonal_routes(query, key):
    previous = [0] * len(query)
    routes = []
    for query_end in range(len(query)):
        current = [0] * len(query)
        best_length = 0
        best_end = -1
        for key_end in range(query_end):
            if query[query_end] == key[key_end]:
                current[key_end + 1] = previous[key_end] + 1
            length = current[key_end + 1]
            if length >= best_length and length > 0:
                best_length = length
                best_end = key_end
        routes.append(best_end)
        previous = current
    return routes


@pytest.mark.parametrize("period", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("seed", range(4))
def test_periodic_state_matches_exact_diagonal(period, seed):
    generator = torch.Generator().manual_seed(seed)
    motif = torch.randint(8, (period,), generator=generator).tolist()
    tokens = 97
    key = [motif[index % period] for index in range(tokens)]
    query = torch.randint(8, (tokens,), generator=generator).tolist()
    state = PeriodicSuffixState(motif)
    actual = [state.update(q, k) for q, k in zip(query, key)]
    assert actual == _diagonal_routes(query, key)


def test_periodic_state_reduces_nonprimitive_block_and_rejects_deviation():
    state = PeriodicSuffixState([1, 2, 1, 2, 1, 2])
    assert state.period == 2
    state.update(1, 1)
    state.update(2, 2)
    with pytest.raises(ValueError, match="left periodic language"):
        state.update(1, 7)
