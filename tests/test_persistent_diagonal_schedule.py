import pytest
import torch

from benchmarks.persistent_diagonal_schedule import (
    SCHEDULES,
    diagonal_schedule,
    load_diagonal_schedule,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.fixture(scope="session")
def schedule_module():
    return load_diagonal_schedule()


@pytest.mark.parametrize("tokens", [2, 17, 32, 33, 79])
@pytest.mark.parametrize("pattern", ["random", "equal", "periodic"])
@pytest.mark.parametrize("worker_blocks", [1, 7, 0])
def test_all_diagonal_schedules_are_equivalent(
    tokens, pattern, worker_blocks, schedule_module
):
    generator = torch.Generator(device="cuda").manual_seed(1700 + tokens)
    query = torch.randint(
        0, 256, (2, 3, tokens), device="cuda", dtype=torch.int32,
        generator=generator,
    )
    key = torch.randint(
        0, 256, query.shape, device="cuda", dtype=torch.int32,
        generator=generator,
    )
    if pattern == "equal":
        key.copy_(query)
    elif pattern == "periodic":
        query.copy_(torch.arange(tokens, device="cuda") % 4)
        key.copy_(query)
    expected = diagonal_schedule(
        query,
        key,
        symbol_dim=8,
        module=schedule_module,
    )
    for schedule in SCHEDULES[1:]:
        actual = diagonal_schedule(
            query,
            key,
            symbol_dim=8,
            schedule=schedule,
            worker_blocks=worker_blocks,
            module=schedule_module,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
