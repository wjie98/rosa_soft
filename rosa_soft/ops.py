import torch
from torch import Tensor
from typing import Tuple, Optional

__all__ = [
    "rosa_cache_create_",
    "rosa_cache_delete_",
    "rosa_cache_update_",
]

def rosa_cache_create_(cache: Tensor, num_heads: int) -> None:
    torch.ops.rosa_soft.rosa_cache_create(cache, num_heads)

def rosa_cache_delete_(cache: Tensor) -> None:
    torch.ops.rosa_soft.rosa_cache_delete(cache)

def rosa_cache_update_(
        cache: Tensor, batch: Tensor,
        query: Tensor, key: Tensor, value: Tensor,
        query_trigger: Optional[Tensor], key_trigger: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    B = cache.numel()
    nqk, ntk = query.size()
    nvv, _   = value.size()

    assert query.size() == (nqk, ntk), f"query.size()={query.size()} != (nqk, ntk)={(nqk, ntk)}"
    assert key.size() == (nqk, ntk), f"key.size()={key.size()} != (nqk, ntk)={(nqk, ntk)}"
    assert value.size() == (nvv, ntk), f"value.size()={value.size()} != (nvv, ntk)={(nvv, ntk)}"

    assert query.is_contiguous(), f"query is not contiguous, query.size()={query.size()}, query.stride()={query.stride()}"
    assert key.is_contiguous(), f"key is not contiguous, key.size()={key.size()}, key.stride()={key.stride()}"
    assert value.is_contiguous(), f"value is not contiguous, value.size()={value.size()}, value.stride()={value.stride()}"

    if isinstance(query_trigger, Tensor):
        assert query_trigger.size() == (nqk, ntk), f"query_trigger.size()={query_trigger.size()} != (nqk, ntk)={(nqk, ntk)}"
        assert query_trigger.is_contiguous(), f"query_trigger is not contiguous, query_trigger.size()={query_trigger.size()}, query_trigger.stride()={query_trigger.stride()}"
    
    if isinstance(key_trigger, Tensor):
        assert key_trigger.size() == (nqk, ntk), f"key_trigger.size()={key_trigger.size()} != (nqk, ntk)={(nqk, ntk)}"
        assert key_trigger.is_contiguous(), f"key_trigger is not contiguous, key_trigger.size()={key_trigger.size()}, key_trigger.stride()={key_trigger.stride()}"

    output, endpos, length = torch.ops.rosa_soft.rosa_cache_update(cache, batch, query, key, value, query_trigger, key_trigger, 0)
    return output, endpos, length


