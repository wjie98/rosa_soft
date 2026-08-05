"""Exact match indexes plus rejected v1 indexing ablations.

The suffix-array implementation supports the frozen compact solver. Direct,
rolling-hash, and dyadic implementations remain comparison backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from torch import Tensor

from benchmarks.filtered_bitflip import _code_list, _validate_codes


__all__ = [
    "DirectMatchIndex",
    "DyadicMatchIndex",
    "LibsaisSuffixArrayMatchIndex",
    "RollingHashMatchIndex",
    "SuffixArrayMatchIndex",
    "build_match_index",
]


class DirectMatchIndex:
    """No-build exact baseline that compares packed codes directly."""

    exact = True
    name = "direct"

    def __init__(self, query_codes: Tensor, key_codes: Tensor) -> None:
        _validate_codes(query_codes, key_codes)
        self.query = tuple(_code_list(query_codes))
        self.key = tuple(_code_list(key_codes))
        self.logical_bytes = len(self.query) + len(self.key)

    def left_matches(self, query_position: int, key_position: int) -> int:
        length = 0
        while (
            query_position - length - 1 >= 0
            and key_position - length - 1 >= 0
            and self.query[query_position - length - 1]
            == self.key[key_position - length - 1]
        ):
            length += 1
        return length

    def right_matches(self, query_position: int, key_position: int) -> int:
        length = 0
        sequence_length = len(self.query)
        while (
            query_position + length + 1 < sequence_length
            and key_position + length + 1 < sequence_length - 1
            and self.query[query_position + length + 1]
            == self.key[key_position + length + 1]
        ):
            length += 1
        return length


class _RollingHashPair:
    _MODULUS = (1 << 61) - 1
    _BASE = 1_000_000_007

    def __init__(self, query: Sequence[int], key: Sequence[int]) -> None:
        sequence_length = len(query)
        powers = [1] * (sequence_length + 1)
        for index in range(sequence_length):
            powers[index + 1] = (
                powers[index] * self._BASE
            ) % self._MODULUS
        self.powers = tuple(powers)
        self.query_prefix = self._prefix(query)
        self.key_prefix = self._prefix(key)

    def _prefix(self, values: Sequence[int]) -> tuple[int, ...]:
        prefix = [0]
        for value in values:
            prefix.append(
                (prefix[-1] * self._BASE + int(value) + 1) % self._MODULUS
            )
        return tuple(prefix)

    def _hash(self, prefix: Sequence[int], start: int, length: int) -> int:
        return (
            prefix[start + length] - prefix[start] * self.powers[length]
        ) % self._MODULUS

    def lce(self, query_position: int, key_position: int, limit: int) -> int:
        low = 0
        high = max(int(limit), 0)
        while low < high:
            middle = (low + high + 1) // 2
            if self._hash(self.query_prefix, query_position, middle) == self._hash(
                self.key_prefix,
                key_position,
                middle,
            ):
                low = middle
            else:
                high = middle - 1
        return low

    @property
    def logical_bytes(self) -> int:
        return 8 * (
            len(self.powers) + len(self.query_prefix) + len(self.key_prefix)
        )


class RollingHashMatchIndex:
    """O(T)-space probabilistic sieve; it cannot make final exact decisions."""

    exact = False
    name = "rolling_hash"

    def __init__(self, query_codes: Tensor, key_codes: Tensor) -> None:
        _validate_codes(query_codes, key_codes)
        query = tuple(_code_list(query_codes))
        key = tuple(_code_list(key_codes))
        self.sequence_length = len(query)
        self.forward = _RollingHashPair(query, key)
        self.reverse = _RollingHashPair(query[::-1], key[::-1])
        self.logical_bytes = (
            self.forward.logical_bytes + self.reverse.logical_bytes
        )

    def left_matches(self, query_position: int, key_position: int) -> int:
        limit = min(query_position, key_position)
        return self.reverse.lce(
            self.sequence_length - query_position,
            self.sequence_length - key_position,
            limit,
        )

    def right_matches(self, query_position: int, key_position: int) -> int:
        limit = min(
            self.sequence_length - query_position - 1,
            self.sequence_length - key_position - 2,
        )
        return self.forward.lce(
            query_position + 1,
            key_position + 1,
            limit,
        )


class _DyadicPair:
    def __init__(self, query: Sequence[int], key: Sequence[int]) -> None:
        query_levels: list[tuple[int, ...]] = [tuple(int(x) for x in query)]
        key_levels: list[tuple[int, ...]] = [tuple(int(x) for x in key)]
        span = 2
        while span <= len(query):
            half = span // 2
            previous_query = query_levels[-1]
            previous_key = key_levels[-1]
            rank_by_pair: dict[tuple[int, int], int] = {}

            def rank(values: tuple[int, ...]) -> tuple[int, ...]:
                result = []
                for start in range(len(query) - span + 1):
                    pair = (values[start], values[start + half])
                    result.append(
                        rank_by_pair.setdefault(pair, len(rank_by_pair))
                    )
                return tuple(result)

            query_levels.append(rank(previous_query))
            key_levels.append(rank(previous_key))
            span *= 2
        self.query_levels = tuple(query_levels)
        self.key_levels = tuple(key_levels)

    def lce(self, query_position: int, key_position: int, limit: int) -> int:
        matched = 0
        for level in range(len(self.query_levels) - 1, -1, -1):
            span = 1 << level
            if span > limit - matched:
                continue
            if (
                self.query_levels[level][query_position + matched]
                == self.key_levels[level][key_position + matched]
            ):
                matched += span
        return matched

    @property
    def logical_bytes(self) -> int:
        entries = sum(len(level) for level in self.query_levels)
        entries += sum(len(level) for level in self.key_levels)
        return entries * 4


class DyadicMatchIndex:
    """Exact canonical power-of-two substring ranks."""

    exact = True
    name = "dyadic"

    def __init__(self, query_codes: Tensor, key_codes: Tensor) -> None:
        _validate_codes(query_codes, key_codes)
        query = tuple(_code_list(query_codes))
        key = tuple(_code_list(key_codes))
        self.sequence_length = len(query)
        self.forward = _DyadicPair(query, key)
        self.reverse = _DyadicPair(query[::-1], key[::-1])
        self.logical_bytes = (
            self.forward.logical_bytes + self.reverse.logical_bytes
        )

    def left_matches(self, query_position: int, key_position: int) -> int:
        limit = min(query_position, key_position)
        return self.reverse.lce(
            self.sequence_length - query_position,
            self.sequence_length - key_position,
            limit,
        )

    def right_matches(self, query_position: int, key_position: int) -> int:
        limit = min(
            self.sequence_length - query_position - 1,
            self.sequence_length - key_position - 2,
        )
        return self.forward.lce(
            query_position + 1,
            key_position + 1,
            limit,
        )


class _RangeMinimumTree:
    def __init__(self, values: Sequence[int]) -> None:
        size = 1
        while size < len(values):
            size *= 2
        infinity = len(values) + 1
        tree = [infinity] * (2 * size)
        tree[size : size + len(values)] = values
        for node in range(size - 1, 0, -1):
            tree[node] = min(tree[2 * node], tree[2 * node + 1])
        self.size = size
        self.tree = tuple(tree)

    def minimum(self, start: int, stop: int) -> int:
        if start >= stop:
            raise ValueError("RMQ range must be nonempty")
        start += self.size
        stop += self.size
        result = self.tree[0]
        while start < stop:
            if start & 1:
                result = min(result, self.tree[start])
                start += 1
            if stop & 1:
                stop -= 1
                result = min(result, self.tree[stop])
            start //= 2
            stop //= 2
        return result

    @property
    def logical_bytes(self) -> int:
        return len(self.tree) * 4


class _SuffixArrayPair:
    def __init__(
        self,
        query: Sequence[int],
        key: Sequence[int],
        *,
        backend: str = "python",
        library_path: Path | str | None = None,
    ) -> None:
        self.query_length = len(query)
        text = tuple(int(value) + 2 for value in query) + (0,)
        text += tuple(int(value) + 2 for value in key) + (1,)
        if backend == "python":
            suffix_array = self._build_suffix_array(text)
            native_lcp = None
        elif backend == "libsais":
            from benchmarks.filtered_bitflip_native import LibsaisBackend

            suffix_array, native_lcp = LibsaisBackend(
                library_path
            ).suffix_array_lcp(text)
        else:
            raise ValueError(f"unknown suffix-array backend: {backend}")
        inverse = [0] * len(text)
        for rank, position in enumerate(suffix_array):
            inverse[position] = rank
        lcp = (
            native_lcp
            if native_lcp is not None
            else self._build_lcp(text, suffix_array, inverse)
        )
        self.text_length = len(text)
        self.suffix_array = suffix_array
        self.inverse = tuple(inverse)
        self.lcp = lcp
        self.rmq = _RangeMinimumTree(lcp)

    @staticmethod
    def _build_suffix_array(text: Sequence[int]) -> tuple[int, ...]:
        size = len(text)
        suffix_array = list(range(size))
        symbols = {value: rank for rank, value in enumerate(sorted(set(text)))}
        ranks = [symbols[value] for value in text]
        width = 1
        while width < size:
            suffix_array.sort(
                key=lambda position: (
                    ranks[position],
                    ranks[position + width] if position + width < size else -1,
                )
            )
            next_ranks = [0] * size
            for index in range(1, size):
                previous = suffix_array[index - 1]
                current = suffix_array[index]
                previous_key = (
                    ranks[previous],
                    ranks[previous + width] if previous + width < size else -1,
                )
                current_key = (
                    ranks[current],
                    ranks[current + width] if current + width < size else -1,
                )
                next_ranks[current] = next_ranks[previous] + int(
                    current_key != previous_key
                )
            ranks = next_ranks
            if ranks[suffix_array[-1]] == size - 1:
                break
            width *= 2
        return tuple(suffix_array)

    @staticmethod
    def _build_lcp(
        text: Sequence[int],
        suffix_array: Sequence[int],
        inverse: Sequence[int],
    ) -> tuple[int, ...]:
        lcp = [0] * len(text)
        height = 0
        for position in range(len(text)):
            rank = inverse[position]
            if rank == 0:
                continue
            previous = suffix_array[rank - 1]
            while (
                position + height < len(text)
                and previous + height < len(text)
                and text[position + height] == text[previous + height]
            ):
                height += 1
            lcp[rank] = height
            if height:
                height -= 1
        return tuple(lcp)

    def lce(self, query_position: int, key_position: int, limit: int) -> int:
        if limit <= 0:
            return 0
        query_rank = self.inverse[query_position]
        key_rank = self.inverse[self.query_length + 1 + key_position]
        if query_rank == key_rank:
            return limit
        start = min(query_rank, key_rank) + 1
        stop = max(query_rank, key_rank) + 1
        return min(self.rmq.minimum(start, stop), limit)

    @property
    def logical_bytes(self) -> int:
        return (
            self.text_length
            + 4
            * (
                len(self.suffix_array)
                + len(self.inverse)
                + len(self.lcp)
            )
            + self.rmq.logical_bytes
        )


class SuffixArrayMatchIndex:
    """Exact generalized suffix-array, LCP, and flat-RMQ backend."""

    exact = True
    name = "suffix_array"

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        backend: str = "python",
        library_path: Path | str | None = None,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        query = tuple(_code_list(query_codes))
        key = tuple(_code_list(key_codes))
        self.sequence_length = len(query)
        self.forward = _SuffixArrayPair(
            query,
            key[:-1],
            backend=backend,
            library_path=library_path,
        )
        self.reverse = _SuffixArrayPair(
            query[::-1],
            key[::-1],
            backend=backend,
            library_path=library_path,
        )
        self.logical_bytes = (
            self.forward.logical_bytes + self.reverse.logical_bytes
        )

    def left_matches(self, query_position: int, key_position: int) -> int:
        limit = min(query_position, key_position)
        return self.reverse.lce(
            self.sequence_length - query_position,
            self.sequence_length - key_position,
            limit,
        )

    def right_matches(self, query_position: int, key_position: int) -> int:
        limit = min(
            self.sequence_length - query_position - 1,
            self.sequence_length - key_position - 2,
        )
        return self.forward.lce(
            query_position + 1,
            key_position + 1,
            limit,
        )


class LibsaisSuffixArrayMatchIndex(SuffixArrayMatchIndex):
    """Exact suffix-array index built by the optional native libsais backend."""

    name = "libsais"

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        library_path: Path | str | None = None,
    ) -> None:
        super().__init__(
            query_codes,
            key_codes,
            backend="libsais",
            library_path=library_path,
        )


def build_match_index(name: str, query_codes: Tensor, key_codes: Tensor):
    constructors = {
        "direct": DirectMatchIndex,
        "rolling_hash": RollingHashMatchIndex,
        "dyadic": DyadicMatchIndex,
        "suffix_array": SuffixArrayMatchIndex,
        "libsais": LibsaisSuffixArrayMatchIndex,
    }
    try:
        constructor = constructors[name]
    except KeyError as error:
        raise ValueError(f"unknown match index: {name}") from error
    return constructor(query_codes, key_codes)
