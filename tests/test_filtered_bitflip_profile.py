import torch

from benchmarks.filtered_bitflip import CodeOccurrenceIndex, semantic_bit_flips
from benchmarks.filtered_bitflip_profile import make_training_codes, profile_case


def test_training_trajectory_endpoints_are_reproducible_and_shift_aligned():
    initial_query, initial_key = make_training_codes(32, 4, "shift_random", 0.0, 7)
    repeated_query, repeated_key = make_training_codes(32, 4, "shift_random", 0.0, 7)
    final_query, final_key = make_training_codes(32, 4, "shift_random", 1.0, 7)
    assert torch.equal(initial_query, repeated_query)
    assert torch.equal(initial_key, repeated_key)
    assert torch.equal(final_query[1:], final_key[:-1])
    assert not torch.equal(initial_query[1:], initial_key[:-1])


def test_motif_trajectory_keeps_multiple_codes_when_clear():
    query, key = make_training_codes(64, 4, "shift_motif", 1.0, 11)
    assert torch.equal(query[1:], key[:-1])
    assert torch.unique(key).numel() > 1


def test_profile_case_checks_all_exact_indexes_and_winner_backends():
    result = profile_case(12, 3, "shift_motif", 0.5, 13, 1)
    assert result["raw_local_pair_events"] > 0
    assert result["match_indexes"]["direct"]["exact"]
    assert not result["match_indexes"]["rolling_hash"]["exact"]
    assert result["winner_indexes"]["rle"]["raw_events"] == result[
        "winner_indexes"
    ]["segment"]["raw_events"]


def test_occurrence_filter_remains_selective_when_healthy_patterns_are_clear():
    sequence_length = 64
    bit_width = 8

    def retained_fraction(trajectory: str) -> float:
        query, key = make_training_codes(
            sequence_length,
            bit_width,
            trajectory,
            1.0,
            17,
        )
        occurrence = CodeOccurrenceIndex.build(query, key)
        retained = sum(
            len(occurrence.changed_pairs(flip))
            for flip in semantic_bit_flips(sequence_length, bit_width)
        )
        return retained / (bit_width * sequence_length * (sequence_length - 1))

    assert retained_fraction("shift_random") < 0.05
    assert retained_fraction("shift_motif") < 0.15
    assert retained_fraction("collapse") == 1.0
