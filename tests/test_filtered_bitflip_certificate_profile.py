from benchmarks.filtered_bitflip_certificate_profile import profile_case


def test_certificate_profile_compares_all_exact_backends():
    result = profile_case(12, 4, "shift_motif", 0.75, 17, 1)
    assert result["shared"]["raw_events"] > 0
    assert result["base_structures"]["certificate_tree"]["candidate_lines"] > 0
    assert result["base_structures"]["suffix_nodes"]["nodes"] > 0
    assert result["execution"]["certificate_periodic"]["event_families"] > 0
    assert result["execution"]["suffix_periodic"]["periodic_jumps"] >= 0
