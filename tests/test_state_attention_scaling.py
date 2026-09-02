from benchmarks.state_attention_scaling import _matrix_configs, build_parser


def test_scaling_matrix_deduplicates_the_base_configuration():
    args = build_parser().parse_args(
        [
            "--association-values",
            "2",
            "4",
            "--head-values",
            "1",
            "2",
            "--depth-values",
            "1",
            "2",
        ]
    )

    configs = _matrix_configs(args)
    keys = {
        (row["associations"], row["heads"], row["context_depth"])
        for row in configs
    }

    assert len(configs) == len(keys) == 4
    base = next(
        row
        for row in configs
        if (row["associations"], row["heads"], row["context_depth"])
        == (4, 2, 1)
    )
    assert base["axes"] == ["associations", "depth", "heads"]
