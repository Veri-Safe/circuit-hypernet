from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DatasetSplitSpec:
    name: str
    path_suffix: str
    fingerprint: str
    rows: int
    puzzles: int


@dataclass(frozen=True)
class ExperimentProfile:
    profile_id: str
    format_version: str
    source_kind: str
    source_step: int
    source_run_name: str | None
    source_run_kind: str | None
    source_data_batches_sha256: str | None
    source_data_permutation_sha256: str | None
    train_splits: tuple[DatasetSplitSpec, ...]
    train_rows: int
    train_puzzles: int
    dev_path_suffix: str
    dev_fingerprint: str
    dev_rows: int
    dev_puzzles: int
    max_steps: int
    milestone_steps: tuple[int, ...]
    protocol_kind: str
    pair_count_distribution: tuple[tuple[int, int], ...] = ()
    expected_event_count: int | None = None
    source_run_max_steps: int | None = None
    source_run_max_epochs: int = 1
    data_seed: int = 24
    micro_row_batch: int = 8
    global_row_batch: int = 256
    train_epochs: int = 1
    rows_per_logical_epoch: int | None = None
    schedule_sha256: str | None = None
    hf_artifact_sha256: tuple[tuple[str, str], ...] = ()
    hf_aggregate_sha256: str | None = None
    tokenizer_config_sha256: str | None = None
    token_mapping_sha256: str | None = None

    @property
    def expected_source_run_max_steps(self) -> int:
        return (
            self.source_step
            if self.source_run_max_steps is None
            else self.source_run_max_steps
        )

    @property
    def expected_source_run_max_epochs(self) -> int:
        return self.source_run_max_epochs


K8_PROFILE = ExperimentProfile(
    profile_id="k8_fullft_step1400",
    format_version="nvarc_pg_k8_v1",
    source_kind="nvarc_full_ft_k8",
    source_step=1400,
    source_run_name=None,
    source_run_kind=None,
    source_data_batches_sha256=None,
    source_data_permutation_sha256=None,
    train_splits=(
        DatasetSplitSpec(
            name="train_k8",
            path_suffix="/train_k8",
            fingerprint="ba68e984e6fadf39",
            rows=378_549,
            puzzles=47_337,
        ),
    ),
    train_rows=378_549,
    train_puzzles=47_337,
    dev_path_suffix="/view_holdout",
    dev_fingerprint="618bf230bccba9f2",
    dev_rows=47_137,
    dev_puzzles=47_137,
    max_steps=1_478,
    milestone_steps=(0, 148, 370, 739, 1109, 1478),
    protocol_kind="arc_agi2_internal",
    micro_row_batch=1,
)


ARC1_CLEAN_PROFILE = ExperimentProfile(
    profile_id="arc1_clean_fullft",
    format_version="nvarc_pg_arc1_clean_fullft_v1",
    source_kind="nvarc_arc1_clean_fullft",
    source_step=706,
    source_run_name="sha256:729b56ba3809f60c0d438da3fe7c6f77461fe713ab8d656376d40aeca4e5a937",
    source_run_kind="arc1_clean_core",
    source_data_batches_sha256=(
        "bbec3409c4773f72c015a9a1bbdad63755805add96bb67c0741eebc29f312d3c"
    ),
    source_data_permutation_sha256=(
        "22216bfc81a67f83170c10182f2d872f37a38c6eab58d1b5484d14dda26cee92"
    ),
    train_splits=(
        DatasetSplitSpec(
            "rearc", "/grids_v15/rearc", "332b5af0e60bee1d", 102_392, 400
        ),
        DatasetSplitSpec(
            "mini", "/grids_v15/mini", "66acf25a0463ab59", 37_632, 147
        ),
        DatasetSplitSpec(
            "concept", "/grids_v15/concept", "2e9f5fdec5842bb6", 40_960, 160
        ),
    ),
    train_rows=180_984,
    train_puzzles=707,
    dev_path_suffix="/arc1_clean_sft_v1/arc1_safe_validation",
    dev_fingerprint="9b749f51ea8b233d",
    dev_rows=684,
    dev_puzzles=114,
    max_steps=706,
    milestone_steps=(0, 71, 177, 353, 530, 706),
    protocol_kind="arc_agi1_internal",
    pair_count_distribution=(
        (3, 1_280),
        (4, 20_992),
        (5, 30_976),
        (6, 69_805),
        (7, 57_163),
        (8, 512),
        (9, 256),
    ),
    expected_event_count=1_068_059,
)


# Keep the original step-706 profile immutable for old checkpoint provenance.
# The selected step-353 base comes from the same completed FullFT trajectory and
# therefore changes only source identity, not the PG data/training contract.
ARC1_CLEAN_STEP353_PROFILE = replace(
    ARC1_CLEAN_PROFILE,
    profile_id="arc1_clean_fullft_step353",
    format_version="nvarc_pg_arc1_clean_fullft_step353_v1",
    source_step=353,
    source_run_max_steps=706,
)


ARC1_REARC_PROFILE = ExperimentProfile(
    profile_id="arc1_rearc_fullft",
    format_version="nvarc_pg_arc1_rearc_fullft_v1",
    source_kind="nvarc_arc1_rearc_fullft",
    source_step=399,
    source_run_name="sha256:88d0ca794cb2daf4fdfcf6e339adf413a1910049c2c861a1db6f507898ee141c",
    source_run_kind="arc1_rearc",
    source_data_batches_sha256=(
        "57a918c44191a315e4c3182d1cce9c785c516c92a6e5afc68fb2bb062e4c1d73"
    ),
    source_data_permutation_sha256=(
        "e34454cf4740b3488b575bc5f9a93effad77ffc8db6a60c95ca4f31ba04096dd"
    ),
    train_splits=(
        DatasetSplitSpec(
            "rearc", "/grids_v15/rearc", "332b5af0e60bee1d", 102_392, 400
        ),
    ),
    train_rows=102_392,
    train_puzzles=400,
    dev_path_suffix="/arc1_clean_sft_v1/arc1_safe_validation",
    dev_fingerprint="9b749f51ea8b233d",
    dev_rows=684,
    dev_puzzles=114,
    max_steps=399,
    milestone_steps=(0, 40, 100, 200, 299, 399),
    protocol_kind="arc_agi1_internal",
    pair_count_distribution=((6, 51_373), (7, 51_019)),
    expected_event_count=665_371,
)


ARC1_REARC128_S42_STEP1400_PROFILE = ExperimentProfile(
    profile_id="arc1_rearc128_s42_step1400",
    format_version="nvarc_pg_arc1_rearc128_s42_step1400_v1",
    source_kind="nvarc_arc1_rearc128_fullft",
    source_step=1_400,
    source_run_name="sha256:8bdca18624269a41c29e1da2d1fbcda3f546941aa084a31f4816a4cda58faf56",
    source_run_kind="arc1_rearc128_long_20ep",
    source_data_batches_sha256=(
        "0df55d8b2c2df4b2e73bcc373170972ee36980427944f90fd8b80da66b3cbd0f"
    ),
    source_data_permutation_sha256=(
        "f458b8d47f92e494349459b02207d6b0ca635a826f4decdc8be8abd19cf18454"
    ),
    train_splits=(
        DatasetSplitSpec(
            "rearc128",
            "/rearc128_s42/hf/rearc128",
            "1351eb1f7ee044e3",
            51_200,
            400,
        ),
    ),
    train_rows=51_200,
    train_puzzles=400,
    dev_path_suffix="",
    dev_fingerprint="",
    dev_rows=0,
    dev_puzzles=0,
    max_steps=1_200,
    milestone_steps=(50, 100, 150, 200, 300, 400, 600, 800, 1_000, 1_200),
    protocol_kind="arc_agi1_public_validation",
    pair_count_distribution=((6, 25_705), (7, 25_495)),
    expected_event_count=332_695,
    source_run_max_steps=4_000,
    source_run_max_epochs=20,
    data_seed=42,
    micro_row_batch=32,
    train_epochs=6,
)


ARC1_REARC128_S42_STEP1400_OUTSIDE_UNION_PROFILE = ExperimentProfile(
    profile_id="arc1_rearc128_s42_step1400_outside_union",
    format_version="nvarc_pg_arc1_rearc128_s42_step1400_outside_union_v1",
    source_kind="nvarc_arc1_rearc128_fullft",
    source_step=1_400,
    source_run_name=None,
    source_run_kind=None,
    source_data_batches_sha256=None,
    source_data_permutation_sha256=None,
    train_splits=(
        DatasetSplitSpec(
            "rearc128_outside_b0_b1_union",
            "/rearc128_outside_delta4_b0_b1_union_v1/hf/train",
            "a60c7086fd33ad31",
            3_200,
            400,
        ),
    ),
    train_rows=3_200,
    train_puzzles=400,
    dev_path_suffix="",
    dev_fingerprint="",
    dev_rows=0,
    dev_puzzles=0,
    max_steps=650,
    milestone_steps=(65, 130, 195, 260, 325, 390, 455, 520, 585, 650),
    protocol_kind="arc_agi1_public_validation",
    pair_count_distribution=((6, 1_590), (7, 1_610)),
    expected_event_count=20_810,
    source_run_max_steps=4_000,
    source_run_max_epochs=20,
    data_seed=24,
    micro_row_batch=16,
    global_row_batch=256,
    train_epochs=50,
    rows_per_logical_epoch=3_328,
    schedule_sha256=(
        "39ddc99de6f2fad628b50fad3065ee827243ff683c72db00f9cc197cc28ca9d7"
    ),
    hf_artifact_sha256=(
        (
            "config.json",
            "8f9eb61dafc23ada43f53cf929bd40b9943bab7befb866840b05fa7107a18aaf",
        ),
        (
            "generation_config.json",
            "976b81a522907a4ee2055be7719e194e65eb290de8a9874aa34fcee1cb99f380",
        ),
        (
            "model-00001-of-00003.safetensors",
            "9f7d68f93f82fd1e76577727be5c0c6ef4eb65203a92de946bf8684f1159c579",
        ),
        (
            "model-00002-of-00003.safetensors",
            "708070c338d1871551151710e12364e60ec74d461496ff220edc150fb9193854",
        ),
        (
            "model-00003-of-00003.safetensors",
            "6724460b9610da32322091cb224670e3bdff64aab008dae097947c5b851111b7",
        ),
        (
            "model.safetensors.index.json",
            "3ab7be78f7ebc7a75d88fb9c3c4af52b21462718a23c0c726ca0c2b48a3e93d6",
        ),
    ),
    hf_aggregate_sha256=(
        "60d6f0e1eecd8fb0a4cd4835142006c3ac040732c89e2b28d5918683caeb50c7"
    ),
    tokenizer_config_sha256=(
        "d59ac739ba44b2da01725336896a608daa107880811a086e015cf2843ae4d83d"
    ),
    token_mapping_sha256=(
        "e1bb75693da1236d6b5b240b3fce6e03a7975c80812d54c1d3e7e30b634a4884"
    ),
)


def _outside_scale_profile(
    *,
    rows_per_puzzle: int,
    fingerprint: str,
    event_count: int,
    pair_count_distribution: tuple[tuple[int, int], ...],
) -> ExperimentProfile:
    label = f"rearc{rows_per_puzzle}"
    return replace(
        ARC1_REARC128_S42_STEP1400_OUTSIDE_UNION_PROFILE,
        profile_id=f"arc1_rearc128_s42_step1400_outside_{label}",
        format_version=f"nvarc_pg_arc1_rearc128_s42_step1400_outside_{label}_v1",
        train_splits=(
            DatasetSplitSpec(
                f"rearc128_outside_{label}",
                f"/rearc128_outside_scale_s42_v1/hf/{label}",
                fingerprint,
                rows_per_puzzle * 400,
                400,
            ),
        ),
        train_rows=rows_per_puzzle * 400,
        train_puzzles=400,
        pair_count_distribution=pair_count_distribution,
        expected_event_count=event_count,
        train_epochs=1,
        rows_per_logical_epoch=None,
        schedule_sha256=None,
    )


ARC1_REARC128_S42_STEP1400_OUTSIDE_SCALE_PROFILES = (
    _outside_scale_profile(
        rows_per_puzzle=16,
        fingerprint="5162e3224a05800e",
        event_count=41_595,
        pair_count_distribution=((6, 3_205), (7, 3_195)),
    ),
    _outside_scale_profile(
        rows_per_puzzle=24,
        fingerprint="855a88555765d329",
        event_count=62_399,
        pair_count_distribution=((6, 4_801), (7, 4_799)),
    ),
    _outside_scale_profile(
        rows_per_puzzle=32,
        fingerprint="9ccf2f358d5473b7",
        event_count=83_207,
        pair_count_distribution=((6, 6_393), (7, 6_407)),
    ),
    _outside_scale_profile(
        rows_per_puzzle=64,
        fingerprint="2fed63244854041c",
        event_count=166_354,
        pair_count_distribution=((6, 12_846), (7, 12_754)),
    ),
)


_E11_TUNE_MILESTONES = (75, 100, 130, 160, 195, 260)
_OUTSIDE_REARC32_SCALE_PROFILE = next(
    profile
    for profile in ARC1_REARC128_S42_STEP1400_OUTSIDE_SCALE_PROFILES
    if profile.train_rows == 12_800
)


def _outside_rearc32_e11_tune_profile(horizon: int) -> ExperimentProfile:
    """Profile the from-scratch E11 search without changing its data/base identity."""

    return replace(
        _OUTSIDE_REARC32_SCALE_PROFILE,
        profile_id=(
            f"arc1_rearc128_s42_step1400_outside_rearc32_e11tune_h{horizon}"
        ),
        format_version=(
            "nvarc_pg_arc1_rearc128_s42_step1400_outside_rearc32_"
            f"e11tune_h{horizon}_v1"
        ),
        max_steps=horizon,
        milestone_steps=_E11_TUNE_MILESTONES,
    )


ARC1_REARC128_S42_STEP1400_OUTSIDE_E11_TUNE_PROFILES = tuple(
    _outside_rearc32_e11_tune_profile(horizon) for horizon in (260, 325, 650)
)


ARC1_REARC8_S42_STEP130_PROFILE = ExperimentProfile(
    profile_id="arc1_rearc8_s42_step130",
    format_version="nvarc_pg_arc1_rearc8_s42_step130_v1",
    source_kind="nvarc_rearc8_s42_fullft",
    source_step=130,
    source_run_name=None,
    source_run_kind=None,
    source_data_batches_sha256=None,
    source_data_permutation_sha256=None,
    train_splits=(
        DatasetSplitSpec(
            "rearc8_pg_s24_excl_base_s42_v1",
            "/rearc8_pg_s24_excl_base_s42_v1/hf/train",
            "e464cacafff91296",
            3_200,
            400,
        ),
    ),
    train_rows=3_200,
    train_puzzles=400,
    dev_path_suffix="",
    dev_fingerprint="",
    dev_rows=0,
    dev_puzzles=0,
    max_steps=650,
    milestone_steps=(130, 260, 390, 520, 650),
    protocol_kind="arc_agi1_public_validation",
    pair_count_distribution=((6, 1_574), (7, 1_626)),
    expected_event_count=20_826,
    source_run_max_steps=650,
    source_run_max_epochs=50,
    data_seed=42,
    micro_row_batch=32,
    train_epochs=50,
    rows_per_logical_epoch=3_328,
    schedule_sha256=(
        "2a98e42bb28f5be78e6c48d852d461d84b2ad4f3980100043ac3c2582f650868"
    ),
    hf_artifact_sha256=(
        (
            "config.json",
            "8f9eb61dafc23ada43f53cf929bd40b9943bab7befb866840b05fa7107a18aaf",
        ),
        (
            "generation_config.json",
            "976b81a522907a4ee2055be7719e194e65eb290de8a9874aa34fcee1cb99f380",
        ),
        (
            "model-00001-of-00003.safetensors",
            "ec4e516185f04291800e78ad48b5f6973fd142fe8c5a62f925f4b3f3ee242b38",
        ),
        (
            "model-00002-of-00003.safetensors",
            "e190e8f6e1f0c2834bf309aa1aca89a1ff761981e9581ba487014f2acb724410",
        ),
        (
            "model-00003-of-00003.safetensors",
            "ccf50e3f18e1edd5de17f7761be4ce5c6b3ae7174860e72b1566f7944338e4a5",
        ),
        (
            "model.safetensors.index.json",
            "ef4b6936638b4ed6e7bb9d8e9cdc2ab9c87d719b1759c626dad8c2f739f6f339",
        ),
    ),
    hf_aggregate_sha256=(
        "b74bcb895fe162511ab8d047648a9d3cbba110fead5f3d58b9a24e9456d0084c"
    ),
    tokenizer_config_sha256=(
        "d59ac739ba44b2da01725336896a608daa107880811a086e015cf2843ae4d83d"
    ),
    token_mapping_sha256=(
        "e1bb75693da1236d6b5b240b3fce6e03a7975c80812d54c1d3e7e30b634a4884"
    ),
)


ARC1_REARC64_S24_STEP2800_DELTA4_B1_PROFILE = ExperimentProfile(
    profile_id="arc1_rearc64_s24_step2800_delta4_b1",
    format_version="nvarc_pg_arc1_rearc64_s24_step2800_delta4_b1_v1",
    source_kind="nvarc_rearc64_s24_fullft",
    source_step=2_800,
    source_run_name=None,
    source_run_kind=None,
    source_data_batches_sha256=None,
    source_data_permutation_sha256=None,
    train_splits=(
        DatasetSplitSpec(
            "rearc128_minus64_delta4_b1",
            "/rearc128_minus64_delta4_s42/hf/rearc128_minus64_delta4_b1",
            "b40f972f075a2717",
            1_600,
            400,
        ),
    ),
    train_rows=1_600,
    train_puzzles=400,
    dev_path_suffix="",
    dev_fingerprint="",
    dev_rows=0,
    dev_puzzles=0,
    max_steps=1_250,
    milestone_steps=(50, 100, 150, 200, 300, 400, 600, 800, 1_000, 1_250),
    protocol_kind="arc_agi1_public_validation",
    pair_count_distribution=((6, 806), (7, 794)),
    expected_event_count=10_394,
    source_run_max_steps=5_000,
    source_run_max_epochs=50,
    data_seed=24,
    micro_row_batch=4,
    global_row_batch=64,
    train_epochs=50,
    hf_artifact_sha256=(
        (
            "config.json",
            "8f9eb61dafc23ada43f53cf929bd40b9943bab7befb866840b05fa7107a18aaf",
        ),
        (
            "generation_config.json",
            "976b81a522907a4ee2055be7719e194e65eb290de8a9874aa34fcee1cb99f380",
        ),
        (
            "model-00001-of-00003.safetensors",
            "3d9f7f9c3067041b31369416355f821fc8a4eb407f6bf7685e21e095d5d75b25",
        ),
        (
            "model-00002-of-00003.safetensors",
            "3a155e119e70eae4d8707c78e616ca515feab6a7ac6e1a4b5318649f0b58b631",
        ),
        (
            "model-00003-of-00003.safetensors",
            "b028152af618d7b60bb7c31c2d965b16e3c1e650418398e68f70bf4b920faf5e",
        ),
        (
            "model.safetensors.index.json",
            "40899af0faf91e1fae40a5298c75135f93f3aac4cfbb36cc9420ad33d2b3f3bc",
        ),
    ),
    hf_aggregate_sha256=(
        "bd6c9d8aec2fde3c0099e63baba885e4f94d38c217ade4ef23363cfc36531d14"
    ),
    tokenizer_config_sha256=(
        "d59ac739ba44b2da01725336896a608daa107880811a086e015cf2843ae4d83d"
    ),
    token_mapping_sha256=(
        "e1bb75693da1236d6b5b240b3fce6e03a7975c80812d54c1d3e7e30b634a4884"
    ),
)


PROFILES = {
    profile.format_version: profile
    for profile in (
        K8_PROFILE,
        ARC1_CLEAN_PROFILE,
        ARC1_CLEAN_STEP353_PROFILE,
        ARC1_REARC_PROFILE,
        ARC1_REARC128_S42_STEP1400_PROFILE,
        ARC1_REARC128_S42_STEP1400_OUTSIDE_UNION_PROFILE,
        *ARC1_REARC128_S42_STEP1400_OUTSIDE_SCALE_PROFILES,
        *ARC1_REARC128_S42_STEP1400_OUTSIDE_E11_TUNE_PROFILES,
        ARC1_REARC8_S42_STEP130_PROFILE,
        ARC1_REARC64_S24_STEP2800_DELTA4_B1_PROFILE,
    )
}
PROFILES_BY_ID = {profile.profile_id: profile for profile in PROFILES.values()}


def profile_for_format(format_version: str) -> ExperimentProfile:
    try:
        return PROFILES[format_version]
    except KeyError as error:
        raise ValueError(f"Unknown NVARC PG format_version: {format_version!r}") from error


def profile_for_id(profile_id: str) -> ExperimentProfile:
    try:
        return PROFILES_BY_ID[profile_id]
    except KeyError as error:
        raise ValueError(f"Unknown NVARC PG profile_id: {profile_id!r}") from error
