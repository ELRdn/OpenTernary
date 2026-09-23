"""Frozen quality-dataset construction tests."""

from openternary.benchmark.quality_dataset import build_quality_dataset


def _rows(prefix: str, language: str, count: int = 8) -> list[dict[str, object]]:
    topics = [
        "astronomy observes distant galaxies through spectroscopic instruments and orbital telescopes",
        "botany studies vascular plants roots leaves flowers pollination seeds and forest ecology",
        "architecture combines structural engineering materials spatial planning and cultural history",
        "oceanography measures currents salinity marine ecosystems seafloor geology and climate exchange",
        "music theory describes harmony rhythm melody counterpoint orchestration and acoustic perception",
        "archaeology interprets pottery settlements inscriptions trade routes and ancient burial practices",
        "mathematics develops algebra geometry probability topology analysis and rigorous logical proofs",
        "meteorology analyzes pressure fronts clouds precipitation wind circulation and seasonal forecasts",
    ]
    return [
        {
            "id": f"{prefix}-{index}",
            "context": f"{language} context {index}. {topics[index]} " * 3,
            "question": f"Question {index}?",
            "answers": {"text": [f"answer-{index}"], "answer_start": [0]},
        }
        for index in range(count)
    ]


def test_quality_dataset_is_deterministic_and_disjoint() -> None:
    kwargs = {
        "calibration_texts": ["calibration source one", "calibration source two"],
        "squad_rows": _rows("en", "English"),
        "jsquad_rows": _rows("ja", "日本語"),
        "source_revisions": {"wikitext": "w", "squad": "s", "jglue": "j"},
        "contexts_per_language_per_split": 2,
        "instructions_per_language_per_split": 1,
    }
    first = build_quality_dataset(**kwargs)
    second = build_quality_dataset(
        **{
            **kwargs,
            "squad_rows": list(reversed(kwargs["squad_rows"])),
            "jsquad_rows": list(reversed(kwargs["jsquad_rows"])),
        }
    )
    assert first == second
    assert len(first["splits"]["validation"]) == 4
    assert len(first["splits"]["test"]) == 4
    assert len(first["instruction_cases"]["validation"]) == 2
    assert first["dataset"]["selection"]["ordering"] == "sha256(source record id)"
