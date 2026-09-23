"""Exercise the production runner with a tiny local teacher, no downloads."""

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file  # noqa: E402

from openternary.calibration.runner import run_calibration  # noqa: E402
from openternary.config.schema import AppConfig  # noqa: E402


@pytest.fixture
def local_teacher(tmp_path, monkeypatch):
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.up_proj = torch.nn.Linear(4, 2, bias=False)
            block.mlp.down_proj = torch.nn.Linear(2, 4, bias=False)
            self.model.layers = torch.nn.ModuleList([block])

        def forward(self, input_ids, attention_mask=None):
            x = torch.nn.functional.one_hot(input_ids % 4, 4).float()
            block = self.model.layers[0].mlp
            return block.down_proj(block.up_proj(x))

    class Tokenizer:
        def __call__(self, text, max_length=16, **kwargs):
            values = list(hashlib.sha256(text.encode()).digest())[:max_length]
            return {
                "input_ids": torch.tensor([values]),
                "attention_mask": torch.tensor([[1] * 5 + [0] * (len(values) - 5)]),
            }

    torch.manual_seed(4)
    teacher = Teacher()
    snapshot = tmp_path / "teacher"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"gemma4"}')
    (snapshot / "tokenizer.json").write_text("{}")
    save_file(teacher.state_dict(), str(snapshot / "model.safetensors"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: teacher)
    cfg = AppConfig.model_validate(
        {
            "device": "cpu",
            "calibration": {
                "enabled": True,
                "dataset": "synthetic",
                "num_samples": 10,
                "seq_len": 16,
                "steps": 2,
                "checkpoint_interval": 1,
            },
        }
    )
    return snapshot, cfg


def test_completed_checkpoint_contains_every_target(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    checkpoint = torch.load(out / "artifacts/checkpoint/step_00004.pt", weights_only=True)
    assert checkpoint["schema_version"] == 3
    assert len(checkpoint["completed_manifest"]) == 2
    assert all(entry["size"] > 0 and len(entry["sha256"]) == 64 for entry in checkpoint["completed_manifest"])
    report = json.loads((out / "calibration.json").read_text())
    assert report["used_dummy_simulation"] is False


def test_padding_length_does_not_change_calibration(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    run_calibration(cfg, snapshot, tmp_path / "short")
    cfg.calibration.seq_len = 32
    run_calibration(cfg, snapshot, tmp_path / "long")
    short = json.loads((tmp_path / "short/calibration.json").read_text())
    long = json.loads((tmp_path / "long/calibration.json").read_text())
    assert short["loss_history"] == pytest.approx(long["loss_history"], abs=1e-8)
    assert short["heldout_loss_after"] == pytest.approx(long["heldout_loss_after"], abs=1e-8)


def test_loss_setting_is_effective(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    run_calibration(cfg, snapshot, tmp_path / "mse")
    cfg.calibration.loss = "l1"
    run_calibration(cfg, snapshot, tmp_path / "l1")
    mse = json.loads((tmp_path / "mse/calibration.json").read_text())
    l1 = json.loads((tmp_path / "l1/calibration.json").read_text())
    assert l1["loss_history"][0] != pytest.approx(mse["loss_history"][0])


def test_optimizer_setting_is_effective(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.calibration.optimizer = "sgd"
    run_calibration(cfg, snapshot, tmp_path / "sgd")
    ckpt = torch.load(tmp_path / "sgd/artifacts/checkpoint/step_00001.pt", weights_only=True)
    assert ckpt["optimizer_state"]["state"] == {}  # SGD without momentum has no Adam moments.


def test_soft_to_hard_runner_checkpoints_and_materializes_hard_codes(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.calibration.method = "recon-soft-to-hard"
    cfg.calibration.steps = 3
    cfg.calibration.temperature_schedule = "cosine"
    cfg.calibration.temperature_start = 1.0
    cfg.calibration.temperature_end = 0.05
    cfg.calibration.hard_fraction = 0.34
    cfg.calibration.zero_logit_bias = 0.1
    out = tmp_path / "soft-to-hard"

    run_calibration(cfg, snapshot, out)

    report = json.loads((out / "calibration.json").read_text())
    assert report["assignment"] == {
        "mode": "soft-to-hard",
        "trainable": False,
        "hardening_source": "frozen-weight-midpoint",
        "hardening_uses_zero_logit_bias": False,
        "temperature_schedule": "cosine",
        "temperature_start": 1.0,
        "temperature_end": 0.05,
        "hard_fraction": 0.34,
        "zero_logit_bias": 0.1,
        "final_state": "hard",
    }
    checkpoint = torch.load(out / "artifacts/checkpoint/step_00006.pt", weights_only=True)
    assert checkpoint["assignment"]["mode"] == "soft-to-hard"
    assert checkpoint["assignment"]["next_temperature"] is None
    assert checkpoint["assignment"]["final_state"] == "hard"
    assert report["code_fingerprint_after"] == report["code_fingerprint_before"]

    fingerprint = report["calibrated_content_fingerprint"]
    run_calibration(cfg, snapshot, out, materialize_only=True)
    reloaded = json.loads((out / "calibration.json").read_text())
    assert reloaded["calibrated_content_fingerprint"] == fingerprint
    assert reloaded["assignment"]["final_state"] == "hard"


def test_soft_to_hard_resume_matches_uninterrupted_hard_snapshot(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.calibration.method = "recon-soft-to-hard"
    cfg.calibration.steps = 3
    cfg.calibration.hard_fraction = 0.34
    out = tmp_path / "soft-resume"
    run_calibration(cfg, snapshot, out)
    expected = json.loads((out / "calibration.json").read_text())

    run_calibration(cfg, snapshot, out, resume_from=out / "artifacts/checkpoint/step_00001.pt")

    actual = json.loads((out / "calibration.json").read_text())
    assert actual["loss_history"] == pytest.approx(expected["loss_history"], abs=1e-9)
    assert actual["calibrated_content_fingerprint"] == expected["calibrated_content_fingerprint"]
    assert actual["assignment"]["final_state"] == "hard"


@pytest.mark.parametrize("checkpoint", ["step_00001.pt", "step_00004.pt"])
def test_resume_matches_uninterrupted_run(local_teacher, tmp_path, checkpoint):
    snapshot, cfg = local_teacher
    cfg.calibration.threshold_enabled = True
    cfg.calibration.method = "recon-threshold"
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    before = json.loads((out / "calibration.json").read_text())
    run_calibration(cfg, snapshot, out, resume_from=out / "artifacts/checkpoint" / checkpoint)
    after = json.loads((out / "calibration.json").read_text())
    assert after["loss_history"] == pytest.approx(before["loss_history"], abs=1e-9)
    assert after["calibrated_content_fingerprint"] == before["calibrated_content_fingerprint"]


def test_resume_rejects_missing_completed_state(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    ckpt_path = out / "artifacts/checkpoint/step_00004.pt"
    checkpoint = torch.load(ckpt_path, weights_only=True)
    checkpoint["completed_manifest"] = checkpoint["completed_manifest"][:1]
    torch.save(checkpoint, ckpt_path)
    with pytest.raises(ValueError, match="completed|manifest|target"):
        run_calibration(cfg, snapshot, out, resume_from=ckpt_path)


@pytest.mark.parametrize("field", ["current_raw_param", "optimizer_state", "rng_state"])
def test_mid_module_resume_rejects_missing_transaction_state(local_teacher, tmp_path, field):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    ckpt_path = out / "artifacts/checkpoint/step_00001.pt"
    checkpoint = torch.load(ckpt_path, weights_only=True)
    checkpoint.pop(field)
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match=field):
        run_calibration(cfg, snapshot, out, resume_from=ckpt_path)


def test_resume_rejects_corrupt_completed_state(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    state = out / "artifacts/calibration_state/model_layers_0_mlp_up_proj.safetensors"
    state.write_bytes(state.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash|size|integrity"):
        run_calibration(cfg, snapshot, out, resume=True)


def test_resume_rejects_missing_activation_cache_shard(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    shard = next((out / "artifacts/activation_cache_train").glob("*/batch_*.pt"))
    shard.unlink()
    with pytest.raises(ValueError, match="cache.*integrity|cache.*missing"):
        run_calibration(cfg, snapshot, out, resume=True)


def test_capture_stops_if_cache_manifest_cannot_be_saved(local_teacher, tmp_path, monkeypatch):
    snapshot, cfg = local_teacher
    original_write_text = type(tmp_path).write_text

    def fail_fingerprint_write(path, *args, **kwargs):
        if path.name == "fingerprint.json":
            raise OSError("injected manifest write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "write_text", fail_fingerprint_write)
    with pytest.raises(ValueError, match="activation cache.*manifest"):
        run_calibration(cfg, snapshot, tmp_path / "run")


def test_resume_rejects_legacy_checkpoint_schema(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    ckpt_path = out / "artifacts/checkpoint/step_00004.pt"
    checkpoint = torch.load(ckpt_path, weights_only=True)
    checkpoint["schema_version"] = 2
    torch.save(checkpoint, ckpt_path)
    with pytest.raises(ValueError, match="schema|conversion"):
        run_calibration(cfg, snapshot, out, resume_from=ckpt_path)


def test_materialize_rejects_missing_checkpoint_contract(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    ckpt_path = out / "artifacts/checkpoint/step_00004.pt"
    checkpoint = torch.load(ckpt_path, weights_only=True)
    checkpoint.pop("resume_contract")
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match="resume contract"):
        run_calibration(cfg, snapshot, out, materialize_only=True)


def test_materialize_rejects_semantically_invalid_state_shape(local_teacher, tmp_path):
    from safetensors.torch import load_file

    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    state_path = out / "artifacts/calibration_state/model_layers_0_mlp_up_proj.safetensors"
    state = load_file(str(state_path))
    save_file({"raw_param": state["raw_param"][:0]}, str(state_path))
    ckpt_path = out / "artifacts/checkpoint/step_00004.pt"
    checkpoint = torch.load(ckpt_path, weights_only=True)
    entry = next(item for item in checkpoint["completed_manifest"] if item["module"].endswith("up_proj"))
    entry["size"] = state_path.stat().st_size
    entry["sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match="shape mismatch"):
        run_calibration(cfg, snapshot, out, materialize_only=True)


def test_materialize_stops_if_report_cannot_be_updated(local_teacher, tmp_path, monkeypatch):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    original_write_text = type(tmp_path).write_text

    def fail_report_write(path, *args, **kwargs):
        if path.name == "calibration.json":
            raise OSError("injected report write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "write_text", fail_report_write)
    with pytest.raises(ValueError, match="materialize-only report"):
        run_calibration(cfg, snapshot, out, materialize_only=True)


def test_cross_directory_resume_copies_completed_state(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    run_calibration(cfg, snapshot, source)

    run_calibration(
        cfg,
        snapshot,
        destination,
        resume_from=source / "artifacts/checkpoint/step_00001.pt",
    )
    expected = json.loads((source / "calibration.json").read_text())
    actual = json.loads((destination / "calibration.json").read_text())
    assert actual["calibrated_content_fingerprint"] == expected["calibrated_content_fingerprint"]
    assert len(list((destination / "artifacts/calibration_state").glob("*.safetensors"))) == 2

    run_calibration(cfg, snapshot, destination, materialize_only=True)


def test_init_from_rejects_missing_checkpoint(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    init_from = tmp_path / "empty-init"
    init_from.mkdir()

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_calibration(cfg, snapshot, tmp_path / "destination", init_from=init_from)


def test_init_from_rejects_corrupt_completed_state(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    source = tmp_path / "source"
    run_calibration(cfg, snapshot, source)
    state = source / "artifacts/calibration_state/model_layers_0_mlp_up_proj.safetensors"
    state.write_bytes(state.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="init-from.*(hash|size|integrity|state)"):
        run_calibration(cfg, snapshot, tmp_path / "destination", init_from=source)


def test_init_from_complete_scale_state_warm_starts_threshold(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    source = tmp_path / "source"
    run_calibration(cfg, snapshot, source)
    threshold_cfg = cfg.model_copy(deep=True)
    threshold_cfg.calibration.method = "recon-threshold"
    threshold_cfg.calibration.threshold_enabled = True
    destination = tmp_path / "destination"

    run_calibration(threshold_cfg, snapshot, destination, init_from=source)

    report = json.loads((destination / "calibration.json").read_text())
    assert report["threshold_enabled"] is True
    assert len(report["reconstruction_by_module"]) == 2


def test_init_from_corrupt_cache_is_not_reused(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    source = tmp_path / "source"
    run_calibration(cfg, snapshot, source)
    source_shard = next((source / "artifacts/activation_cache_train").glob("*/batch_*.pt"))
    source_shard.write_bytes(source_shard.read_bytes() + b"corrupt")
    destination = tmp_path / "destination"

    run_calibration(cfg, snapshot, destination, init_from=source)

    cache_root = destination / "artifacts/activation_cache_train"
    metadata = json.loads((cache_root / "fingerprint.json").read_text())
    for entry in metadata["files"]:
        path = cache_root / entry["file"]
        assert path.stat().st_size == entry["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_init_from_rejects_changed_teacher_source(local_teacher, tmp_path):
    from safetensors.torch import load_file

    snapshot, cfg = local_teacher
    source = tmp_path / "source"
    run_calibration(cfg, snapshot, source)
    weights = load_file(str(snapshot / "model.safetensors"))
    weights[next(iter(weights))].add_(1)
    save_file(weights, str(snapshot / "model.safetensors"))

    with pytest.raises(ValueError, match="init-from source fingerprint mismatch"):
        run_calibration(cfg, snapshot, tmp_path / "destination", init_from=source)


@pytest.mark.parametrize("change", ["loss", "weight"])
@pytest.mark.parametrize("operation", ["resume", "materialize"])
def test_resume_rejects_changed_experiment(local_teacher, tmp_path, change, operation):
    from safetensors.torch import load_file

    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    if change == "loss":
        cfg.calibration.loss = "l1"
    else:
        weights = load_file(str(snapshot / "model.safetensors"))
        weights[next(iter(weights))].add_(1)
        save_file(weights, str(snapshot / "model.safetensors"))
    with pytest.raises(ValueError, match="contract|fingerprint"):
        run_calibration(cfg, snapshot, out, resume=operation == "resume", materialize_only=operation == "materialize")


def test_metrics_compare_same_modules_with_valid_element_weights(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    report = json.loads((out / "calibration.json").read_text())
    modules = report["reconstruction_by_module"]
    assert sorted(m["heldout_elements"] for m in modules.values()) == [20, 40]
    assert sorted(m["train_elements"] for m in modules.values()) == [80, 160]
    assert report["initial_loss"] == pytest.approx(
        sum(m["train_before"] * m["train_elements"] for m in modules.values()) / 240
    )
    assert report["heldout_loss_after"] == pytest.approx(
        sum(m["heldout_after"] * m["heldout_elements"] for m in modules.values()) / 60
    )
    assert report["best_loss"] is None  # module-local minima are not a global model metric.


def test_materialize_only_rejects_incomplete_checkpoint(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    with pytest.raises(ValueError, match="complete"):
        run_calibration(
            cfg, snapshot, out, materialize_only=True, resume_from=out / "artifacts/checkpoint/step_00001.pt"
        )


def test_disabled_targets_are_not_calibrated(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.quantization.target.mlp = False
    with pytest.raises(ValueError, match="target|quantizable"):
        run_calibration(cfg, snapshot, tmp_path / "run")


def test_optimizer_failure_can_resume_last_durable_checkpoint(local_teacher, tmp_path, monkeypatch):
    snapshot, cfg = local_teacher
    baseline = tmp_path / "baseline"
    run_calibration(cfg, snapshot, baseline)
    original_step = torch.optim.Adam.step
    completed_steps = 0

    def interrupted_step(optimizer, *args, **kwargs):
        nonlocal completed_steps
        completed_steps += 1
        if completed_steps == 2:
            raise torch.OutOfMemoryError("injected allocation failure")
        return original_step(optimizer, *args, **kwargs)

    out = tmp_path / "interrupted"
    monkeypatch.setattr(torch.optim.Adam, "step", interrupted_step)
    with pytest.raises(torch.OutOfMemoryError):
        run_calibration(cfg, snapshot, out)
    assert not (out / "calibration.json").exists()
    monkeypatch.setattr(torch.optim.Adam, "step", original_step)
    run_calibration(cfg, snapshot, out, resume=True)
    expected = json.loads((baseline / "calibration.json").read_text())
    actual = json.loads((out / "calibration.json").read_text())
    assert actual["calibrated_content_fingerprint"] == expected["calibrated_content_fingerprint"]


def test_optimizer_oom_rolls_back_and_retries_same_step(local_teacher, tmp_path, monkeypatch):
    """An OOM after a partial optimizer update must be transactionally retried."""
    from openternary.calibration import runner as calibration_runner
    from openternary.utils.device import BackendInfo

    snapshot, cfg = local_teacher
    cfg.device = "auto"
    fake_backend = BackendInfo(
        backend="cuda-rocm",
        device_name="simulated transactional OOM device",
        total_bytes=8 * 1024**3,
        free_bytes=6 * 1024**3,
        is_hip=True,
        is_cuda=False,
        bf16_supported=False,
    )
    monkeypatch.setattr(calibration_runner, "detect_backend", lambda: fake_backend)

    baseline = tmp_path / "baseline-transactional"
    run_calibration(cfg, snapshot, baseline)

    original_step = torch.optim.Adam.step
    injected = False

    def updated_then_oom(optimizer, *args, **kwargs):
        nonlocal injected
        result = original_step(optimizer, *args, **kwargs)
        if not injected:
            injected = True
            raise torch.OutOfMemoryError("injected after optimizer mutation")
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", updated_then_oom)
    retried = tmp_path / "retried-transactional"
    run_calibration(cfg, snapshot, retried)

    expected = json.loads((baseline / "calibration.json").read_text())
    actual = json.loads((retried / "calibration.json").read_text())
    assert injected is True
    assert actual["calibrated_content_fingerprint"] == expected["calibrated_content_fingerprint"]
    assert actual["loss_history"] == pytest.approx(expected["loss_history"], abs=1e-9)
    assert actual["effective_runtime"]["oom_retries"] == 1
    assert actual["effective_runtime"]["minimum_microbatch_used"] < actual["effective_runtime"]["initial_microbatch"]


def test_materialize_rejects_missing_parameters(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    state = out / "artifacts/calibration_state/model_layers_0_mlp_up_proj.safetensors"
    save_file({"unrelated": torch.zeros(1)}, str(state))
    with pytest.raises(ValueError, match="parameter"):
        run_calibration(cfg, snapshot, out, materialize_only=True)


def test_fresh_run_does_not_overwrite_existing_checkpoint(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    before = (out / "config.yaml").read_bytes()
    cfg.calibration.loss = "l1"
    with pytest.raises(ValueError, match="existing|resume"):
        run_calibration(cfg, snapshot, out)
    assert (out / "config.yaml").read_bytes() == before


def test_real_path_rejects_tokenized_split_overlap(local_teacher, tmp_path, monkeypatch):
    snapshot, cfg = local_teacher

    class CollidingTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.ones(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: CollidingTokenizer())
    with pytest.raises(ValueError, match="overlap|contamination"):
        run_calibration(cfg, snapshot, tmp_path / "run")


def test_real_path_rejects_materialize_bypass(local_teacher, tmp_path, monkeypatch):
    snapshot, cfg = local_teacher
    monkeypatch.setenv("OT_SKIP_MATERIALIZE", "1")
    with pytest.raises(ValueError, match="test-only"):
        run_calibration(cfg, snapshot, tmp_path / "run")


def test_small_real_snapshot_does_not_use_simulation(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.calibration.num_samples = 4
    out = tmp_path / "run"
    run_calibration(cfg, snapshot, out)
    report = json.loads((out / "calibration.json").read_text())
    assert report["used_dummy_simulation"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None,
    reason="requires a real ROCm device",
)
def test_small_real_snapshot_runs_transactional_rocm_path(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.device = "auto"
    cfg.calibration.num_samples = 4
    out = tmp_path / "rocm-real-path"
    run_calibration(cfg, snapshot, out)
    report = json.loads((out / "calibration.json").read_text())
    assert report["backend"] == "cuda-rocm"
    assert report["used_dummy_simulation"] is False
    assert report["effective_runtime"]["modules"]
    assert {item["device"] for item in report["effective_runtime"]["modules"]} == {"cuda:0"}
    assert report["calibrated_content_fingerprint"]


@pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None,
    reason="requires a real ROCm device",
)
def test_soft_to_hard_runs_transactional_rocm_path_and_saves_hard_snapshot(local_teacher, tmp_path):
    snapshot, cfg = local_teacher
    cfg.device = "auto"
    cfg.calibration.method = "recon-soft-to-hard"
    cfg.calibration.steps = 3
    cfg.calibration.hard_fraction = 0.34
    cfg.calibration.num_samples = 4
    out = tmp_path / "soft-rocm-real-path"

    run_calibration(cfg, snapshot, out)

    report = json.loads((out / "calibration.json").read_text())
    assert report["backend"] == "cuda-rocm"
    assert report["assignment"]["final_state"] == "hard"
    assert {item["device"] for item in report["effective_runtime"]["modules"]} == {"cuda:0"}
    assert report["calibrated_content_fingerprint"]


def test_missing_explicit_snapshot_is_not_simulated(local_teacher, tmp_path):
    _, cfg = local_teacher
    with pytest.raises(FileNotFoundError, match="snapshot"):
        run_calibration(cfg, tmp_path / "missing", tmp_path / "run")
