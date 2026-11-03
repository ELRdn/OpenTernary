"""`openternary setup`: detection parsers, live-index parsing, choice, plan and CLI safety (no network/GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from packaging.tags import Tag
from typer.testing import CliRunner

from openternary.cli.main import app
from openternary.services import hardware
from openternary.services import torch_setup as ts
from openternary.services.hardware import Gpu, Host

WIN312 = {Tag("cp312", "cp312", "win_amd64"), Tag("py3", "none", "any")}
WIN314 = {Tag("cp314", "cp314", "win_amd64"), Tag("py3", "none", "any")}
LINUX312 = {Tag("cp312", "cp312", "manylinux_2_28_x86_64"), Tag("py3", "none", "any")}

AMD_TORCH = [
    "torch-2.13.0+rocm10.0.0-cp312-cp312-win_amd64.whl",
    "torch-2.13.0+rocm10.0.0-cp313-cp313-win_amd64.whl",
    "torch-2.12.0+rocm7.13.0-cp312-cp312-win_amd64.whl",
    "torch-2.13.0+rocm10.0.0-cp312-cp312-manylinux_2_28_x86_64.whl",
]
AMD_VISION = [
    "torchvision-0.28.0+rocm10.0.0-cp312-cp312-win_amd64.whl",
    "torchvision-0.27.0+rocm7.13.0-cp312-cp312-win_amd64.whl",
]
PYTORCH_TORCH = [
    "torch-2.13.0+cu130-cp312-cp312-win_amd64.whl",
    "torch-2.13.0+cu126-cp312-cp312-win_amd64.whl",
    "torch-2.13.0+cpu-cp312-cp312-win_amd64.whl",
    "torch-2.13.0+xpu-cp312-cp312-win_amd64.whl",
    "torch-2.14.0.dev20261101+cu130-cp312-cp312-win_amd64.whl",
    "torch-2.13.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl",
    "torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
]
PYTORCH_VISION = [
    "torchvision-0.28.0+cu130-cp312-cp312-win_amd64.whl",
    "torchvision-0.28.0+cu126-cp312-cp312-win_amd64.whl",
    "torchvision-0.28.0+cpu-cp312-cp312-win_amd64.whl",
]
PYPI_TORCH = ["torch-2.15.0-cp312-cp312-win_amd64.whl", "torch-2.13.0-cp312-cp312-win_amd64.whl"]
PYPI_VISION = ["torchvision-0.28.0-cp312-cp312-win_amd64.whl"]  # no 0.30 yet for torch 2.15


def windows(*gpus: Gpu, python: str = "3.12") -> Host:
    return Host(os="windows", machine="amd64", wsl=False, python=python, gpus=list(gpus))


RX9070XT = Gpu("amd", "AMD Radeon RX 9070 XT", arch="gfx1201", source="Win32_VideoController")
RTX = Gpu("nvidia", "NVIDIA GeForce RTX 4090", driver="575.00", cuda_driver="12.8", source="nvidia-smi")


# --- detection parsers ---------------------------------------------------------------------------
def test_windows_video_controller_detects_rx9070xt_arch() -> None:
    raw = json.dumps(
        [
            {
                "Name": "AMD Radeon RX 9070 XT",
                "PNPDeviceID": "PCI\\VEN_1002&DEV_7550&SUBSYS_E4521DA2&REV_C0\\6&1",
                "DriverVersion": "32.0.21025.10016",
            },
            {"Name": "Microsoft Basic Display Adapter", "PNPDeviceID": "ROOT\\BASICDISPLAY\\0000"},
        ]
    )
    (gpu,) = hardware.parse_windows_video_controllers(raw)
    assert (gpu.vendor, gpu.arch, gpu.driver) == ("amd", "gfx1201", "32.0.21025.10016")


def test_windows_single_object_and_name_fallback() -> None:
    raw = json.dumps({"Name": "AMD Radeon RX 7800 XT", "PNPDeviceID": "PCI\\VEN_1002&DEV_FFFF"})
    (gpu,) = hardware.parse_windows_video_controllers(raw)
    assert gpu.arch == "gfx1101"
    assert hardware.parse_windows_video_controllers("not json") == []


def test_nvidia_smi_reads_driver_and_cuda_ceiling() -> None:
    banner = "| NVIDIA-SMI 575.00   Driver Version: 575.00   CUDA Version: 12.8     |"
    (gpu,) = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 575.00\n", banner)
    assert (gpu.name, gpu.driver, gpu.cuda_driver) == ("NVIDIA GeForce RTX 4090", "575.00", "12.8")
    assert hardware.parse_nvidia_smi(None, None) == []


def test_rocminfo_skips_cpu_agents() -> None:
    raw = (
        "*******\nAgent 1\n*******\n  Name:                    AMD Ryzen 9 7950X\n  Marketing Name: AMD Ryzen\n"
        "*******\nAgent 2\n*******\n  Name:                    gfx1201\n"
        "  Marketing Name:          AMD Radeon RX 9070 XT\n"
    )
    (gpu,) = hardware.parse_rocminfo(raw)
    assert (gpu.arch, gpu.name) == ("gfx1201", "AMD Radeon RX 9070 XT")


def test_wsl_detection_from_proc_version() -> None:
    assert hardware.is_wsl("Linux version 5.15.167.4-microsoft-standard-WSL2")


# --- index listing -------------------------------------------------------------------------------
def test_listing_parses_html_with_encoded_local_versions_and_json() -> None:
    html = (
        '<a href="/whl/rocm/torch-2.13.0%2Brocm10.0.0-cp312-cp312-win_amd64.whl#sha256=ab">'
        "torch-2.13.0+rocm10.0.0-cp312-cp312-win_amd64.whl</a>"
        '<a href="torch-2.12.0%2Bcpu-cp312-cp312-win_amd64.whl"></a><a href="torch-2.0.tar.gz">sdist</a>'
    )
    assert ts.parse_listing("text/html", html) == [
        "torch-2.13.0+rocm10.0.0-cp312-cp312-win_amd64.whl",
        "torch-2.12.0+cpu-cp312-cp312-win_amd64.whl",
    ]
    payload = json.dumps({"files": [{"filename": "torch-2.13.0-cp312-cp312-win_amd64.whl"}, {"filename": "x.tar.gz"}]})
    assert ts.parse_listing("application/vnd.pypi.simple.v1+json", payload) == [
        "torch-2.13.0-cp312-cp312-win_amd64.whl"
    ]


def test_index_override_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENTERNARY_INDEX_AMD", "https://mirror.example/amd")
    assert ts.provider_urls("amd") == ("https://mirror.example/amd", "https://mirror.example/amd")


# --- choice -------------------------------------------------------------------------------------
def offer_for(host: Host, supported: set[Tag]) -> ts.Offer:
    listings = {"amd": AMD_TORCH, "pytorch": PYTORCH_TORCH, "pypi": PYPI_TORCH}
    offer = ts.build_offer(host, listings, supported=supported)
    offer.candidates = ts.with_torchvision(
        offer.candidates, {"amd": AMD_VISION, "pytorch": PYTORCH_VISION, "pypi": PYPI_VISION}, supported
    )
    return offer


def test_amd_on_windows_recommends_amd_rocm_with_matching_torchvision() -> None:
    host = windows(RX9070XT)
    pool, warnings = ts.rank(host, offer_for(host, WIN312).candidates, ts.wanted_backend(host, "auto"))
    best = pool[0]
    assert (best.provider, best.version, best.torchvision) == ("amd", "2.13.0+rocm10.0.0", "0.28.0+rocm10.0.0")
    assert all(c.backend == "rocm" for c in pool) and not warnings


def test_python_without_wheels_reports_which_pythons_have_them() -> None:
    host = windows(RX9070XT, python="3.14")
    offer = offer_for(host, WIN314)
    pool, _ = ts.rank(host, offer.candidates, "rocm")
    assert pool == []
    assert offer.other_pythons["rocm"] == ["3.12", "3.13"]


def test_cuda_respects_driver_ceiling_and_skips_xpu_and_prereleases() -> None:
    host = windows(RTX)
    offer = offer_for(host, WIN312)
    pool, _ = ts.rank(host, offer.candidates, "cuda")
    assert [c.variant for c in pool] == ["cu126"]  # cu130 needs a CUDA 13 driver
    assert not any(c.variant == "xpu" or "dev" in c.version for c in offer.candidates)


def test_pairs_with_torchvision_beat_newer_torch_without_one() -> None:
    host = windows()
    pool, _ = ts.rank(host, offer_for(host, WIN312).candidates, "cpu")
    assert pool[0].torchvision is not None
    assert any(c.version == "2.15.0" and c.torchvision is None for c in pool)


def test_linux_cpu_prefers_cpu_index_but_falls_back_to_pypi() -> None:
    host = Host(os="linux", machine="x86_64", wsl=False, python="3.12")
    offer = ts.build_offer(
        host,
        {"pytorch": PYTORCH_TORCH, "pypi": ["torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"]},
        supported=LINUX312,
    )
    pool, _ = ts.rank(host, offer.candidates, "cpu")
    assert [c.variant for c in pool] == ["cpu", "pypi"]


def test_minimum_torch_from_openternary_is_enforced() -> None:
    from packaging.version import Version

    offer = ts.build_offer(windows(RX9070XT), {"amd": AMD_TORCH}, supported=WIN312, minimum=Version("2.13"))
    assert {c.version for c in offer.candidates} == {"2.13.0+rocm10.0.0"}


def test_verification_follows_the_requested_backend() -> None:
    assert ts.probe_device("rocm") == "cuda"
    assert ts.probe_device("mps") == "mps"
    assert ts.probe_device("cpu") == "cpu"


# --- plan -------------------------------------------------------------------------------------
def test_plan_keeps_torch_on_the_chosen_source_and_everything_else_on_pypi() -> None:
    host = windows(RX9070XT)
    pool, _ = ts.rank(host, offer_for(host, WIN312).candidates, "rocm")
    plan = ts.make_plan(pool[0], ["transformers<6,>=5.6.2"], "uv", python="py", backend="rocm")
    cmd = plan.command
    assert cmd[:5] == ["uv", "pip", "install", "--python", "py"]
    assert cmd[cmd.index("--index-url") + 1] == ts.PYPI
    assert cmd[cmd.index("--extra-index-url") + 1] == "https://stable.repo.amd.com/rocm/whl-next"
    assert "torch[device-all]==2.13.0+rocm10.0.0" in cmd and "torchvision==0.28.0+rocm10.0.0" in cmd
    assert cmd[-1] == "transformers<6,>=5.6.2"


def test_plan_without_torchvision_does_not_touch_it() -> None:
    candidate = ts.Candidate("pypi", "cpu", "2.15.0", "pypi", ts.PYPI)
    plan = ts.make_plan(candidate, [], "uv", python="py")
    assert "--extra-index-url" not in plan.command
    assert not any("torchvision" in part for part in plan.command)


def test_profile_round_trip_and_restore_plan(tmp_path: Path) -> None:
    candidate = ts.Candidate("amd", "rocm", "2.13.0+rocm10.0.0", "rocm10.0.0", "https://idx", "0.28.0+rocm10.0.0")
    plan = ts.make_plan(candidate, ["datasets>=2.18"], "uv", python="py", backend="rocm")
    path = tmp_path / ts.PROFILE
    ts.write_profile(plan, windows(RX9070XT), {"status": "passed", "device_name": "RX 9070 XT"}, path)
    ts.write_profile(plan, windows(RX9070XT), {"status": "failed"}, path)
    profile = ts.read_profile(path)
    assert profile is not None and [h["status"] for h in profile["history"]] == ["passed", "failed"]
    assert ts.verified_here(profile) == {"2.13.0+rocm10.0.0"}
    again = ts.plan_from_profile(profile, ["datasets>=2.18"], "uv")
    assert again.command[again.command.index("--python") + 1] != ""
    assert [p for p in again.command if p.startswith("torch")] == [p for p in plan.command if p.startswith("torch")]
    with pytest.raises(ValueError, match="incomplete"):
        ts.plan_from_profile({"backend": "rocm"}, [], "uv")


# --- CLI ----------------------------------------------------------------------------------------
@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {"install": 0}
    listings = {
        ("amd", "torch"): AMD_TORCH,
        ("amd", "torchvision"): AMD_VISION,
        ("pytorch", "torch"): PYTORCH_TORCH,
        ("pytorch", "torchvision"): PYTORCH_VISION,
        ("pypi", "torch"): PYPI_TORCH,
        ("pypi", "torchvision"): PYPI_VISION,
    }
    monkeypatch.setattr(hardware, "detect", lambda: windows(RX9070XT))
    monkeypatch.setattr(ts, "list_wheels", lambda provider, package: listings[(provider, package)])
    monkeypatch.setattr(ts, "sys_tags", lambda: iter(WIN312))
    monkeypatch.setattr(ts, "find_uv", lambda: "uv")
    monkeypatch.setattr(ts, "profile_path", lambda: tmp_path / ts.PROFILE)

    def install(plan: ts.Plan) -> int:
        calls["install"] += 1
        return calls.get("install_code", 0)

    monkeypatch.setattr(ts, "install", install)
    monkeypatch.setattr(ts, "verify", lambda backend: calls.get("verify", {"status": "passed", "device_name": "RX"}))
    return calls


def envelope(result: Any) -> dict[str, Any]:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cli_dry_run_returns_the_plan_and_installs_nothing(offline: dict[str, Any]) -> None:
    result = CliRunner().invoke(app, ["setup", "--dry-run", "--json"])
    env = envelope(result)
    assert env["exit_code"] == 0 and offline["install"] == 0
    assert env["result"]["plan"]["candidate"]["version"] == "2.13.0+rocm10.0.0"
    assert env["result"]["backend"] == "rocm"


def test_cli_refuses_to_install_without_confirmation(offline: dict[str, Any]) -> None:
    result = CliRunner().invoke(app, ["setup", "--json"])
    env = envelope(result)
    assert env["exit_code"] == 2 and "--yes" in env["error"]
    assert offline["install"] == 0


def test_cli_install_verify_and_record(offline: dict[str, Any], tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["setup", "--yes", "--json", "--torch", "2.12.0"])
    env = envelope(result)
    assert env["exit_code"] == 0 and offline["install"] == 1
    profile = json.loads((tmp_path / ts.PROFILE).read_text(encoding="utf-8"))
    assert profile["version"] == "2.12.0+rocm7.13.0" and profile["verification"]["status"] == "passed"


def test_cli_failed_device_check_is_recorded_and_exits_5(offline: dict[str, Any], tmp_path: Path) -> None:
    offline["verify"] = {"status": "failed", "reason": "hipErrorNoBinaryForGpu"}
    env = envelope(CliRunner().invoke(app, ["setup", "--yes", "--json"]))
    assert env["exit_code"] == 5 and "hipErrorNoBinaryForGpu" in env["error"]
    assert json.loads((tmp_path / ts.PROFILE).read_text())["verification"]["status"] == "failed"


def test_cli_failed_install_exits_1(offline: dict[str, Any]) -> None:
    offline["install_code"] = 2
    env = envelope(CliRunner().invoke(app, ["setup", "--yes", "--json"]))
    assert env["exit_code"] == 1


def test_cli_python_mismatch_explains_what_to_do(offline: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect", lambda: windows(RX9070XT, python="3.14"))
    monkeypatch.setattr(ts, "sys_tags", lambda: iter(WIN314))
    result = CliRunner().invoke(app, ["setup", "--dry-run", "--json"])
    env = envelope(result)
    assert env["exit_code"] == 3
    assert env["result"]["other_pythons"]["rocm"] == ["3.12", "3.13"]


def test_cli_list_shows_everything_without_installing(offline: dict[str, Any]) -> None:
    env = envelope(CliRunner().invoke(app, ["setup", "--list", "--json"]))
    assert env["exit_code"] == 0 and offline["install"] == 0
    backends = {c["backend"] for c in env["result"]["candidates_all"]}
    assert {"rocm", "cuda", "cpu"} <= backends
