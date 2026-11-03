"""Host and GPU detection for `openternary setup`. Never imports torch; never installs anything.

Every probe shells out to a vendor tool that may be absent, so each one is optional and its
parser is a pure function (tested with captured outputs). Detection reports what it saw; the
choice of PyTorch build is made in ``services.torch_setup``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Runner = Callable[[list[str]], str | None]


@dataclass
class Gpu:
    vendor: str  # nvidia | amd | intel | apple
    name: str
    arch: str | None = None  # AMD gfx target, e.g. gfx1201
    driver: str | None = None
    cuda_driver: str | None = None  # highest CUDA version the NVIDIA driver supports, e.g. "13.0"
    source: str = ""
    integrated: bool = False  # CPU-integrated graphics (never preferred over a discrete GPU)


@dataclass
class Host:
    os: str  # windows | linux | macos
    machine: str
    wsl: bool
    python: str
    gpus: list[Gpu] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def first(self, vendor: str) -> Gpu | None:
        return next((g for g in self.gpus if g.vendor == vendor), None)


def run(command: list[str], timeout: float = 15.0) -> str | None:
    """Run a probe command; None when it is missing, fails or times out."""
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


# --- AMD device id / marketing name -> gfx target ------------------------------------------
# PCI device ids (vendor 0x1002). Unknown cards fall back to the name table, then to None.
AMD_DEVICE_ARCH = {
    "7550": "gfx1201",  # Navi 48: RX 9070 XT / 9070 / 9070 GRE
    "7590": "gfx1200",  # Navi 44: RX 9060 XT / 9060
    "744C": "gfx1100",  # Navi 31: RX 7900 XTX / XT / GRE
    "747E": "gfx1101",  # Navi 32: RX 7800 XT / 7700 XT
    "7480": "gfx1102",  # Navi 33: RX 7600 / 7600 XT
    "73BF": "gfx1030",  # Navi 21: RX 6950 / 6900 / 6800
    "73DF": "gfx1031",  # Navi 22: RX 6750 / 6700
    "73FF": "gfx1032",  # Navi 23: RX 6650 / 6600
    "164E": "gfx1036",  # Raphael: Ryzen 7000 desktop integrated graphics
}
INTEGRATED_NAME = re.compile(r"^AMD Radeon(\(TM\))? Graphics$|Radeon\s+\d{3}M\b|Vega \d+ Graphics", re.IGNORECASE)
AMD_NAME_ARCH = (
    (r"RX\s*9070", "gfx1201"),
    (r"RX\s*9060", "gfx1200"),
    (r"RX\s*79\d\d", "gfx1100"),
    (r"RX\s*7[78]\d\d", "gfx1101"),
    (r"RX\s*76\d\d", "gfx1102"),
    (r"RX\s*6[89]\d\d", "gfx1030"),
    (r"RX\s*67\d\d", "gfx1031"),
    (r"RX\s*66\d\d", "gfx1032"),
    (r"Radeon\s*8060S|Ryzen AI Max", "gfx1151"),
)


def amd_arch(device_id: str | None, name: str | None) -> str | None:
    if device_id and device_id.upper() in AMD_DEVICE_ARCH:
        return AMD_DEVICE_ARCH[device_id.upper()]
    for pattern, arch in AMD_NAME_ARCH:
        if arch and name and re.search(pattern, name, re.IGNORECASE):
            return arch
    return None


# --- parsers (pure) ---------------------------------------------------------------------------
def parse_nvidia_smi(query_csv: str | None, banner: str | None) -> list[Gpu]:
    if not query_csv:
        return []
    cuda = None
    if banner:
        match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", banner)
        cuda = match.group(1) if match else None
    gpus = []
    for line in query_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            gpus.append(
                Gpu(
                    "nvidia",
                    parts[0],
                    driver=parts[1] if len(parts) > 1 else None,
                    cuda_driver=cuda,
                    source="nvidia-smi",
                )
            )
    return gpus


def parse_windows_video_controllers(raw: str | None) -> list[Gpu]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    rows = data if isinstance(data, list) else [data]
    gpus = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "")
        pnp = str(row.get("PNPDeviceID") or "").upper()
        vendor_id = re.search(r"VEN_([0-9A-F]{4})", pnp)
        device_id = re.search(r"DEV_([0-9A-F]{4})", pnp)
        vendor = {"1002": "amd", "10DE": "nvidia", "8086": "intel"}.get(vendor_id.group(1) if vendor_id else "", "")
        if not vendor:
            continue
        gpus.append(
            Gpu(
                vendor,
                name,
                arch=amd_arch(device_id.group(1) if device_id else None, name) if vendor == "amd" else None,
                driver=str(row.get("DriverVersion") or "") or None,
                source="Win32_VideoController",
                integrated=vendor in {"amd", "intel"} and bool(INTEGRATED_NAME.search(name.strip())),
            )
        )
    return gpus


def parse_rocminfo(raw: str | None) -> list[Gpu]:
    if not raw:
        return []
    gpus: list[Gpu] = []
    for block in raw.split("*******"):
        arch = re.search(r"^\s*Name:\s*(gfx[0-9a-f]+)\s*$", block, re.MULTILINE)
        if not arch:
            continue
        marketing = re.search(r"^\s*Marketing Name:\s*(.+?)\s*$", block, re.MULTILINE)
        name = marketing.group(1) if marketing else arch.group(1)
        if all(g.arch != arch.group(1) or g.name != name for g in gpus):
            gpus.append(Gpu("amd", name, arch=arch.group(1), source="rocminfo"))
    return gpus


def linux_sysfs_amd(root: Path = Path("/sys/class/drm")) -> list[Gpu]:
    gpus = []
    for vendor_file in sorted(root.glob("card*/device/vendor")):
        try:
            if vendor_file.read_text().strip().lower() != "0x1002":
                continue
            device_id = (vendor_file.parent / "device").read_text().strip().lower().removeprefix("0x")
        except OSError:
            continue
        arch = amd_arch(device_id, None)
        gpus.append(Gpu("amd", f"AMD GPU 0x{device_id}", arch=arch, source="sysfs"))
    return gpus


def is_wsl(proc_version: str | None = None) -> bool:
    if proc_version is None:
        try:
            proc_version = Path("/proc/version").read_text()
        except OSError:
            proc_version = ""
    return "microsoft" in proc_version.lower() or Path("/dev/dxg").exists()


# --- detection ----------------------------------------------------------------------------------
WINDOWS_GPU_QUERY = (
    "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress"
)


def detect(runner: Runner = run) -> Host:
    system = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    host = Host(
        os=system,
        machine=platform.machine().lower(),
        wsl=system == "linux" and is_wsl(),
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    nvidia = parse_nvidia_smi(
        runner(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        runner(["nvidia-smi"]),
    )
    host.gpus += nvidia
    if system == "windows":
        shell = "pwsh" if shutil.which("pwsh") else "powershell"
        for gpu in parse_windows_video_controllers(runner([shell, "-NoProfile", "-Command", WINDOWS_GPU_QUERY])):
            if gpu.vendor == "nvidia" and nvidia:
                continue  # nvidia-smi already gave richer data
            host.gpus.append(gpu)
    elif system == "linux":
        amd = parse_rocminfo(runner(["rocminfo"])) or linux_sysfs_amd()
        host.gpus += amd
        if host.wsl and not amd and not nvidia:
            host.notes.append("wsl_gpu_not_visible")
    elif system == "macos" and host.machine == "arm64":
        host.gpus.append(Gpu("apple", "Apple silicon GPU (MPS)", source="platform"))
    if os.environ.get("OPENTERNARY_GPU_ARCH"):
        for gpu in host.gpus:
            if gpu.vendor == "amd":
                gpu.arch = os.environ["OPENTERNARY_GPU_ARCH"]
    host.gpus.sort(key=lambda g: (g.integrated, g.vendor != "nvidia", g.arch is None))  # discrete first
    amd = [g for g in host.gpus if g.vendor == "amd"]
    if amd and not any(g.arch for g in amd):  # an unknown iGPU next to a known card is not a problem
        host.notes.append("amd_arch_unknown")
    return host
