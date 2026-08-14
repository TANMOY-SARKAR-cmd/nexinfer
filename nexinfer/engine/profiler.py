"""Hardware resource profiler.

Detects CPU cores/RAM, NVIDIA GPUs (via nvidia-smi or torch.cuda),
AMD GPUs (rocm-smi / lsgpu), Intel iGPU/NPU (OpenVINO heuristics),
TPUs (libtpu env vars / TPU_VISIBLE_CHIPS), and generic display
adapters on Windows (DXGI) -- all without requiring any vendor SDK to
be installed. Each detected device becomes a vendor-neutral
``DeviceId`` consumed by the orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexinfer.backends.base import DeviceInfo
from nexinfer.engine.types import DeviceId, DeviceKind

log = logging.getLogger("nexinfer.profiler")


def _cmd(args: list[str], timeout: float = 3.0) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _detect_cpu() -> list[DeviceInfo]:
    cores = os.cpu_count() or 1
    # RAM: cross-platform
    total_ram = 0
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        total_ram = int(line.split()[1]) * 1024
        except OSError:
            pass
    elif platform.system() == "Windows":
        try:
            import ctypes
            total_ram = (ctypes.c_ulonglong(0)).value  # placeholder
            import ctypes.wintypes
            kb = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kb))  # type: ignore
            total_ram = kb.value * 1024
        except Exception:  # noqa: BLE001
            total_ram = 8 * 1024 ** 3
    else:
        total_ram = 8 * 1024 ** 3
    return [
        DeviceInfo(
            device_id="/cpu:0",
            kind=DeviceKind.CPU,
            vendor="generic",
            name=f"CPU ({cores} cores)",
            total_memory_bytes=max(total_ram, 1),
            compute_score=1.0,
        )
    ]


def _detect_nvidia() -> list[DeviceInfo]:
    out = _cmd(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,json,noheader,nounits"])
    if not out:
        return []
    devices = []
    try:
        for row in json.loads(f"[{out}]"):
            idx, name, mem_mb = int(row["index"]), row["name"], float(row["memory.total"])
            devices.append(
                DeviceInfo(
                    device_id=f"/gpu:nvidia:{idx}",
                    kind=DeviceKind.GPU_NVIDIA,
                    vendor="nvidia",
                    name=name,
                    total_memory_bytes=int(mem_mb * 1024 * 1024),
                    compute_score=mem_mb / 1024,  # proxy: bigger VRAM -> higher score
                )
            )
    except (json.JSONDecodeError, KeyError):
        pass
    return devices


def _detect_amd() -> list[DeviceInfo]:
    out = _cmd(["rocm-smi", "--showmeminfo", "vram", "--json"]) or _cmd(["lsgpu"])
    if not out:
        return []
    devices = []
    try:
        data = json.loads(out) if out.startswith("{") else {}
        cards = data.get("system", {}).get("GPU", {}) if isinstance(data, dict) else {}
        for key, info in cards.items():
            idx = int("".join(c for c in key if c.isdigit()) or "0")
            vram_bytes = int(info.get("VRAM Total Memory (B)", 0) or 0)
            devices.append(
                DeviceInfo(
                    device_id=f"/gpu:amd:{idx}",
                    kind=DeviceKind.GPU_AMD,
                    vendor="amd",
                    name=info.get("Card series", key),
                    total_memory_bytes=vram_bytes,
                    compute_score=vram_bytes / (1024 ** 3),
                )
            )
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return devices


def _detect_intel() -> list[DeviceInfo]:
    """Intel iGPU/NPU: heuristics from lspci (Linux) or env hints."""
    devices = []
    if platform.system() == "Linux":
        out = _cmd(["lspci"]) or ""
        idx_g, idx_n = 0, 0
        for line in out.splitlines():
            low = line.lower()
            if "intel" in low and "vga" in low and "uhd" in low or "intel" in low and "iris" in low:
                devices.append(
                    DeviceInfo(
                        device_id=f"/gpu:intel:{idx_g}",
                        kind=DeviceKind.GPU_INTEL,
                        vendor="intel",
                        name="Intel Integrated Graphics",
                        total_memory_bytes=4 * 1024 ** 3,  # shared memory; conservatively 4GB
                        compute_score=1.0,
                    )
                )
                idx_g += 1
            if "neural" in low or "npu" in low:
                devices.append(
                    DeviceInfo(
                        device_id=f"/npu:intel:{idx_n}",
                        kind=DeviceKind.NPU_INTEL,
                        vendor="intel",
                        name="Intel NPU",
                        total_memory_bytes=0,  # shared with system memory
                        compute_score=0.5,
                    )
                )
                idx_n += 1
    if os.environ.get("OPENVINO_DEVICES"):
        for dev in os.environ["OPENVINO_DEVICES"].split(","):
            devices.append(DeviceInfo(dev, DeviceKind.NPU_INTEL, "intel", dev, 0, 0.5))
    return devices


def _detect_tpu() -> list[DeviceInfo]:
    """TPU detection: libtpu / TPU_VISIBLE_CHIPS / GCE metadata env hints."""
    devices = []
    visible = os.environ.get("TPU_VISIBLE_CHIPS") or os.environ.get("TPU_CHIPS_TO_PROCESS_ASSIGNMENT")
    num = 0
    if visible:
        num = max(1, len([c for c in visible.split(",") if c.strip()]))
    elif os.environ.get("TPU_NAME") or shutil.which("tpuinfo") or os.path.exists("/sys/bus/pci/devices"):
        # On TPU VMs, /dev/accel* or TPU_NAME is typically present
        num = int(os.environ.get("TPU_NUM_DEVICES", "1"))
    for i in range(num):
        devices.append(
            DeviceInfo(
                device_id=f"/tpu:{i}",
                kind=DeviceKind.TPU,
                vendor="google",
                name=f"TPU core {i}",
                total_memory_bytes=16 * 1024 ** 3,
                compute_score=8.0,
            )
        )
    return devices


def _detect_windows_dxgi() -> list[DeviceInfo]:
    """Windows-only generic GPU detection via a small PowerShell probe."""
    if platform.system() != "Windows":
        return []
    script = (
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM "
        "| ConvertTo-Json"
    )
    out = _cmd(["powershell", "-NoProfile", "-Command", script])
    if not out:
        return []
    devices = []
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        for i, gpu in enumerate(data):
            name = gpu.get("Name", "Unknown GPU")
            if any(k in name.lower() for k in ("nvidia", "amd", "radeon", "intel")):
                continue  # already covered by vendor-specific detectors
            devices.append(
                DeviceInfo(
                    device_id=f"/gpu:generic:{i}",
                    kind=DeviceKind.GPU_NVIDIA,  # generic bucket
                    vendor="generic",
                    name=name,
                    total_memory_bytes=int(gpu.get("AdapterRAM") or 0),
                    compute_score=1.0,
                )
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return devices


def profile_system(benchmark: bool = False) -> list[DeviceInfo]:
    """Detect all devices on the system."""
    devices: list[DeviceInfo] = []
    devices.extend(_detect_cpu())
    devices.extend(_detect_nvidia())
    devices.extend(_detect_amd())
    devices.extend(_detect_intel())
    devices.extend(_detect_tpu())
    devices.extend(_detect_windows_dxgi())

    if benchmark:
        for dev in devices:
            dev.compute_score = _microbench(dev)
    return devices


def _microbench(dev: DeviceInfo, mat_size: int = 1024, seconds: float = 1.0) -> float:
    """Quick matmul throughput probe on the given device (CPU path only
    for portability; backends override with device-native benches)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((mat_size, mat_size)).astype(np.float32)
    b = rng.standard_normal((mat_size, mat_size)).astype(np.float32)
    import time
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        a @ b
        n += 1
    elapsed = time.perf_counter() - t0
    gflops = n * 2 * mat_size ** 3 / elapsed / 1e9
    return float(gflops)


@dataclass
class SystemProfile:
    devices: list[DeviceInfo]
    cpu_cores: int
    total_ram_gb: float
    gpu_vram_gb: float

    @classmethod
    def from_system(cls, benchmark: bool = False) -> "SystemProfile":
        devices = profile_system(benchmark=benchmark)
        cpu = next((d for d in devices if d.kind == DeviceKind.CPU), None)
        gpus = [d for d in devices if d.kind in (DeviceKind.GPU_NVIDIA, DeviceKind.GPU_AMD, DeviceKind.GPU_INTEL)]
        return cls(
            devices=devices,
            cpu_cores=os.cpu_count() or 1,
            total_ram_gb=(cpu.total_memory_bytes / 1024 ** 3) if cpu else 0.0,
            gpu_vram_gb=sum(d.total_memory_bytes for d in gpus) / 1024 ** 3,
        )
