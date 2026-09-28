"""Power draw, for the 3D view's optional power layer.

Measured:
  cpu, cpu_core  RAPL energy counters in /sys/class/powercap (Intel and AMD).
                 The kernel makes them root-only (CVE-2020-8694, "PLATYPUS");
                 contrib/60-flowscope-rapl.rules grants read access to a group.
  gpu            NVIDIA NVML (libnvidia-ml, part of the driver) via ctypes.
Estimated:
  usb:<dev>      each USB device's declared maximum draw (bMaxPower × 5 V) —
                 a budget the device asked for, not a measurement.

Nothing else on a typical board exposes power: NICs, audio, SSDs and the
chipset have no readable sensors.
"""
from __future__ import annotations

import ctypes
import glob
import os
import time
from dataclasses import dataclass

_RAPL = "/sys/class/powercap"
_USB = "/sys/bus/usb/devices"


@dataclass
class Power:
    watts: float | None  # None = unavailable (see note)
    source: str
    detail: str = ""
    note: str = ""  # why it's unavailable, e.g. "needs permission"
    estimated: bool = False


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


class _Rapl:
    """Package (and core) power from RAPL energy counters."""

    def __init__(self):
        self.last: dict[str, tuple[int, float]] = {}
        self.domains = []  # (key, path, name)
        for d in sorted(glob.glob(f"{_RAPL}/intel-rapl:*")):
            if not os.path.exists(f"{d}/energy_uj"):
                continue
            try:
                name = _read(f"{d}/name")
            except OSError:
                continue
            base = os.path.basename(d)
            if name.startswith("package"):
                self.domains.append(("cpu", d, name, base))
            elif name == "core":
                self.domains.append(("cpu_core", d, name, base))

    def sample(self, out: dict[str, Power]) -> None:
        now = time.monotonic()
        acc: dict[str, list] = {}  # key → [watts, names, note]
        for key, path, name, base in self.domains:
            entry = acc.setdefault(key, [0.0, [], ""])
            entry[1].append(base)
            try:
                energy = int(_read(f"{path}/energy_uj"))
            except PermissionError:
                entry[2] = "needs permission"
                continue
            except (OSError, ValueError):
                entry[2] = "unreadable"
                continue
            prev = self.last.get(path)
            self.last[path] = (energy, now)
            if prev is None or now <= prev[1]:
                entry[2] = entry[2] or "warming up"
                continue
            delta = energy - prev[0]
            if delta < 0:  # counter wrapped
                try:
                    delta += int(_read(f"{path}/max_energy_range_uj"))
                except (OSError, ValueError):
                    delta = 0
            entry[0] += delta / 1e6 / (now - prev[1])
        labels = {"cpu": "CPU package", "cpu_core": "CPU cores"}
        for key, (watts, names, note) in acc.items():
            out[key] = Power(None if note else watts, f"RAPL ({', '.join(names)})",
                             labels[key], note)


class _Nvml:
    """Board power of the first NVIDIA GPU through NVML."""

    def __init__(self):
        self.lib = None
        self.handle = ctypes.c_void_p()
        self.name = "NVIDIA GPU"
        self.limit = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            count = ctypes.c_uint()
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
                lib.nvmlShutdown()
                return
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                lib.nvmlShutdown()
                return
            buf = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(self.handle, buf, 96) == 0:
                self.name = buf.value.decode(errors="replace")
            mw = ctypes.c_uint()
            if lib.nvmlDeviceGetEnforcedPowerLimit(self.handle, ctypes.byref(mw)) == 0:
                self.limit = mw.value / 1000
            self.lib = lib
        except (OSError, AttributeError):
            self.lib = None

    def sample(self, out: dict[str, Power]) -> None:
        if self.lib is None:
            return
        mw = ctypes.c_uint()
        ok = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(mw)) == 0
        detail = self.name + (f" · {self.limit:.0f} W limit" if self.limit else "")
        out["gpu"] = Power(mw.value / 1000 if ok else None, "NVML", detail,
                           "" if ok else "unsupported")

    def close(self) -> None:
        if self.lib is not None:
            self.lib.nvmlShutdown()
            self.lib = None


def _usb(out: dict[str, Power]) -> None:
    for base in glob.glob(f"{_USB}/*"):
        name = os.path.basename(base)
        if ":" in name or not os.path.exists(f"{base}/bMaxPower"):
            continue
        try:
            ma = int(_read(f"{base}/bMaxPower").rstrip("mA") or 0)
        except (OSError, ValueError):
            continue
        out[f"usb:{name}"] = Power(ma * 5 / 1000, "USB descriptor",
                                   f"declared max {ma} mA at 5 V", estimated=True)


class PowerMeter:
    def __init__(self):
        self.rapl = _Rapl()
        self.nvml = _Nvml()
        self.readings: dict[str, Power] = {}

    def sample(self) -> None:
        out: dict[str, Power] = {}
        self.rapl.sample(out)
        self.nvml.sample(out)
        _usb(out)
        self.readings = out

    def close(self) -> None:
        self.nvml.close()


def fmt_watts(w: float | None) -> str:
    if w is None:
        return "n/a"
    return f"{w:.2f} W" if w < 10 else f"{w:.1f} W" if w < 100 else f"{w:.0f} W"
