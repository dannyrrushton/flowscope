"""Data sources. Each collector returns Readings holding *cumulative* counters;
the Monitor turns successive readings into rates.

Everything here reads kernel interfaces (/proc, /sys, /dev/input, HCI ioctls)
directly so no third-party packages or root are required. Sources that need
extra privilege (usbmon, /proc/tty/driver) are used when readable.
"""
from __future__ import annotations

import fcntl
import glob
import os
import re
import select
import socket
import struct
import threading
from dataclasses import dataclass, field

BYTES, EVENTS, IRQS = "B", "ev", "irq"


@dataclass
class Reading:
    key: str
    category: str
    name: str
    detail: str
    rx: int | None  # cumulative inbound counter (None = unavailable)
    tx: int | None  # cumulative outbound counter
    unit: str = BYTES
    usb: str | None = None  # sysfs path of the owning USB device, if any
    note: str = ""  # e.g. "needs root"
    stacked: bool = False  # layered on other channels (dm/md); excluded from totals


def _read(path: str, default: str = "") -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def usb_ancestor(sysfs_path: str) -> str | None:
    """Walk up from a sysfs device path to the USB device (dir with idVendor)."""
    try:
        p = os.path.realpath(sysfs_path)
    except OSError:
        return None
    while p and p != "/" and "/usb" in p:
        if os.path.exists(os.path.join(p, "idVendor")):
            return p
        p = os.path.dirname(p)
    return None


class Collector:
    category = ""

    def collect(self) -> list[Reading]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------- network
class NetworkCollector(Collector):
    category = "Network"

    def collect(self):
        out = []
        with open("/proc/net/dev") as f:
            lines = f.readlines()[2:]
        for line in lines:
            name, data = line.split(":", 1)
            name = name.strip()
            v = data.split()
            base = f"/sys/class/net/{name}"
            if name == "lo":
                kind = "loopback"
            elif os.path.exists(f"{base}/wireless") or os.path.exists(f"{base}/phy80211"):
                kind = "Wi-Fi"
            elif os.path.exists(f"{base}/bridge"):
                kind = "bridge"
            elif not os.path.exists(f"{base}/device"):
                kind = "virtual"
            else:
                kind = "ethernet"
            state = _read(f"{base}/operstate", "?")
            speed = _read(f"{base}/speed")
            detail = f"{kind} · {state}"
            if speed.isdigit() and int(speed) > 0:
                detail += f" · {int(speed)} Mb/s"
            out.append(Reading(f"net:{name}", self.category, name, detail,
                               int(v[0]), int(v[8]), BYTES,
                               usb_ancestor(f"{base}/device")))
        return out


# --------------------------------------------------------------------- storage
class StorageCollector(Collector):
    category = "Storage"

    def collect(self):
        out = []
        for base in sorted(glob.glob("/sys/block/*")):
            name = os.path.basename(base)
            if name.startswith(("loop", "ram")):
                continue
            stat = _read(f"{base}/stat").split()
            if len(stat) < 7:
                continue
            model = _read(f"{base}/device/model") or _read(f"{base}/dm/name")
            size_gb = int(_read(f"{base}/size", "0") or 0) * 512 / 1e9
            detail = " · ".join(x for x in (model, f"{size_gb:,.0f} GB") if x)
            stacked = bool(os.listdir(f"{base}/slaves")) if os.path.isdir(
                f"{base}/slaves") else False
            if stacked:
                detail += " · on " + ", ".join(sorted(os.listdir(f"{base}/slaves")))
            out.append(Reading(f"blk:{name}", self.category, name, detail,
                               int(stat[2]) * 512, int(stat[6]) * 512, BYTES,
                               usb_ancestor(f"{base}/device"), stacked=stacked))
        return out


# ----------------------------------------------------------------------- audio
_SAMPLE_BYTES = {"S8": 1, "U8": 1, "S16": 2, "U16": 2, "S24_3": 3, "U24_3": 3,
                 "S20_3": 3, "S24": 4, "U24": 4, "S32": 4, "U32": 4, "FLOAT": 4,
                 "FLOAT64": 8, "IEC958_SUBFRAME": 4, "DSD_U8": 1, "DSD_U16": 2,
                 "DSD_U32": 4}


def _sample_bytes(fmt: str) -> int:
    fmt = re.sub(r"_(LE|BE)$", "", fmt)
    return _SAMPLE_BYTES.get(fmt, 4)


def _audio_kind(text: str) -> str:
    t = text.upper()
    if any(k in t for k in ("IEC958", "SPDIF", "S/PDIF", "TOSLINK", "OPTICAL", "COAXIAL")):
        return "S/PDIF"
    if "HDMI" in t or "DISPLAYPORT" in t or re.search(r"\bDP\b", t):
        return "HDMI/DP"
    if "DIGITAL" in t:
        return "digital"
    return "analog"


class AudioCollector(Collector):
    """ALSA PCM substreams via /proc/asound. Works under PipeWire/Pulse too,
    since they hold the hardware device open. Byte counters are derived from
    the hardware pointer (frames) × frame size."""
    category = "Audio"

    def collect(self):
        out = []
        for sub in sorted(glob.glob("/proc/asound/card*/pcm*[cp]/sub*")):
            m = re.search(r"card(\d+)/pcm(\d+)([cp])/sub(\d+)", sub)
            card, dev, direction, subn = m.groups()
            info = dict(_kv(_read(f"{os.path.dirname(sub)}/info")))
            card_name = _read(f"/proc/asound/card{card}/id", f"card{card}")
            pcm_name = info.get("name", "") or info.get("id", "")
            kind = _audio_kind(f"{pcm_name} {info.get('id', '')}")
            capture = direction == "c"
            status = dict(_kv(_read(f"{sub}/status")))
            state = status.get("state", "closed")
            counter = 0
            fmt = ""
            if "hw_ptr" in status:
                hw = dict(_kv(_read(f"{sub}/hw_params")))
                ch = int(hw.get("channels", "2") or 2)
                sb = _sample_bytes(hw.get("format", ""))
                counter = int(status["hw_ptr"]) * ch * sb
                rate = hw.get("rate", "").split()[0] if hw.get("rate") else ""
                fmt = f"{hw.get('format', '')} {ch}ch {rate}Hz"
            detail = " · ".join(x for x in (
                kind, "capture" if capture else "playback",
                state.lower(), fmt) if x)
            name = f"{card_name}: {pcm_name or 'PCM'} (hw:{card},{dev})"
            if subn != "0":
                name += f" #{subn}"
            out.append(Reading(
                f"pcm:{card}:{dev}{direction}{subn}", self.category, name, detail,
                counter if capture else 0, 0 if capture else counter, BYTES,
                usb_ancestor(f"/sys/class/sound/card{card}/device")))
        return out


def _kv(text: str):
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            yield k.strip().lower(), v.strip()


# ------------------------------------------------------------------- bluetooth
_HCI_FMT = "<H8s6sIB8s3xIIIHHHH10I"  # struct hci_dev_info
_HCIGETDEVINFO = 0x800448D3  # _IOR('H', 211, int)


class BluetoothCollector(Collector):
    category = "Bluetooth"

    def __init__(self):
        self.sock = None
        try:
            self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_RAW,
                                      socket.BTPROTO_HCI)
        except (OSError, AttributeError):
            pass

    def collect(self):
        out = []
        for base in sorted(glob.glob("/sys/class/bluetooth/hci[0-9]*")):
            name = os.path.basename(base)
            if ":" in name:
                continue  # connection entries hci0:N
            dev_id = int(name[3:])
            conns = len(glob.glob(f"{base}:*")) + len(glob.glob(f"{base}/{name}:*"))
            rx = tx = None
            detail = f"{conns} connection{'s' if conns != 1 else ''}"
            if self.sock is not None:
                buf = bytearray(struct.calcsize(_HCI_FMT))
                struct.pack_into("<H", buf, 0, dev_id)
                try:
                    fcntl.ioctl(self.sock, _HCIGETDEVINFO, buf, True)
                    v = struct.unpack(_HCI_FMT, buf)
                    addr = ":".join(f"{b:02X}" for b in reversed(v[2]))
                    up = bool(v[3] & 1)
                    (err_rx, err_tx, cmd_tx, evt_rx, acl_tx, acl_rx,
                     sco_tx, sco_rx, rx, tx) = v[13:23]
                    detail = (f"{addr} · {'up' if up else 'down'} · {detail} · "
                              f"ACL {acl_rx}/{acl_tx} · SCO {sco_rx}/{sco_tx} pkts")
                except OSError:
                    pass
            out.append(Reading(f"bt:{name}", self.category, name, detail, rx, tx,
                               BYTES, usb_ancestor(f"{base}/device")))
        return out

    def close(self):
        if self.sock:
            self.sock.close()


# ---------------------------------------------------------------------- serial
_TTY_DRIVER_RE = re.compile(r"^\s*(\d+):.*?\btx:(\d+)\s+rx:(\d+)")


class SerialCollector(Collector):
    """Serial ports from /sys/class/tty. Per-port byte counters come from
    /proc/tty/driver/serial, which the kernel only exposes to root."""
    category = "Serial"

    def collect(self):
        counters: dict[str, tuple[int, int]] = {}
        text = _read("/proc/tty/driver/serial")
        for line in text.splitlines():
            m = _TTY_DRIVER_RE.match(line)
            if m:
                counters[f"ttyS{m.group(1)}"] = (int(m.group(3)), int(m.group(2)))
        readable = bool(text)

        out = []
        for base in sorted(glob.glob("/sys/class/tty/*")):
            name = os.path.basename(base)
            if not os.path.exists(f"{base}/device"):
                continue  # virtual consoles / ptys
            is_legacy = name.startswith("ttyS")
            if is_legacy and _read(f"{base}/type", "0") == "0":
                continue  # no UART behind this port
            usb = usb_ancestor(f"{base}/device")
            driver = os.path.basename(os.path.realpath(f"{base}/device/driver"))
            parts = [driver]
            if usb:
                parts.append(_usb_label(usb))
            irq = _read(f"{base}/irq")
            if irq and irq != "0":
                parts.append(f"irq {irq}")
            rx, tx = counters.get(name, (None, None))
            note = "" if rx is not None else ("needs root" if not readable and is_legacy
                                               else "no counters")
            out.append(Reading(f"tty:{name}", self.category, f"/dev/{name}",
                               " · ".join(p for p in parts if p), rx, tx, BYTES,
                               usb, note))
        return out


# ----------------------------------------------------------------------- input
_EV_SIZE = struct.calcsize("llHHi")  # struct input_event


class InputCollector(Collector):
    """Counts events from /dev/input/event* (needs the `input` group).
    Only event *counts* are kept; contents (e.g. keystrokes) are discarded."""
    category = "Input"

    def __init__(self):
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.fds: dict[int, str] = {}
        self.denied = False
        self._stop = threading.Event()
        self._rescan()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _rescan(self):
        known = set(self.fds.values())
        self.denied = False
        for path in glob.glob("/dev/input/event*"):
            if path in known:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                self.denied = True
                continue
            except OSError:
                continue
            self.fds[fd] = path
            with self.lock:
                self.counts.setdefault(path, 0)

    def _run(self):
        n = 0
        while not self._stop.is_set():
            fds = list(self.fds)
            try:
                ready, _, _ = select.select(fds, [], [], 0.5) if fds else ([], [], [])
            except (OSError, ValueError):
                ready = []
            if not fds:
                self._stop.wait(0.5)
            for fd in ready:
                path = self.fds.get(fd)
                try:
                    data = os.read(fd, _EV_SIZE * 64)
                except BlockingIOError:
                    continue
                except OSError:  # unplugged
                    os.close(fd)
                    self.fds.pop(fd, None)
                    continue
                events = 0
                for i in range(0, len(data) - _EV_SIZE + 1, _EV_SIZE):
                    ev_type = struct.unpack_from("H", data, i + _EV_SIZE - 8)[0]
                    if ev_type != 0:  # skip EV_SYN
                        events += 1
                with self.lock:
                    self.counts[path] = self.counts.get(path, 0) + events
            n += 1
            if n % 10 == 0:
                self._rescan()

    def collect(self):
        with self.lock:
            counts = dict(self.counts)
        live = set(self.fds.values())
        out = []
        for path in sorted(counts, key=lambda p: int(re.sub(r"\D", "", p) or 0)):
            if path not in live:
                continue
            ev = os.path.basename(path)
            sys_dev = f"/sys/class/input/{ev}/device"
            name = _read(f"{sys_dev}/name", ev)
            phys = _read(f"{sys_dev}/phys")
            bus = {"0003": "USB", "0005": "Bluetooth", "0011": "PS/2",
                   "0019": "platform", "0018": "I²C"}.get(
                _read(f"{sys_dev}/id/bustype"), "")
            detail = " · ".join(x for x in (ev, bus, phys) if x)
            out.append(Reading(f"in:{ev}", self.category, name, detail,
                               counts[path], None, EVENTS, usb_ancestor(sys_dev)))
        if self.denied and not out:
            out.append(Reading("in:denied", self.category, "Input devices",
                               "add yourself to the 'input' group", None, None,
                               EVENTS, note="permission denied"))
        return out

    def close(self):
        self._stop.set()
        for fd in list(self.fds):
            try:
                os.close(fd)
            except OSError:
                pass


# ------------------------------------------------------------------ interrupts
class InterruptCollector(Collector):
    category = "Interrupts"

    def collect(self):
        out = []
        with open("/proc/interrupts") as f:
            ncpu = len(f.readline().split())
            for line in f:
                label, _, rest = line.partition(":")
                parts = rest.split()
                counts = [int(x) for x in parts[:ncpu] if x.isdigit()]
                desc = " ".join(parts[len(counts):])
                label = label.strip()
                if label.isdigit():
                    # "IR-PCI-MSI 327680-edge xhci_hcd" → devices are the last field(s)
                    tokens = desc.split()
                    name = tokens[-1] if tokens else f"IRQ {label}"
                    detail = f"IRQ {label} · {' '.join(tokens[:-1])}"
                else:
                    name, detail = desc or label, f"{label} (per-CPU)"
                out.append(Reading(f"irq:{label}", self.category, name, detail,
                                   sum(counts), None, IRQS))
        return out


# ------------------------------------------------------------------------- USB
_USBMON_DIR = "/sys/kernel/debug/usb/usbmon"


class UsbCollector(Collector):
    """USB device tree from sysfs. Real bus traffic comes from usbmon when it
    is readable (root + `modprobe usbmon`); otherwise each device's traffic is
    the sum of the channels it hosts (NICs, disks, audio, BT, input)."""
    category = "USB"

    def __init__(self):
        self.lock = threading.Lock()
        self.bus_bytes: dict[tuple[int, int], list[int]] = {}
        self.usbmon = os.access(f"{_USBMON_DIR}/0u", os.R_OK)
        if self.usbmon:
            threading.Thread(target=self._usbmon_run, daemon=True).start()

    def _usbmon_run(self):
        # Text API lines: tag ts C Bi:1:005:2 0 64 = ...  (C = completion)
        with open(f"{_USBMON_DIR}/0u", "r", errors="replace") as f:
            for line in f:
                p = line.split()
                if len(p) < 6 or p[2] != "C":
                    continue
                try:
                    addr = p[3].split(":")
                    bus, dev = int(addr[1]), int(addr[2])
                    length = int(p[5])
                except (IndexError, ValueError):
                    continue
                idx = 0 if addr[0][1] == "i" else 1
                with self.lock:
                    self.bus_bytes.setdefault((bus, dev), [0, 0])[idx] += length

    def devices(self) -> dict[str, tuple[str, str]]:
        """realpath → (label, detail) for every USB device (not interfaces)."""
        devs = {}
        for base in glob.glob("/sys/bus/usb/devices/*"):
            if ":" in os.path.basename(base):
                continue
            real = os.path.realpath(base)
            speed = _read(f"{base}/speed")
            vid, pid = _read(f"{base}/idVendor"), _read(f"{base}/idProduct")
            detail = " · ".join(x for x in (
                f"{vid}:{pid}", f"bus {_read(f'{base}/busnum')} dev {_read(f'{base}/devnum')}",
                f"{speed} Mb/s" if speed else "") if x)
            devs[real] = (_usb_label(real), detail)
        return devs

    def collect(self, hosted: list[Reading] | None = None):
        hosted = hosted or []
        with self.lock:
            bus_bytes = {k: list(v) for k, v in self.bus_bytes.items()}
        out = []
        for real, (label, detail) in sorted(self.devices().items(),
                                            key=lambda kv: os.path.basename(kv[0])):
            children = [r for r in hosted if r.usb == real]
            is_hub = _read(f"{real}/bDeviceClass") == "09"
            if children:
                detail += " · hosts " + ", ".join(sorted({r.name for r in children})[:4])
            if self.usbmon:
                key = (int(_read(f"{real}/busnum", "0")), int(_read(f"{real}/devnum", "0")))
                rx, tx = bus_bytes.get(key, [0, 0])
                unit, note = BYTES, ""
            else:
                bytes_ch = [r for r in children if r.unit == BYTES]
                src = bytes_ch or children
                unit = src[0].unit if src else BYTES
                rx = sum(r.rx or 0 for r in src) if any(r.rx is not None for r in src) else None
                tx = sum(r.tx or 0 for r in src) if any(r.tx is not None for r in src) else None
                note = "" if src else ("hub" if is_hub else "traffic needs usbmon")
            out.append(Reading(f"usb:{os.path.basename(real)}", self.category, label,
                               detail, rx, tx, unit, real, note))
        return out


def _usb_label(real: str) -> str:
    product = _read(f"{real}/product")
    maker = _read(f"{real}/manufacturer")
    if product and maker and not product.startswith(maker):
        return f"{maker} {product}"
    if product or maker:
        return product or maker
    hub = _read(f"{real}/bDeviceClass") == "09"
    return f"{'USB hub' if hub else 'USB device'} {os.path.basename(real)}"


def default_collectors() -> list[Collector]:
    return [NetworkCollector(), UsbCollector(), SerialCollector(), AudioCollector(),
            BluetoothCollector(), InputCollector(), StorageCollector(),
            InterruptCollector()]
