"""Turns cumulative collector readings into per-channel rates and history."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .collectors import BYTES, Collector, Reading, UsbCollector, default_collectors
from .power import PowerMeter

HISTORY = 180  # samples kept per channel
# Categories whose byte rates are summed for "everything" totals. USB rows are
# aggregates of other channels and would double count.
BYTE_CATEGORIES = ("Network", "Serial", "Audio", "Bluetooth", "Storage")


@dataclass
class Channel:
    key: str
    category: str
    name: str = ""
    detail: str = ""
    unit: str = BYTES
    note: str = ""
    has_rx: bool = True
    has_tx: bool = True
    stacked: bool = False
    rx_rate: float = 0.0
    tx_rate: float = 0.0
    rx_total: int = 0  # accumulated since monitoring started
    tx_total: int = 0
    rx_hist: deque = field(default_factory=lambda: deque([0.0] * HISTORY, HISTORY))
    tx_hist: deque = field(default_factory=lambda: deque([0.0] * HISTORY, HISTORY))
    last: tuple = (None, None)
    seen: float = 0.0

    @property
    def active(self) -> bool:
        return self.rx_rate > 0 or self.tx_rate > 0

    @property
    def ever_active(self) -> bool:
        return self.rx_total > 0 or self.tx_total > 0


def _delta(prev, cur):
    if prev is None or cur is None:
        return 0
    return cur - prev if cur >= prev else 0  # counter reset (stream reopened)


class Monitor:
    def __init__(self, collectors: list[Collector] | None = None):
        self.collectors = collectors if collectors is not None else default_collectors()
        self.channels: dict[str, Channel] = {}
        self.errors: dict[str, str] = {}
        self.last_time: float | None = None
        self.categories = [c.category for c in self.collectors]
        self.power = PowerMeter()

    def sample(self) -> None:
        now = time.monotonic()
        readings: list[Reading] = []
        usb = None
        for c in self.collectors:
            if isinstance(c, UsbCollector):
                usb = c
                continue
            try:
                readings.extend(c.collect())
                self.errors.pop(c.category, None)
            except Exception as e:  # keep other sources alive
                self.errors[c.category] = f"{type(e).__name__}: {e}"
        if usb is not None:
            try:
                readings.extend(usb.collect(readings))
                self.errors.pop(usb.category, None)
            except Exception as e:
                self.errors[usb.category] = f"{type(e).__name__}: {e}"

        dt = (now - self.last_time) if self.last_time else None
        for r in readings:
            ch = self.channels.get(r.key)
            if ch is None:
                ch = self.channels[r.key] = Channel(r.key, r.category)
                ch.last = (r.rx, r.tx)
            ch.name, ch.detail, ch.unit, ch.note = r.name, r.detail, r.unit, r.note
            ch.has_rx, ch.has_tx = r.rx is not None, r.tx is not None
            ch.stacked = r.stacked
            drx, dtx = _delta(ch.last[0], r.rx), _delta(ch.last[1], r.tx)
            ch.last = (r.rx, r.tx)
            ch.rx_total += drx
            ch.tx_total += dtx
            ch.rx_rate = drx / dt if dt else 0.0
            ch.tx_rate = dtx / dt if dt else 0.0
            ch.rx_hist.append(ch.rx_rate)
            ch.tx_hist.append(ch.tx_rate)
            ch.seen = now
        # drop channels whose device disappeared
        for key in [k for k, c in self.channels.items() if c.seen != now]:
            del self.channels[key]
        self.last_time = now
        self.power.sample()

    def category_totals(self, category: str) -> tuple[float, float, str]:
        chans = [c for c in self.channels.values()
                 if c.category == category and not c.stacked]
        # USB rows aggregate other channels, so their sum is still per-category
        unit = chans[0].unit if chans else BYTES
        same = [c for c in chans if c.unit == unit]
        return (sum(c.rx_rate for c in same), sum(c.tx_rate for c in same), unit)

    def close(self):
        for c in self.collectors:
            c.close()
        self.power.close()


def fmt_rate(v: float, unit: str) -> str:
    if unit != BYTES:
        return f"{v:,.0f} {unit}/s" if v >= 10 or v == 0 else f"{v:.1f} {unit}/s"
    return fmt_bytes(v) + "/s"


def fmt_bytes(v: float) -> str:
    for u in ("B", "kB", "MB", "GB", "TB"):
        if abs(v) < 1000 or u == "TB":
            return f"{v:.0f} {u}" if u == "B" else f"{v:.1f} {u}"
        v /= 1000
    return f"{v:.1f} TB"
