"""3D view of the inside of the machine: every channel is a trace routed across
the motherboard from its device to a controller, controllers feed the chipset
over bus traces, and the chipset feeds the CPU. Two-way links are in/out
differential pairs. Trace color is a log-scale heat map of the traffic volume;
pulses run along each trace (toward the CPU = in, away from it = out).

Rendered with NVIDIA OptiX (ray-traced parts, glowing pipes, AI denoiser) when
the native renderer in flowscope/optix is built and an RTX-class GPU is present;
otherwise in software with Cairo (perspective projection + painter's sort).
Labels, legend and tooltips are always drawn with Cairo on top.
"""
from __future__ import annotations

import math
import sys
import time
from array import array
from dataclasses import dataclass, field

import cairo

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gtk, PangoCairo  # noqa: E402

from . import routing  # noqa: E402
from .collectors import BYTES  # noqa: E402
from .monitor import BYTE_CATEGORIES, Channel, Monitor, fmt_rate  # noqa: E402
from .power import Power, fmt_watts  # noqa: E402

Vec = tuple[float, float, float]

# Heat scale: log10 range of the rate that maps onto the ramp, per unit.
HEAT_RANGE = {BYTES: (2.0, 9.0)}  # 100 B/s … 1 GB/s
EVENT_RANGE = (0.0, 5.0)  # 1 … 100k events or IRQs per second

# Volume ramp: navy → blue → azure → ice → white, rising in lightness so it
# stays a sequential scale. Same stops as heat_color() in optix/src/device.cu.
HEAT_STOPS = ("#0a1f5c", "#1f5fe0", "#2ea1f5", "#8fdbff", "#effaff")
# Power layer: a separate red sequential ramp on a log watts scale, same stops
# as power_color() in optix/src/device.cu (dark) and a light-surface variant.
POWER_STOPS = {"dark": ("#3d0707", "#8f1111", "#d42424", "#ff6e6e", "#ffd6d6"),
               "light": ("#fbd0d0", "#e34948", "#a31c1c", "#5c0a0a")}
POWER_RANGE = (-1.0, 2.5)  # log10 watts: 0.1 W … 316 W
# Neutral grays for all the hardware so the heat is the only color in the
# Cairo scene.
THEME = {
    "dark": {
        "bg": (0.102, 0.102, 0.098), "ink": (1.0, 1.0, 1.0),
        "board": (0.17, 0.176, 0.165), "chip": (0.26, 0.265, 0.25),
        "port": (0.42, 0.42, 0.40), "m2": (0.30, 0.30, 0.285), "gpu": (0.20, 0.20, 0.21),
        "ram": (0.22, 0.225, 0.21), "idle": (0.30, 0.30, 0.285),
        "ramp": HEAT_STOPS, "outline": (0.0, 0.0, 0.0),
        "dot": (1.0, 1.0, 1.0), "dot_ring": (0.08, 0.08, 0.08),
    },
    "light": {
        "bg": (0.988, 0.988, 0.984), "ink": (0.043, 0.043, 0.043),
        "board": (0.84, 0.835, 0.81), "chip": (0.70, 0.695, 0.67),
        "port": (0.56, 0.555, 0.53), "m2": (0.76, 0.755, 0.73), "gpu": (0.62, 0.62, 0.60),
        "ram": (0.74, 0.735, 0.71), "idle": (0.80, 0.795, 0.77),
        "ramp": ("#cde2fb", "#5598e7", "#1c5cab", "#0d366b"), "outline": (0.32, 0.32, 0.30),
        "dot": (0.05, 0.05, 0.05), "dot_ring": (1.0, 1.0, 1.0),
    },
}


def _hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def heat(rate: float, unit: str) -> float:
    """0‥1 position of a rate on the log heat scale (0 also for no traffic)."""
    if rate <= 0:
        return 0.0
    lo, hi = HEAT_RANGE.get(unit, EVENT_RANGE)
    return min(max((math.log10(rate) - lo) / (hi - lo), 0.0), 1.0)


def power_heat(watts: float | None) -> float:
    if not watts or watts <= 0:
        return 0.0
    lo, hi = POWER_RANGE
    return min(max((math.log10(watts) - lo) / (hi - lo), 0.0), 1.0)


def ramp(t: float, mode: str, stops=None) -> tuple[float, float, float]:
    stops = [_hex(s) for s in (stops or THEME[mode]["ramp"])]
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a, b = stops[i], stops[i + 1]
    return tuple(a[k] + (b[k] - a[k]) * f for k in range(3))


# ---------------------------------------------------------------- geometry
def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec) -> Vec:
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


_LIGHT = _norm((-0.35, 1.0, 0.45))
_FACES = (  # corner indices (counter-clockwise from outside) + outward normal
    ((4, 5, 6, 7), (0, 1, 0)), ((0, 3, 2, 1), (0, -1, 0)),
    ((0, 1, 5, 4), (0, 0, -1)), ((2, 3, 7, 6), (0, 0, 1)),
    ((1, 2, 6, 5), (1, 0, 0)), ((0, 4, 7, 3), (-1, 0, 0)),
)


@dataclass
class Box:
    lo: Vec
    hi: Vec
    color: str  # THEME key
    label: str = ""
    part: str = ""  # hardware.PARTS builder, e.g. "ctrl:Network"
    pipe: int = -1  # index of the pipe whose heat this part shows
    index: int = 0  # position among its siblings
    power: str = ""  # power reading key (power.py) shown by the power layer
    power_slot: int = -1  # its entry in the renderer's per-pipe buffer

    @property
    def top(self) -> Vec:
        return ((self.lo[0] + self.hi[0]) / 2, self.hi[1], (self.lo[2] + self.hi[2]) / 2)

    def corners(self) -> list[Vec]:
        (x0, y0, z0), (x1, y1, z1) = self.lo, self.hi
        return [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
                (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]


def _box(center_xz: tuple[float, float], size: Vec, color: str, y0: float = 0.0,
         label: str = "", part: str = "") -> Box:
    x, z = center_xz
    sx, sy, sz = size
    return Box((x - sx / 2, y0, z - sz / 2), (x + sx / 2, y0 + sy, z + sz / 2), color, label,
               part)


@dataclass
class Pipe:
    """A board trace from `a` (device end) to `b` (CPU end). Links that carry
    both directions are a differential pair: one trace for in, one for out."""
    key: str
    category: str
    label: str
    a: Vec
    b: Vec
    radius: float  # half-width of one trace
    channel: str | None = None  # monitor channel key, for selection
    pair: bool = True
    rx: float = 0.0
    tx: float = 0.0
    unit: str = BYTES
    detail: str = ""
    route: list[tuple[float, float]] = field(default_factory=list)  # centreline (x, z)
    points: list[Vec] = field(default_factory=list)
    cum: list[float] = field(default_factory=list)
    traces: list[tuple[list[Vec], int]] = field(default_factory=list)  # (points, TRACE_*)
    phase_in: float = 0.0
    phase_out: float = 0.0

    def build(self, route: list[tuple[float, float]]) -> None:
        self.route = route = routing.chamfer(route, 0.35)
        y = self.radius * 0.3  # a raised bead on the PCB surface
        self.points = [(x, y, z) for x, z in route]
        self.cum = [0.0]
        for i in range(1, len(self.points)):
            self.cum.append(self.cum[-1] + math.dist(self.points[i - 1], self.points[i]))
        if self.pair:
            gap = self.radius * 1.6
            self.traces = [([(x, y, z) for x, z in routing.offset(route, d)], kind)
                           for d, kind in ((gap, TRACE_IN), (-gap, TRACE_OUT))]
        else:
            self.traces = [(self.points, TRACE_BOTH)]

    @property
    def width(self) -> float:
        """Total width of the trace or pair, in cm."""
        return self.radius * (5.2 if self.pair else 2.0)

    def at(self, s: float, pts: list[Vec] | None = None) -> Vec:
        """Point at fraction `s` of the length (0 = device, 1 = CPU) along
        the centreline, or along `pts`, one of this pipe's parallel traces."""
        pts = pts or self.points
        target = s * self.cum[-1]
        for i in range(1, len(self.cum)):
            if self.cum[i] >= target:
                seg = self.cum[i] - self.cum[i - 1] or 1.0
                f = (target - self.cum[i - 1]) / seg
                a, b = pts[i - 1], pts[i]
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                        a[2] + (b[2] - a[2]) * f)
        return pts[-1]

    def rate(self, kind: int) -> float:
        return self.rx if kind == TRACE_IN else self.tx if kind == TRACE_OUT else self.rx + self.tx


TRACE_IN, TRACE_OUT, TRACE_BOTH = 0, 1, 2


# ------------------------------------------------------------------ layout
# Controller position (x, z) and package size on the board, where its devices
# sit (first slot, step between slots), which side of the controller faces
# them, and how many get their own trace. Board spans x −16‥16 (rear I/O on
# the left edge), z −12‥12.
LAYOUT = {
    "Network":    dict(ctrl=(-12, -8), chip=(1.2, 0.2, 1.2), first=(-16.3, -11), step=(0, 1.7),
                       max=5, box=(1.6, 1.35, 1.5), side="left", label="NIC"),
    "USB":        dict(ctrl=(-11, -1.6), chip=(1.2, 0.2, 1.2), first=(-16.3, -2.5),
                       step=(0, 1.45), max=6, box=(1.6, 0.75, 1.35), side="left",
                       label="USB controller"),
    "Audio":      dict(ctrl=(-11, 7.5), chip=(1.8, 0.35, 1.8), first=(-16.3, 6.8),
                       step=(0, 1.3), max=4, box=(1.6, 1.0, 1.0), side="left",
                       label="Audio codec"),
    "Storage":    dict(ctrl=(7.0, 3.6), chip=(1.2, 0.2, 1.2), first=(11.6, -6.4), step=(0, 2.6),
                       max=4, box=(4.2, 0.35, 2.2), side="right", kind="m2",
                       label="Storage controller"),
    "Bluetooth":  dict(ctrl=(10, -8.6), chip=(1.6, 0.6, 1.8), first=(8.5, -11.4),
                       step=(1.6, 0), max=3, box=(0.7, 0.45, 0.7), side="top",
                       label="Bluetooth"),
    "Serial":     dict(ctrl=(-6.5, 8.8), chip=(1.4, 0.18, 1.4), first=(-9.5, 11.5),
                       step=(1.5, 0), max=4, box=(1.3, 0.8, 0.55), side="bottom",
                       label="Serial (Super I/O)"),
    "Input":      dict(ctrl=(1.5, 8.8), chip=(1.0, 0.18, 1.0), first=(-1.5, 11.5), step=(1.5, 0),
                       max=6, box=(1.3, 0.8, 0.55), side="bottom", label="Input"),
    "Interrupts": dict(ctrl=(-8.2, -8.6), chip=(1.2, 0.18, 1.2), first=(-9.5, -11.4),
                       step=(1.3, 0), max=6, box=(0.8, 0.15, 0.5), side="top",
                       label="Interrupt controller"),
}
# Parts from hardware.py that traces should avoid when choosing a route
# (socket, VRM heatsinks, PCIe slots, M.2 heatsink, ATX connector).
KEEPOUT = [(-2.0, -3.0, 3.0, 3.0), (-7.0, -3.0, 0.5, 3.1), (-1.8, -7.6, 3.0, 0.45),
           (-9.5, 2.2, 4.5, 0.4), (-12.5, 4.4, 1.3, 0.4), (-5.4, 4.8, 4.2, 1.0),
           (15.2, -1.5, 0.5, 2.7), (-10.3, 3.05, 5.0, 0.95)]
# The graphics card standing in the x16 slot: footprint centre, size, base.
GPU_XZ, GPU_SIZE, GPU_Y0 = (-10.3, 3.05), (10.0, 3.5, 1.9), 0.3
TRUNK_PITCH, LEAF_PITCH, SINGLE_PITCH = 0.85, 0.5, 0.3
CPU_XZ, PCH_XZ = (-2.0, -3.0), (3.0, 6.0)


def _rect(box: Box) -> routing.Rect:
    (x0, _, z0), (x1, _, z1) = box.lo, box.hi
    return ((x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (z1 - z0) / 2)


def _rank(ch: Channel) -> tuple:
    return (ch.ever_active, ch.rx_total + ch.tx_total, -len(ch.name))


class FlowScene(Gtk.DrawingArea):
    """Orbitable 3D view. Call `update(monitor)` after each sample."""

    def __init__(self, renderer: str = "auto"):
        super().__init__(hexpand=True, vexpand=True)
        self.renderer = renderer  # "auto", "optix" or "cairo"
        self.rt = None  # optix_backend.Renderer once loaded
        self.rt_status = ""
        self._rt_tried = False
        self._rt_dirty = True
        self._t0 = time.monotonic()
        self.yaw, self.pitch, self.dist = math.radians(-28), math.radians(56), 64.0
        self.target: Vec = (2.5, 0.0, 1.5)
        self.boxes: list[Box] = []
        self.pipes: list[Pipe] = []
        self.signature: tuple = ()
        self.category = "All"
        self.selected: str | None = None
        self.on_select = None  # callback(channel_key)
        self.hover: Pipe | None = None
        self.hover_part: Box | None = None  # powered part under the mouse
        self.power_layer = False
        self.power: dict[str, Power] = {}
        self.power_keys: list[str] = []  # renderer slots after the pipes
        self._more_usb: list[str] = []  # USB devices folded into "+N more"
        self.mouse: tuple[float, float] | None = None
        self._hit: list[tuple[Pipe, list, float, float]] = []
        self._last_frame: int | None = None
        self._drag_start = (0.0, 0.0, 0.0, 0.0)
        self._dragged = False

        self.set_draw_func(self._on_draw)
        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._drag_begin)
        drag.connect("drag-update", self._drag_update)
        self.add_controller(drag)
        click = Gtk.GestureClick()
        click.connect("released", self._click)
        self.add_controller(click)
        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self._scroll)
        self.add_controller(scroll)
        motion = Gtk.EventControllerMotion()
        motion.connect("motion", lambda _c, x, y: self._motion(x, y))
        motion.connect("leave", lambda _c: self._motion(None, None))
        self.add_controller(motion)
        self.add_tick_callback(self._frame)

    # ------------------------------------------------------------ data
    def set_category(self, category: str) -> None:
        self.category = category
        self.queue_draw()

    def set_power_layer(self, on: bool) -> None:
        self.power_layer = on
        if not on:
            self.hover_part = None
        self.queue_draw()

    def power_of(self, key: str) -> Power | None:
        """Reading for a power key; "more:USB" sums the folded-in devices."""
        if key == "more:USB":
            parts = [self.power[k] for k in self._more_usb if k in self.power]
            if not parts:
                return None
            return Power(sum(p.watts or 0 for p in parts), "USB descriptors",
                         f"{len(parts)} devices' declared maximums combined", estimated=True)
        return self.power.get(key)

    def set_selected(self, key: str | None) -> None:
        self.selected = key
        self.queue_draw()

    def update(self, mon: Monitor) -> None:
        chans = mon.channels
        self.power = mon.power.readings
        chosen: dict[str, list[Channel]] = {}
        rest: dict[str, list[Channel]] = {}
        for cat, spec in LAYOUT.items():
            pool = sorted((c for c in chans.values() if c.category == cat and not c.stacked),
                          key=_rank, reverse=True)
            n = spec["max"] if len(pool) <= spec["max"] else spec["max"] - 1
            chosen[cat] = sorted(pool[:n], key=lambda c: c.name.lower())
            rest[cat] = pool[n:]
        sig = tuple((cat, tuple(c.key for c in chosen[cat]), bool(rest[cat]))
                    for cat in LAYOUT)
        if sig != self.signature:
            self._build(chosen, rest)
            self.signature = sig

        totals = {cat: mon.category_totals(cat) for cat in mon.categories}
        byte_rx = sum(totals[c][0] for c in BYTE_CATEGORIES if c in totals)
        byte_tx = sum(totals[c][1] for c in BYTE_CATEGORIES if c in totals)
        for p in self.pipes:
            if p.channel is not None and p.channel in chans:
                ch = chans[p.channel]
                p.rx, p.tx, p.unit = ch.rx_rate, ch.tx_rate, ch.unit
                p.detail = f"{ch.category} · {ch.detail}"
            elif p.key.startswith("more:"):
                others = rest[p.category]
                p.rx = sum(c.rx_rate for c in others)
                p.tx = sum(c.tx_rate for c in others)
                p.unit = others[0].unit if others else BYTES
                p.detail = f"{len(others)} more {p.category.lower()} channels combined"
            elif p.key.startswith("trunk:"):
                rx, tx, unit = totals.get(p.category, (0.0, 0.0, BYTES))
                p.rx, p.tx, p.unit = rx, tx, unit
            elif p.key == "cpu":
                p.rx, p.tx, p.unit = byte_rx, byte_tx, BYTES
        self.queue_draw()

    def _build(self, chosen, rest) -> None:
        cpu = _box(CPU_XZ, (4.6, 0.75, 4.6), "chip", label="CPU", part="cpu")
        pch = _box(PCH_XZ, (2.8, 1.1, 2.8), "chip", label="Chipset", part="pch")
        cpu.pipe = pch.pipe = 0
        self.boxes = [cpu, pch]
        for i in range(4):  # memory, for scale
            self.boxes.append(_box((3.2 + i * 0.9, -3.0), (0.55, 3.2, 10.0), "ram", part="ram"))
        cpu.power = "cpu"
        gpu = _box(GPU_XZ, GPU_SIZE, "gpu", GPU_Y0, label="GPU", part="gpu")
        gpu.power = "gpu"
        self.boxes.append(gpu)
        self._more_usb = [c.key for c in rest.get("USB", [])]
        self.pipes = [Pipe("cpu", "All", "Chipset ⇄ CPU (all byte channels)",
                           pch.top, cpu.top, 0.14, detail=" + ".join(BYTE_CATEGORIES))]
        cpu_rect = KEEPOUT[0]  # traces stop at the socket frame
        trunks = [(self.pipes[0], cpu_rect, _rect(pch), True)]  # routed CPU → chipset
        for cat, spec in LAYOUT.items():
            ctrl = _box(spec["ctrl"], spec["chip"], "chip", label=spec["label"],
                        part=f"ctrl:{cat}")
            ctrl.pipe = len(self.pipes)
            self.boxes.append(ctrl)
            hub, what = (cpu_rect, "CPU") if cat == "Interrupts" else (_rect(pch), "chipset")
            trunk = Pipe(f"trunk:{cat}", cat, f"{spec['label']} ⇄ {what}", ctrl.top,
                         cpu.top if cat == "Interrupts" else pch.top, 0.1,
                         pair=cat not in ("Input", "Interrupts"),
                         detail=f"Sum of all {cat.lower()} channels")
            self.pipes.append(trunk)
            trunks.append((trunk, _rect(ctrl), hub, False))

            leaves = [(c.key, c.name, c.key, c.has_tx) for c in chosen[cat]]
            if rest[cat]:
                leaves.append((f"more:{cat}", f"+{len(rest[cat])} more", None,
                               any(c.has_tx for c in rest[cat])))
            fx, fz = spec["first"]
            dx, dz = spec["step"]
            devices, pipes = [], []
            for i, (key, name, chan, pair) in enumerate(leaves):
                dev = _box((fx + dx * i, fz + dz * i), spec["box"], spec.get("kind", "port"),
                           0.0, name, part=f"leaf:{cat}")
                dev.pipe, dev.index = len(self.pipes), i
                if cat == "USB":
                    dev.power = key  # usb:<dev> or more:USB
                self.boxes.append(dev)
                pipe = Pipe(key, cat, name, dev.top, ctrl.top, 0.07, chan, pair=pair)
                self.pipes.append(pipe)
                devices.append(_rect(dev))
                pipes.append(pipe)
            pitch = LEAF_PITCH if any(p.pair for p in pipes) else SINGLE_PITCH
            for pipe, route in zip(pipes, routing.fan(_rect(ctrl), spec["side"], devices,
                                                      pitch, 0.35)):
                pipe.build(route)
        self._route_trunks(trunks)
        # power readings ride in the renderer's per-pipe buffer, after the pipes
        self.power_keys = []
        for box in self.boxes:
            if box.power:
                box.power_slot = len(self.pipes) + len(self.power_keys)
                self.power_keys.append(box.power)
        self._rt_dirty = True

    def _route_trunks(self, trunks) -> None:
        """Bring every controller's bus into the chipset (or CPU), each on the
        side of the hub that keeps it clear of other parts, fanned per side."""
        obstacles = [_rect(b) for b in self.boxes] + KEEPOUT
        groups: dict[tuple, list] = {}
        for pipe, src, hub, reverse in trunks:
            others = [r for r in obstacles if r not in (src, hub)]
            best = None
            for horizontal in (True, False):
                a = 0 if horizontal else 1
                gap = abs(src[a] - hub[a]) - src[2 + a] - hub[2 + a]
                if gap < 1.2:
                    continue
                side = routing.facing_side(hub, src, horizontal)
                route = routing.fan(hub, side, [src], TRUNK_PITCH, 0.8)[0]
                score = (routing.blocked(route, others), -gap)
                if best is None or score < best[0]:
                    best = (score, side)
            side = best[1] if best else routing.facing_side(hub, src, True)
            groups.setdefault((hub, side), []).append((pipe, src, reverse))
        for (hub, side), members in groups.items():
            routes = routing.fan(hub, side, [m[1] for m in members], TRUNK_PITCH, 0.8)
            for (pipe, _src, reverse), route in zip(members, routes):
                pipe.build(route[::-1] if reverse else route)

    # ------------------------------------------------------------ input
    def _drag_begin(self, _g, x, y):
        self._drag_start = (x, y, self.yaw, self.pitch)
        self._dragged = False

    def _drag_update(self, _g, dx, dy):
        if abs(dx) + abs(dy) > 3:
            self._dragged = True
        _, _, yaw, pitch = self._drag_start
        self.yaw = yaw - dx * 0.008
        self.pitch = min(max(pitch + dy * 0.008, math.radians(8)), math.radians(88))
        self.queue_draw()

    def _scroll(self, _c, _dx, dy):
        self.dist = min(max(self.dist * (1.1 ** dy), 26.0), 140.0)
        self.queue_draw()
        return True

    def _motion(self, x, y):
        self.mouse = None if x is None else (x, y)
        self.hover = self._pick(x, y) if x is not None else None
        self.hover_part = (self._pick_part(x, y) if x is not None and self.hover is None
                           and self.power_layer else None)
        self.set_cursor_from_name("pointer" if self.hover and self.hover.channel else None)
        self.queue_draw()

    def _pick_part(self, x, y) -> Box | None:
        """The nearest powered part whose screen outline contains (x, y)."""
        w, h = self.get_width(), self.get_height()
        _, project, _ = self._camera(w, h)
        best, best_z = None, 1e9
        for box in self.boxes:
            if not box.power:
                continue
            pts = [project(c) for c in box.corners()]
            # convex hull test via the projected bounding polygon of all corners
            hull = _hull([(p[0], p[1]) for p in pts])
            if _inside(hull, x, y):
                z = sum(p[2] for p in pts) / 8
                if z < best_z:
                    best, best_z = box, z
        return best

    def _click(self, _g, _n, x, y):
        if self._dragged:
            return
        p = self._pick(x, y)
        key = p.channel if p else None
        self.selected = key
        if self.on_select:
            self.on_select(key)
        self.queue_draw()

    def _pick(self, x, y) -> Pipe | None:
        best, best_d = None, 1e9
        for pipe, pts, width, depth in self._hit:
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                vx, vy = bx - ax, by - ay
                ll = vx * vx + vy * vy or 1.0
                t = min(max(((x - ax) * vx + (y - ay) * vy) / ll, 0.0), 1.0)
                d = math.hypot(x - ax - vx * t, y - ay - vy * t) - width / 2
                d += depth * 1e-3  # prefer the nearer pipe on ties
                if d < best_d:
                    best, best_d = pipe, d
        return best if best_d < 6 else None

    # --------------------------------------------------------- animation
    def _frame(self, _w, clock) -> bool:
        now = clock.get_frame_time()
        if self._last_frame is None:
            self._last_frame = now
            return True
        dt = (now - self._last_frame) / 1e6
        if dt < 1 / 40:  # ~40 fps is plenty and keeps Python busy less
            return True
        self._last_frame = now
        dt = min(dt, 0.2)
        for p in self.pipes:
            length = p.cum[-1] if p.cum else 1.0
            for attr, rate in (("phase_in", p.rx), ("phase_out", p.tx)):
                if rate > 0:
                    speed = (2.0 + 14.0 * heat(rate, p.unit)) / length  # units/s
                    setattr(p, attr, (getattr(p, attr) + speed * dt) % 1.0)
        self.queue_draw()
        return True

    # ------------------------------------------------------------ render
    def _mode(self) -> str:
        fg = self.get_color()
        return "dark" if (fg.red + fg.green + fg.blue) / 3 > 0.5 else "light"

    def _on_draw(self, _area, cr, w, h):
        self.render(cr, w, h, self._mode(), self.get_scale_factor())

    def close(self) -> None:
        if self.rt is not None:
            self.rt.close()
            self.rt = None

    def _optix(self):
        """The OptiX renderer, loading it on first use (None → use Cairo)."""
        if not self._rt_tried:
            self._rt_tried = True
            if self.renderer == "cairo":
                self.rt_status = "Cairo renderer (--renderer cairo)"
            else:
                from . import optix_backend
                self.rt, reason = optix_backend.load()
                if self.rt is None:
                    self.rt_status = f"Cairo renderer: {reason}"
                    if self.renderer == "optix":
                        print(f"flowscope: {reason}; using Cairo", file=sys.stderr)
                else:
                    self.rt_status = f"Ray-traced with OptiX on {self.rt.device}"
        return self.rt

    def _camera(self, w, h):
        cp = math.cos(self.pitch)
        cam = (self.target[0] + self.dist * cp * math.sin(self.yaw),
               self.target[1] + self.dist * math.sin(self.pitch),
               self.target[2] + self.dist * cp * math.cos(self.yaw))
        f = _norm(_sub(self.target, cam))
        r = _norm(_cross(f, (0.0, 1.0, 0.0)))
        u = _cross(r, f)
        focal = min(w, h) * 1.9
        cx, cy = w / 2, h / 2 - h * 0.02

        def project(p: Vec):
            d = _sub(p, cam)
            z = max(_dot(d, f), 0.5)
            return (cx + _dot(d, r) * focal / z, cy - _dot(d, u) * focal / z, z)

        project.basis = (cam, f, r, u, cx, cy)
        return cam, project, focal

    def render(self, cr, w, h, mode: str, scale: int = 1) -> None:
        if self.pipes and self._optix() is not None:
            try:
                self._render_optix(cr, w, h, scale)
                mode = "dark"  # the ray-traced studio is always dark
            except RuntimeError as e:
                print(f"flowscope: OptiX render failed ({e}); using Cairo", file=sys.stderr)
                self.close()
                self.rt_status = f"Cairo renderer: OptiX failed ({e})"
                self._render_cairo(cr, w, h, mode)
        else:
            self._render_cairo(cr, w, h, mode)
        th = THEME[mode]
        _, project, _ = self._camera(w, h)
        self._labels(cr, project, th)
        self._legend(cr, w, h, th, mode)
        if (self.hover is not None or self.hover_part is not None) and self.mouse is not None:
            self._tooltip(cr, w, h, th, mode)

    def _pipe_flags(self, p: Pipe) -> int:
        from . import optix_backend as ob
        flags = 0
        if p is self.hover:
            flags |= ob.HOVER
        if p.channel is not None and p.channel == self.selected:
            flags |= ob.SELECTED
        if (self.category != "All" and p.category not in (self.category, "All")
                and not flags):
            flags |= ob.FADED
        return flags

    def _render_optix(self, cr, w, h, scale) -> None:
        if self._rt_dirty:
            from . import hardware
            mesh = hardware.build(self)
            # every trace is a run of straight curve segments; each segment knows
            # its pipe, which trace of the pair it is, and where it sits (0‥1)
            verts, index, info = array("f"), array("I"), array("f")
            for i, p in enumerate(self.pipes):
                length = p.cum[-1] or 1.0
                for pts, kind in p.traces:
                    base = len(verts) // 4
                    for q in pts:
                        verts.extend([*q, p.radius])
                    for j in range(len(pts) - 1):
                        index.append(base + j)
                        info.extend([i, kind, p.cum[j] / length, p.cum[j + 1] / length])
            self.rt.set_scene(mesh.verts, mesh.tri_mat, mesh.materials, verts, index, info)
            self._rt_dirty = False
        data = array("f")
        for p in self.pipes:
            h_in, h_out = heat(p.rx, p.unit), heat(p.tx, p.unit)
            length = p.cum[-1]
            data.extend([heat(p.rx + p.tx, p.unit), h_in, h_out, p.radius,
                         p.phase_in, p.phase_out,
                         max(2, int(length / (3.2 - 2.2 * h_in))) if p.rx > 0 else 0,
                         max(2, int(length / (3.2 - 2.2 * h_out))) if p.tx > 0 else 0,
                         self._pipe_flags(p), length, 0.0, 0.0])
        from . import optix_backend as ob
        for key in self.power_keys:
            pw = self.power_of(key) if self.power_layer else None
            data.extend([power_heat(pw.watts if pw else None), 0, 0, 0, 0, 0, 0, 0,
                         ob.ESTIMATE if pw and pw.estimated else 0, 0, 0, 0])
        self.rt.set_pipes(data)

        _, project, focal = self._camera(w, h)
        cam, f, r, u, cx, cy = project.basis
        pw, ph = max(int(w * scale), 1), max(int(h * scale), 1)
        pixels = self.rt.render([*cam, focal * scale, *f, cx * scale, *r, cy * scale, *u,
                                 time.monotonic() - self._t0], pw, ph)
        surface = cairo.ImageSurface.create_for_data(pixels, cairo.FORMAT_RGB24, pw, ph, pw * 4)
        cr.save()
        cr.scale(1 / scale, 1 / scale)
        cr.set_source_surface(surface, 0, 0)
        cr.paint()
        cr.restore()

        self._hit = []
        for p in self.pipes:
            self._add_hit(p, project, focal)
            if p is self.hover or (p.channel is not None and p.channel == self.selected):
                self._outline(cr, p, project, (1.0, 1.0, 1.0, 0.85), 1.5)

    def _render_cairo(self, cr, w, h, mode: str) -> None:
        th = THEME[mode]
        cr.set_source_rgb(*th["bg"])
        cr.paint()
        if not self.pipes:
            return
        cam, project, focal = self._camera(w, h)
        dim = self.category != "All"

        # board: always underneath everything, so drawn first
        board = [project(p) for p in ((-16, 0, -12), (16, 0, -12), (16, 0, 12), (-16, 0, 12))]
        self._poly(cr, board, th["board"], th["outline"], 0.6)
        # traces lie on the board, so they go under every part too
        self._hit = []
        for p in self.pipes:
            self._pipe(cr, p, project, focal, th, mode, dim)

        items = []  # (depth, face corners, color)
        for box in self.boxes:
            corners = box.corners()
            pc = [project(c) for c in corners]
            for idx, n in _FACES:
                centre = tuple(sum(corners[i][k] for i in idx) / 4 for k in range(3))
                if _dot(n, _sub(cam, centre)) <= 0:
                    continue  # back face
                shade = 0.55 + 0.45 * max(0.0, _dot(n, _LIGHT))
                base = th[box.color]
                pw = self.power_of(box.power) if self.power_layer and box.power else None
                if pw and pw.watts:
                    tint = ramp(power_heat(pw.watts), mode, POWER_STOPS[mode])
                    k = 0.45 if pw.estimated else 0.85
                    base = tuple(b * (1 - k) + t * k for b, t in zip(base, tint))
                color = tuple(c * shade for c in base)
                depth = sum(pc[i][2] for i in idx) / 4
                items.append((depth, [pc[i] for i in idx], color))
        items.sort(key=lambda it: -it[0])
        for _depth, corners, color in items:
            self._poly(cr, corners, color, th["outline"], 0.35)

    @staticmethod
    def _poly(cr, pts, fill, outline, alpha):
        cr.move_to(pts[0][0], pts[0][1])
        for x, y, _ in pts[1:]:
            cr.line_to(x, y)
        cr.close_path()
        cr.set_source_rgb(*fill)
        cr.fill_preserve()
        cr.set_source_rgba(*outline, alpha)
        cr.set_line_width(1)
        cr.stroke()

    def _add_hit(self, p: Pipe, project, focal) -> None:
        pts = [project(q) for q in p.points]
        width = max(p.width * focal / q[2] for q in pts)
        self._hit.append((p, [(x, y) for x, y, _ in pts], width,
                          sum(q[2] for q in pts) / len(pts)))

    def _outline(self, cr, p: Pipe, project, rgba, width) -> None:
        """Hover / selection highlight drawn over a pipe's traces."""
        cr.set_source_rgba(*rgba)
        cr.set_line_width(width)
        cr.set_line_join(1)
        for pts3, _kind in p.traces:
            pts = [project(q) for q in pts3]
            cr.move_to(pts[0][0], pts[0][1])
            for x, y, _ in pts[1:]:
                cr.line_to(x, y)
            cr.stroke()

    def _pipe(self, cr, p: Pipe, project, focal, th, mode, dim):
        self._add_hit(p, project, focal)
        highlight = p.channel is not None and p.channel == self.selected
        hovered = p is self.hover
        faded = dim and p.category not in (self.category, "All") and not (highlight or hovered)
        alpha = 0.22 if faded else 1.0
        cr.set_line_cap(1)  # round
        cr.set_line_join(1)
        if highlight or hovered:
            for pts3, _kind in p.traces:
                pts = [project(q) for q in pts3]
                wd = max(2.0 * p.radius * focal / pts[0][2], 1.0)
                self._outline(cr, p, project, (*th["ink"], 0.9), wd + (5.0 if highlight else 3.5))
                break
        for pts3, kind in p.traces:
            pts = [project(q) for q in pts3]
            rate = p.rate(kind)
            color = ramp(heat(rate, p.unit), mode) if rate > 0 else th["idle"]
            for rgb, extra, a in ((th["outline"], 1.5, 0.5), (color, 0.0, 1.0)):
                cr.set_source_rgba(*rgb, a * alpha)
                for (ax, ay, az), (bx, by, bz) in zip(pts, pts[1:]):
                    cr.set_line_width(max(2.0 * p.radius * focal / ((az + bz) / 2), 1.0) + extra)
                    cr.move_to(ax, ay)
                    cr.line_to(bx, by)
                    cr.stroke()
            if faded or rate <= 0:
                continue
            # pulses: toward the CPU on the in trace, away from it on the out trace
            ht = heat(rate, p.unit)
            n = max(2, int(p.cum[-1] / (3.2 - 2.2 * ht)))
            phase = p.phase_out if kind == TRACE_OUT else p.phase_in
            for i in range(n):
                s = (phase + i / n) % 1.0
                x, y, z = project(p.at(1.0 - s if kind == TRACE_OUT else s, pts3))
                rad = max(2.0 * p.radius * focal / z * 0.75, 1.3)
                cr.arc(x, y, rad + 0.8, 0, math.tau)
                cr.set_source_rgba(*th["dot_ring"], 0.7)
                cr.fill()
                cr.arc(x, y, rad, 0, math.tau)
                cr.set_source_rgb(*th["dot"])
                cr.fill()

    def _text(self, cr, x, y, text, th, alpha=0.85, bold=False, pill=True):
        layout = PangoCairo.create_layout(cr)
        layout.set_markup(f"<b>{_esc(text)}</b>" if bold else _esc(text), -1)
        tw, tht = layout.get_pixel_size()
        if pill:
            cr.set_source_rgba(*th["bg"], 0.75)
            _round_rect(cr, x - 4, y - 2, tw + 8, tht + 4, 4)
            cr.fill()
        cr.set_source_rgba(*th["ink"], alpha)
        cr.move_to(x, y)
        PangoCairo.show_layout(cr, layout)
        return tw, tht

    def _labels(self, cr, project, th):
        for box in self.boxes:
            if not box.label:
                continue
            is_device = box.color in ("port", "m2")
            pipe = next((p for p in self.pipes if p.label == box.label
                         and (p.a == box.top)), None) if is_device else None
            if is_device and pipe is not self.hover and not (
                    pipe and pipe.channel and pipe.channel == self.selected):
                continue
            x, y, _ = project((box.top[0], box.top[1] + 0.4, box.top[2]))
            layout = PangoCairo.create_layout(cr)
            layout.set_text(box.label, -1)
            tw = layout.get_pixel_size()[0]
            self._text(cr, x - tw / 2, y - 22, box.label, th, 0.9)

    def _legend(self, cr, w, h, th, mode):
        self._text(cr, 16, 12, "Inside the machine", th, 1.0, bold=True, pill=False)
        self._text(cr, 16, 32, "Drag to orbit · scroll to zoom · click a trace to inspect it"
                   + (f" · {self.rt_status}" if self.rt_status else ""), th, 0.6, pill=False)
        # heat ramp
        x0, bw, bh = 16, min(300, w - 32), 10
        y0 = h - 58
        cr.set_source_rgba(*th["bg"], 0.82)
        _round_rect(cr, x0 - 8, y0 - 28, min(w - 16, 860), 66, 8)
        cr.fill()
        self._text(cr, x0, y0 - 22, "Trace color: data volume (log scale)", th, 0.8, pill=False)
        steps = 64
        for i in range(steps):
            cr.set_source_rgb(*ramp(i / (steps - 1), mode))
            cr.rectangle(x0 + bw * i / steps, y0, bw / steps + 0.5, bh)
            cr.fill()
        cr.set_source_rgba(*th["outline"], 0.4)
        cr.set_line_width(1)
        cr.rectangle(x0 + 0.5, y0 + 0.5, bw - 1, bh - 1)
        cr.stroke()
        lo, hi = HEAT_RANGE[BYTES]
        for exp, text in ((2, "100 B/s"), (4, "10 kB/s"), (6, "1 MB/s"), (8, "100 MB/s")):
            fx = x0 + bw * (exp - lo) / (hi - lo)
            layout = PangoCairo.create_layout(cr)
            layout.set_text(text, -1)
            tw = layout.get_pixel_size()[0]
            tx = min(max(fx - tw / 2, x0), x0 + bw - tw)
            cr.set_source_rgba(*th["ink"], 0.7)
            cr.move_to(tx, y0 + bh + 3)
            PangoCairo.show_layout(cr, layout)
        # idle swatch + particle key
        ix = x0 + bw + 24
        cr.set_source_rgb(*th["idle"])
        _round_rect(cr, ix, y0, 22, bh, 3)
        cr.fill()
        self._text(cr, ix + 28, y0 - 4, "no traffic", th, 0.7, pill=False)
        self._text(cr, ix, y0 + bh + 3,
                   "Events & IRQs: 1 – 100k/s · two-way links are in/out pairs, "
                   "pulses run toward the CPU on in", th, 0.6, pill=False)
        if self.power_layer:
            self._power_legend(cr, w, x0, bw, bh, y0 - 66, th, mode)

    def _power_legend(self, cr, w, x0, bw, bh, y0, th, mode):
        cr.set_source_rgba(*th["bg"], 0.82)
        _round_rect(cr, x0 - 8, y0 - 28, min(w - 16, 860), 60, 8)
        cr.fill()
        self._text(cr, x0, y0 - 22, "Part glow: power draw (log scale)", th, 0.8, pill=False)
        steps = 64
        for i in range(steps):
            cr.set_source_rgb(*ramp(i / (steps - 1), mode, POWER_STOPS[mode]))
            cr.rectangle(x0 + bw * i / steps, y0, bw / steps + 0.5, bh)
            cr.fill()
        lo, hi = POWER_RANGE
        for exp, text in ((-1, "0.1 W"), (0, "1 W"), (1, "10 W"), (2, "100 W")):
            fx = x0 + bw * (exp - lo) / (hi - lo)
            layout = PangoCairo.create_layout(cr)
            layout.set_text(text, -1)
            tw = layout.get_pixel_size()[0]
            cr.set_source_rgba(*th["ink"], 0.7)
            cr.move_to(min(max(fx - tw / 2, x0), x0 + bw - tw), y0 + bh + 3)
            PangoCairo.show_layout(cr, layout)
        notes = ["CPU: RAPL", "GPU: NVML", "USB: declared maximum (estimate, dimmer)"]
        cpu = self.power.get("cpu")
        if cpu is None:
            notes[0] = "CPU: no RAPL counters"
        elif cpu.watts is None:
            notes[0] = f"CPU: {cpu.note} (see README)"
        if "gpu" not in self.power:
            notes[1] = "GPU: no NVIDIA GPU / NVML"
        self._text(cr, x0 + bw + 24, y0 - 2, " · ".join(notes), th, 0.7, pill=False)

    def _tooltip_lines(self) -> list[str]:
        p = self.hover
        if p is not None:
            lines = [p.label]
            if p.detail:
                lines.append(p.detail)
            rates = [f"In {fmt_rate(p.rx, p.unit)}"]
            if p.unit == BYTES:
                rates.append(f"Out {fmt_rate(p.tx, p.unit)}")
            lines.append("   ".join(rates))
            return lines
        box = self.hover_part
        pw = self.power_of(box.power)
        title = {"cpu": "CPU", "gpu": "GPU"}.get(box.power, box.label)
        if pw is None:
            return [f"{title} power", "no reading"]
        value = (f"{'≤ ' if pw.estimated else ''}{fmt_watts(pw.watts)}"
                 f"{' (estimate)' if pw.estimated else ''}" if pw.watts is not None
                 else pw.note + (": see README" if pw.note == "needs permission" else ""))
        return [f"{title} power: {value}", f"{pw.source} · {pw.detail}".strip(" ·")]

    def _tooltip(self, cr, w, h, th, mode):
        lines = self._tooltip_lines()
        layout = PangoCairo.create_layout(cr)
        layout.set_markup(f"<b>{_esc(lines[0])}</b>\n" + _esc("\n".join(lines[1:])), -1)
        tw, tht = layout.get_pixel_size()
        mx, my = self.mouse
        bx = mx + 16 if mx + 16 + tw + 16 < w else mx - tw - 28
        by = min(max(my + 16, 8), h - tht - 20)
        bg = (0.1, 0.1, 0.1, 0.94) if mode == "dark" else (1, 1, 1, 0.96)
        cr.set_source_rgba(*bg)
        _round_rect(cr, bx, by, tw + 16, tht + 12, 6)
        cr.fill_preserve()
        cr.set_source_rgba(*th["outline"], 0.4)
        cr.set_line_width(1)
        cr.stroke()
        cr.set_source_rgba(*th["ink"], 1)
        cr.move_to(bx + 8, by + 6)
        PangoCairo.show_layout(cr, layout)


def _hull(pts):
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def _inside(poly, x, y) -> bool:
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        c = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if c != 0:
            if sign and (c > 0) != (sign > 0):
                return False
            sign = c
    return True


def _round_rect(cr, x, y, w, h, r):
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    cr.close_path()


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
