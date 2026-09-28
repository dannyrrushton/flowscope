"""Triangle meshes for the ray-traced motherboard: PCB, socket, VRM, DIMMs,
slots, heatsinks, capacitors, rear-I/O ports, drives… placed on the same
layout as the pipes in scene3d. Units are centimetres, y is up, the PCB top
is y = 0. Materials are 16 floats each (see Material in optix/src/shared.h)."""
from __future__ import annotations

import math
from array import array

from . import optix_backend as ob

Vec = tuple[float, float, float]


class Mesh:
    def __init__(self):
        self.verts = array("f")
        self.tri_mat = array("I")
        self.materials = array("f")
        self._mat_index: dict[tuple, int] = {}

    def material(self, albedo, rough=0.5, metal=0.0, pattern=ob.PLAIN, heat=-1,
                 rect=(0.0, 0.0, 0.0, 0.0), extra=(0.0, 0.0, 0.0, 0.0), emission=0.0) -> int:
        key = (tuple(albedo), rough, metal, pattern, heat, tuple(rect), tuple(extra), emission)
        idx = self._mat_index.get(key)
        if idx is None:
            idx = self._mat_index[key] = len(self.materials) // 16
            self.materials.extend([*albedo, rough, metal, emission, float(pattern), float(heat),
                                   *rect, *extra])
        return idx

    def tri(self, a: Vec, b: Vec, c: Vec, mat: int) -> None:
        self.verts.extend([*a, *b, *c])
        self.tri_mat.append(mat)

    def quad(self, a, b, c, d, mat) -> None:
        self.tri(a, b, c, mat)
        self.tri(a, c, d, mat)

    def box(self, lo: Vec, hi: Vec, mat: int, top: int | None = None) -> None:
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        t = mat if top is None else top
        self.quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), t)  # top
        self.quad((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0), mat)
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), mat)
        self.quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1), mat)
        self.quad((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0), mat)
        self.quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1), mat)

    def cbox(self, cx, cz, sx, sy, sz, mat, y0=0.0, top=None) -> None:
        """Box by footprint centre and size, sitting on y0."""
        self.box((cx - sx / 2, y0, cz - sz / 2), (cx + sx / 2, y0 + sy, cz + sz / 2), mat, top)

    def cylinder(self, a: Vec, b: Vec, r: float, mat: int, cap: int | None = None,
                 segs: int = 18) -> None:
        ax = [b[i] - a[i] for i in range(3)]
        ln = math.sqrt(sum(v * v for v in ax)) or 1.0
        ax = [v / ln for v in ax]
        ref = (1.0, 0.0, 0.0) if abs(ax[1]) > 0.9 else (0.0, 1.0, 0.0)
        u = _norm(_cross(ax, ref))
        v = _cross(ax, u)
        ring = [(math.cos(2 * math.pi * i / segs), math.sin(2 * math.pi * i / segs))
                for i in range(segs)]

        def pt(base, c, s):
            return tuple(base[k] + r * (u[k] * c + v[k] * s) for k in range(3))

        capm = mat if cap is None else cap
        for i in range(segs):
            c0, s0 = ring[i]
            c1, s1 = ring[(i + 1) % segs]
            self.quad(pt(a, c0, s0), pt(a, c1, s1), pt(b, c1, s1), pt(b, c0, s0), mat)
            self.tri(b, pt(b, c0, s0), pt(b, c1, s1), capm)
            self.tri(a, pt(a, c1, s1), pt(a, c0, s0), mat)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(sum(v * v for v in a)) or 1.0
    return tuple(v / n for v in a)


# --------------------------------------------------------------- materials
class Palette:
    def __init__(self, m: Mesh):
        self.m = m
        self.pcb = m.material((0.018, 0.034, 0.026), 0.42, 0.0, ob.PCB)
        self.steel = m.material((0.62, 0.62, 0.64), 0.32, 1.0, ob.BRUSHED)
        self.alu = m.material((0.82, 0.82, 0.84), 0.28, 1.0, ob.BRUSHED)
        self.gunmetal = m.material((0.13, 0.13, 0.14), 0.3, 0.9, ob.BRUSHED)
        self.black = m.material((0.018, 0.018, 0.02), 0.5)
        self.black_gloss = m.material((0.012, 0.012, 0.014), 0.18)
        self.white = m.material((0.72, 0.71, 0.67), 0.55)
        self.gold = m.material((1.0, 0.78, 0.34), 0.22, 1.0)
        self.brass = m.material((0.85, 0.64, 0.32), 0.35, 1.0)
        self.ferrite = m.material((0.16, 0.16, 0.17), 0.65)
        self.cap_sleeve = m.material((0.09, 0.04, 0.14), 0.35)
        self.cap_gold = m.material((0.75, 0.55, 0.12), 0.35)
        self.usb3 = m.material((0.09, 0.09, 0.10), 0.45)  # not blue: blue means data volume
        self.substrate = m.material((0.05, 0.16, 0.07), 0.4)
        self.ram_pcb = m.material((0.03, 0.11, 0.05), 0.45)
        self.accent = m.material((0.55, 0.56, 0.58), 0.25, 1.0, ob.BRUSHED)  # neutral, never blue
        self.floor = m.material((0.045, 0.045, 0.05), 0.45, 0.6, ob.FLOOR)
        self.fr4 = m.material((0.32, 0.27, 0.17), 0.8)
        self.jack = [m.material(c, 0.45) for c in ((0.10, 0.45, 0.10), (0.75, 0.25, 0.45),
                                                   (0.70, 0.60, 0.10), (0.05, 0.05, 0.05))]

    def chip(self, cx, cz, hx, hz, heat, metal=False, marks=True, power=-1):
        """Package top with a data hotspot (blue) from pipe `heat`; with the
        power layer on it shows power slot `power` (red) instead."""
        albedo = (0.7, 0.7, 0.72) if metal else (0.03, 0.03, 0.032)
        return self.m.material(albedo, 0.35 if metal else 0.55, 0.9 if metal else 0.0,
                               ob.CHIP, heat, (cx, cz, hx, hz),
                               (0.0 if marks else 1.0, 0, 0, power + 1))

    def power(self, albedo, rough, metal, slot, rect=(0.0, 0.0, 0.0, 0.0)):
        """Surface lit red by power slot `slot` (plain when the layer is off)."""
        if slot < 0:
            return self.m.material(albedo, rough, metal)
        return self.m.material(albedo, rough, metal, ob.POWER, slot, rect)

    def label(self, cx, cz, hx, hz, band=0.7):
        return self.m.material((0.13, 0.13, 0.14), 0.3, 0.9, ob.LABEL, -1,
                               (cx, cz, hx, hz), (0.86, 0.86, 0.84, band))

    def led(self, heat, color):
        return self.m.material((0.05, 0.05, 0.05), 0.3, 0.0, ob.GLOW, heat,
                               extra=(*color, 0.0))


# ------------------------------------------------------------------ parts
def build(scene) -> Mesh:
    """Mesh for every Box in `scene.boxes` plus the fixed board furniture."""
    m = Mesh()
    p = Palette(m)
    cpu_power = next((b.power_slot for b in scene.boxes if b.power == "cpu"), -1)
    board_furniture(m, p, cpu_power)
    for box in scene.boxes:
        builder = PARTS.get(box.part.split(":")[0])
        if builder:
            builder(m, p, box)
    return m


def board_furniture(m: Mesh, p: Palette, cpu_power: int = -1) -> None:
    # case floor, PCB, standoffs (no I/O shield: it would hide the rear ports
    # from the default camera)
    m.box((-26, -1.4, -19), (32, -1.2, 19), p.floor)
    m.box((-16, -0.16, -12), (16, 0.0, 12), p.pcb)
    for x, z in ((-15, -11), (-15, 11), (7, -11), (7, 11), (15, -11), (15, 11), (-4, 0)):
        m.cylinder((x, -1.2, z), (x, -0.16, z), 0.28, p.brass)
        m.cylinder((x, 0.0, z), (x, 0.08, z), 0.36, p.steel)

    # VRM: chokes + finned heatsinks around the socket, then the solid caps.
    # The chokes and heatsink tops carry the CPU's power in the power layer.
    for i in range(6):
        z = -5.0 + i * 1.0
        m.cbox(-5.6, z, 0.75, 0.6, 0.75, p.ferrite,
               top=p.power((0.16, 0.16, 0.17), 0.65, 0.0, cpu_power, (-5.6, z, 0.37, 0.37)))
    for i in range(5):
        x = -4.3 + i * 1.0
        m.cbox(x, -6.6, 0.75, 0.6, 0.75, p.ferrite,
               top=p.power((0.16, 0.16, 0.17), 0.65, 0.0, cpu_power, (x, -6.6, 0.37, 0.37)))
    heatsink(m, p, -7.0, -3.0, 1.0, 6.2, 1.5, along="z", power=cpu_power)
    heatsink(m, p, -1.8, -7.6, 6.0, 0.9, 1.3, along="x", power=cpu_power)
    for i in range(6):
        capacitor(m, p, -8.1, -5.4 + i * 1.0, 0.3, 0.9)
    # 8-pin EPS and 24-pin ATX power connectors
    connector(m, p, -11.5, -11.0, 2.1, 1.0, rows=2, cols=4)
    connector(m, p, 15.2, -1.5, 0.95, 5.4, rows=12, cols=2, white=False)
    # PCIe slots and the M.2 heatsink
    pcie_slot(m, p, -9.5, 2.2, 9.0, reinforced=True)
    pcie_slot(m, p, -12.5, 4.4, 2.6)
    m.box((-9.6, 0.0, 3.8), (-1.2, 0.35, 5.8), p.gunmetal)
    m.box((-9.4, 0.35, 4.0), (-1.4, 0.37, 5.6), p.accent)
    # SATA ports at the right edge, CMOS battery, audio caps
    for i in range(3):
        for j in range(2):
            m.cbox(15.2, 4.6 + i * 1.3, 0.9, 0.75, 1.0, p.black, y0=j * 0.8)
    m.cylinder((7.2, 0.0, 9.3), (7.2, 0.32, 9.3), 1.0, p.steel)
    for x, z in ((-13.6, 9.2), (-13.6, 10.4), (-12.4, 11.0), (-9.4, 6.4)):
        capacitor(m, p, x, z, 0.4, 1.1, gold=True)
    for x, z in ((12.6, 1.0), (13.4, 1.0), (7.5, 1.5), (1.0, 3.0), (-1.2, 8.0)):
        capacitor(m, p, x, z, 0.25, 0.7)
    # a scatter of small SMD parts
    for i in range(40):
        h = (i * 2654435761) & 0xffffffff
        x = -14 + (h % 2800) / 100
        z = -11 + ((h >> 12) % 2200) / 100
        if -6 < x < 8 and -9 < z < 3:
            continue
        m.cbox(x, z, 0.2, 0.08, 0.12, p.black if i % 3 else p.ferrite)


def heatsink(m, p, cx, cz, sx, sz, h, along="x", power=-1):
    """Finned aluminium block; fins run along `along`. Fin tops glow with
    power slot `power` in the power layer."""
    m.cbox(cx, cz, sx, 0.3, sz, p.gunmetal)
    top = p.power((0.13, 0.13, 0.14), 0.3, 0.9, power, (cx, cz, sx / 2, sz / 2))
    fins = 7
    for i in range(fins):
        if along == "x":
            z = cz - sz / 2 + (i + 0.5) * sz / fins
            m.cbox(cx, z, sx, h, sz / fins * 0.45, p.gunmetal, y0=0.3, top=top)
        else:
            x = cx - sx / 2 + (i + 0.5) * sx / fins
            m.cbox(x, cz, sx / fins * 0.45, h, sz, p.gunmetal, y0=0.3, top=top)


def capacitor(m, p, x, z, r, h, gold=False):
    m.cylinder((x, 0.0, z), (x, h, z), r, p.cap_gold if gold else p.cap_sleeve, p.alu)


def connector(m, p, cx, cz, sx, sz, rows, cols, white=True):
    body = p.white if white else p.black
    m.cbox(cx, cz, sx, 1.0, sz, body, top=p.black)
    for i in range(rows):
        for j in range(cols):
            x = cx - sx / 2 + (j + 0.5) * sx / cols
            z = cz - sz / 2 + (i + 0.5) * sz / rows
            m.cbox(x, z, sx / cols * 0.55, 0.02, sz / rows * 0.55, body, y0=1.0)


def pcie_slot(m, p, cx, cz, length, reinforced=False):
    m.cbox(cx, cz, length, 0.75, 0.75, p.black)
    m.cbox(cx, cz, length - 0.2, 0.02, 0.2, p.gold, y0=0.75)
    if reinforced:
        m.box((cx - length / 2 - 0.05, 0.0, cz - 0.42), (cx + length / 2 + 0.05, 0.8, cz - 0.38),
              p.steel)
        m.box((cx - length / 2 - 0.05, 0.0, cz + 0.38), (cx + length / 2 + 0.05, 0.8, cz + 0.42),
              p.steel)


def _fp(box):
    """Footprint centre, half sizes, base and top of a scene Box."""
    (x0, y0, z0), (x1, y1, z1) = box.lo, box.hi
    return (x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (z1 - z0) / 2, y0, y1


def part_cpu(m, p, box):
    cx, cz, hx, hz, _, top = _fp(box)
    m.cbox(cx, cz, 2 * hx + 1.4, 0.22, 2 * hz + 1.4, p.steel)  # socket frame
    m.cbox(cx - hx - 0.9, cz, 0.12, 0.12, 2 * hz + 1.0, p.steel, y0=0.22)  # lever
    m.cbox(cx, cz, 2 * hx, 0.15, 2 * hz, p.substrate, y0=0.22)
    m.box((cx - hx + 0.5, 0.37, cz - hz + 0.5), (cx + hx - 0.5, top, cz + hz - 0.5),
          p.steel, top=p.chip(cx, cz, hx - 0.5, hz - 0.5, box.pipe, metal=True,
                              power=box.power_slot))


def part_pch(m, p, box):
    cx, cz, hx, hz, _, top = _fp(box)
    m.cbox(cx, cz, 2 * hx - 0.6, 0.12, 2 * hz - 0.6, p.black)
    m.cbox(cx, cz, 2 * hx, 0.28, 2 * hz, p.alu, y0=0.12)
    hot = p.chip(cx, cz, hx, hz, box.pipe, metal=True, marks=False)
    for i in range(7):
        z = cz - hz + (i + 0.5) * 2 * hz / 7
        m.box((cx - hx, 0.4, z - 0.09), (cx + hx, top, z + 0.09), p.alu, top=hot)


def part_ram(m, p, box):
    cx, cz, hx, hz, _, top = _fp(box)
    m.cbox(cx, cz, 0.75, 0.6, 2 * hz + 1.2, p.black)  # slot
    for dz in (-hz - 0.45, hz + 0.45):  # latches
        m.cbox(cx, cz + dz, 0.8, 0.9, 0.3, p.white)
    m.box((cx - 0.06, 0.6, cz - hz), (cx + 0.06, top - 0.2, cz + hz), p.ram_pcb)
    for dx in (-0.2, 0.2):  # heat spreaders
        m.box((cx + dx - 0.05, 0.7, cz - hz + 0.1), (cx + dx + 0.05, top, cz + hz - 0.1),
              p.gunmetal)
    m.box((cx - 0.25, top - 0.05, cz - hz + 0.1), (cx + 0.25, top, cz + hz - 0.1), p.accent)


def part_ctrl(m, p, box):
    cat = box.part.split(":")[1]
    cx, cz, hx, hz, _, top = _fp(box)
    if cat == "Audio":  # codec under an EMI shield
        m.box((cx - hx, 0, cz - hz), (cx + hx, top, cz + hz), p.steel,
              top=p.chip(cx, cz, hx, hz, box.pipe, metal=True, marks=False))
        return
    if cat == "Bluetooth":  # M.2 wireless card with a shielded module
        m.box((cx - hx - 0.4, 0.3, cz - hz - 1.0), (cx + hx + 0.4, 0.38, cz + hz + 0.5), p.ram_pcb)
        m.cbox(cx, cz - hz - 0.8, 2 * hx + 0.8, 0.3, 0.4, p.black)
        m.box((cx - hx, 0.38, cz - hz), (cx + hx, top, cz + hz), p.steel,
              top=p.chip(cx, cz, hx, hz, box.pipe, metal=True, marks=False))
        return
    # QFP / QFN package with gull-wing leads
    m.cbox(cx, cz, 2 * hx + 0.3, 0.04, 2 * hz + 0.3, p.steel)
    m.box((cx - hx, 0.04, cz - hz), (cx + hx, top, cz + hz), p.black,
          top=p.chip(cx, cz, hx, hz, box.pipe))


def part_leaf(m, p, box):
    cat = box.part.split(":")[1]
    cx, cz, hx, hz, y0, top = _fp(box)
    lo, hi = (cx - hx, y0, cz - hz), (cx + hx, top, cz + hz)
    face = cx - hx - 0.01  # outward face of rear-I/O ports (−x)
    if cat == "Network":  # RJ45: steel shell, dark socket, link / activity LEDs
        m.box(lo, hi, p.steel)
        m.box((face - 0.01, 0.12, cz - 0.42), (face + 0.02, top - 0.3, cz + 0.42), p.black)
        m.box((face - 0.02, top - 0.24, cz - hz + 0.1), (face, top - 0.1, cz - hz + 0.35),
              p.led(box.pipe, (0.15, 1.0, 0.2)))
        m.box((face - 0.02, top - 0.24, cz + hz - 0.35), (face, top - 0.1, cz + hz - 0.1),
              p.led(box.pipe, (1.0, 0.55, 0.05)))
    elif cat == "USB":  # USB-A: shell (lit by its estimated power), cavity, tongue
        m.box(lo, hi, p.steel, top=p.power((0.62, 0.62, 0.64), 0.32, 1.0, box.power_slot,
                                           (cx, cz, hx, hz)))
        m.box((face - 0.01, 0.1, cz - hz + 0.1), (face + 0.02, top - 0.1, cz + hz - 0.1), p.black)
        m.box((face - 0.03, 0.3, cz - hz + 0.15), (face + 0.2, 0.45, cz + hz - 0.15), p.usb3)
    elif cat == "Audio":  # 3.5 mm jack: coloured body with a ring
        m.box(lo, hi, p.jack[box.index % len(p.jack)])
        y = (y0 + top) / 2
        m.cylinder((face + 0.05, y, cz), (face - 0.12, y, cz), 0.3, p.steel, p.black)
    elif cat == "Storage":  # M.2 NVMe SSD: socket, stick, controller + NAND, screw
        x0 = cx - hx
        m.box((x0, 0.0, cz - hz), (x0 + 0.55, 0.45, cz + hz), p.black)  # M.2 socket
        m.box((x0 + 0.1, 0.2, cz - hz + 0.1), (x0 + 0.55, 0.26, cz + hz - 0.1), p.gold)
        m.box((x0 + 0.3, 0.2, cz - hz + 0.1), (cx + hx - 0.2, 0.28, cz + hz - 0.1), p.ram_pcb)
        c = (x0 + 1.25, cz)
        m.box((c[0] - 0.5, 0.28, cz - 0.5), (c[0] + 0.5, top, cz + 0.5), p.black,
              top=p.chip(c[0], cz, 0.5, 0.5, box.pipe))  # controller glows with its traffic
        for nx in (x0 + 2.35, x0 + 3.4):
            m.box((nx - 0.45, 0.28, cz - 0.6), (nx + 0.45, top - 0.03, cz + 0.6), p.black,
                  top=p.label(nx, cz, 0.45, 0.6))
        m.cylinder((cx + hx - 0.1, 0.0, cz), (cx + hx - 0.1, 0.32, cz), 0.22, p.brass, p.steel)
    elif cat == "Bluetooth":  # U.FL / SMA antenna connector
        m.cylinder((cx, 0.0, cz), (cx, top - 0.1, cz), hx * 0.8, p.gold)
        m.cylinder((cx, top - 0.1, cz), (cx, top, cz), hx * 0.5, p.black)
    elif cat in ("Serial", "Input"):  # 2×5 pin header on a plastic base
        base = p.black if cat == "Serial" else p.usb3
        m.box(lo, (hi[0], y0 + 0.25, hi[2]), base)
        for i in range(5):
            for j in range(2):
                x = cx - hx + (i + 0.5) * 2 * hx / 5
                z = cz - hz + (j + 0.5) * 2 * hz / 2
                m.cbox(x, z, 0.07, top - 0.25, 0.07, p.gold, y0=0.25)
    elif cat == "Interrupts":  # small SOIC packages
        m.cbox(cx, cz, 2 * hx + 0.2, 0.03, 2 * hz - 0.1, p.steel)
        m.box((cx - hx, 0.03, cz - hz), (cx + hx, top, cz + hz), p.black,
              top=p.chip(cx, cz, hx, hz, box.pipe))
    else:
        m.box(lo, hi, p.black)


def part_gpu(m, p, box):
    """Dual-slot graphics card standing in the x16 slot (z = 2.2): PCB, back
    plate, I/O bracket, fin stack under an open-top shroud, two fans on the
    front face, light bar and power plug. Fin tops and the light bar glow with
    the GPU's power in the power layer."""
    cx, cz, hx, hz, _, top = _fp(box)
    x0, x1, slot = cx - hx, cx + hx, 2.2
    front = cz + hz
    m.box((x0 + 0.6, 0.35, slot - 0.06), (x1 - 0.3, top - 0.2, slot + 0.06), p.ram_pcb)
    m.box((x0 + 0.8, 0.3, slot - 0.07), (x0 + 5.8, 0.75, slot + 0.07), p.gold)  # edge fingers
    m.box((x0 + 0.6, 0.9, slot - 0.2), (x1 - 0.3, top - 0.1, slot - 0.07), p.gunmetal)  # backplate
    m.box((x0, -0.3, slot - 0.3), (x0 + 0.12, top + 0.3, front), p.steel)  # I/O bracket
    fins_top = p.power((0.8, 0.8, 0.82), 0.3, 1.0, box.power_slot,
                       (cx, (slot + front) / 2, hx, (front - slot) / 2))
    for i in range(34):
        fx = x0 + 0.6 + (i + 0.5) * (2 * hx - 0.9) / 34
        m.box((fx - 0.04, 1.0, slot + 0.12), (fx + 0.04, top - 0.08, front - 0.32), p.alu,
              top=fins_top)
    m.box((x0 + 0.3, 0.95, front - 0.3), (x1, top, front), p.gunmetal)  # shroud face
    m.box((x0 + 0.3, 0.95, slot + 0.1), (x1, 1.05, front), p.gunmetal)  # shroud floor
    m.box((x1 - 0.12, 0.95, slot + 0.1), (x1, top, front), p.gunmetal)  # end cap
    radius = min((top - 1.0) / 2 - 0.12, hx / 2 - 0.3)
    fy = (top + 1.0) / 2
    for fx in (cx - hx / 2 + 0.2, cx + hx / 2 - 0.2):
        m.cylinder((fx, fy, front), (fx, fy, front + 0.04), radius + 0.08, p.black)
        m.cylinder((fx, fy, front), (fx, fy, front + 0.16), 0.38, p.gunmetal, p.accent)
        for k in range(7):  # blades
            a = 2 * math.pi * k / 7

            def pt(r, t):
                return (fx + r * math.cos(t), fy + r * math.sin(t), front + 0.09)
            m.quad(pt(0.38, a), pt(radius, a + 0.12), pt(radius, a + 0.62), pt(0.38, a + 0.4),
                   p.black_gloss)
    m.box((x0 + 0.8, top - 0.3, front), (x1 - 0.5, top - 0.14, front + 0.03),
          p.power((0.12, 0.12, 0.13), 0.4, 0.0, box.power_slot))  # light bar
    m.box((x1 - 2.4, top, slot + 0.1), (x1 - 1.0, top + 0.35, slot + 0.8), p.black)  # 8-pin


PARTS = {"cpu": part_cpu, "pch": part_pch, "ram": part_ram, "ctrl": part_ctrl,
         "leaf": part_leaf, "gpu": part_gpu}
