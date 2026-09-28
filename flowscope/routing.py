"""PCB-style trace routing on the board plane (x, z): orthogonal legs with 45°
chamfered corners, fanned into the side of a component so neighbouring traces
never cross, plus parallel offsets for differential (in / out) pairs."""
from __future__ import annotations

import math

Pt = tuple[float, float]
Rect = tuple[float, float, float, float]  # centre x, centre z, half x, half z


def _side_axis(side: str) -> tuple[bool, int]:
    """(arrives horizontally?, sign from the component toward its sources)."""
    return side in ("left", "right"), (-1 if side in ("left", "top") else 1)


def facing_side(dest: Rect, src: Rect, horizontal: bool) -> str:
    if horizontal:
        return "left" if src[0] < dest[0] else "right"
    return "top" if src[1] < dest[1] else "bottom"


def fan(dest: Rect, side: str, sources: list[Rect], pitch: float, margin: float,
        pin_pitch: float | None = None) -> list[list[Pt]]:
    """Route every source rect into `side` of `dest`.

    Each route leaves its source along the main axis, turns onto its own lane
    (a line parallel to the side), slides across to its own pin, then goes
    straight in. Pins keep the sources' order and lanes are handed out so that
    the trace with the longest slide turns closest to `dest` — together that
    makes the fan crossing-free."""
    horiz, sgn = _side_axis(side)
    a, b = (0, 1) if horiz else (1, 0)  # main / cross axis indices
    face = dest[a] + sgn * dest[2 + a]
    n = len(sources)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: sources[i][b])
    pp = min(pin_pitch or pitch, 2 * dest[2 + b] * 0.9 / n) if n > 1 else 0.0
    pins = [0.0] * n
    for rank, i in enumerate(order):
        pins[i] = dest[b] + (rank - (n - 1) / 2) * pp
    by_slide = sorted(range(n), key=lambda i: -abs(sources[i][b] - pins[i]))
    lanes = [0.0] * n
    for k, i in enumerate(by_slide):
        lanes[i] = face + sgn * (margin + k * pitch)

    routes = []
    for i, src in enumerate(sources):
        start_a = src[a] - sgn * src[2 + a]  # the source's side facing dest
        sb, pin = src[b], pins[i]
        lane = lanes[i]
        room = (start_a - face) * sgn
        if (lane - face) * sgn > room - 0.15:
            lane = face + sgn * room * 0.5
        if abs(sb - pin) < 1e-3:
            pts = [(start_a, sb), (face, pin)]
        else:
            pts = [(start_a, sb), (lane, sb), (lane, pin), (face, pin)]
        routes.append([(p[0], p[1]) if horiz else (p[1], p[0]) for p in pts])
    return routes


def chamfer(pts: list[Pt], size: float = 0.3) -> list[Pt]:
    """Cut each corner at 45°."""
    if len(pts) < 3:
        return list(pts)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p, c, n = pts[i - 1], pts[i], pts[i + 1]
        lp, ln = math.dist(p, c), math.dist(c, n)
        k = min(size, lp / 2, ln / 2)
        if k < 1e-4:
            out.append(c)
            continue
        out.append((c[0] + (p[0] - c[0]) * k / lp, c[1] + (p[1] - c[1]) * k / lp))
        out.append((c[0] + (n[0] - c[0]) * k / ln, c[1] + (n[1] - c[1]) * k / ln))
    out.append(pts[-1])
    return out


def offset(pts: list[Pt], d: float) -> list[Pt]:
    """Parallel polyline at distance d (left of travel direction), mitred."""
    def normal(p, q):
        dx, dz = q[0] - p[0], q[1] - p[1]
        ln = math.hypot(dx, dz) or 1.0
        return (-dz / ln, dx / ln)

    out = []
    for i, p in enumerate(pts):
        if i == 0:
            n = normal(pts[0], pts[1])
        elif i == len(pts) - 1:
            n = normal(pts[-2], pts[-1])
        else:
            n1, n2 = normal(pts[i - 1], p), normal(p, pts[i + 1])
            mx, mz = n1[0] + n2[0], n1[1] + n2[1]
            ml = math.hypot(mx, mz) or 1.0
            cos_half = (mx * n1[0] + mz * n1[1]) / ml
            n = (mx / ml / max(cos_half, 0.3), mz / ml / max(cos_half, 0.3))
        out.append((p[0] + n[0] * d, p[1] + n[1] * d))
    return out


def blocked(pts: list[Pt], rects: list[Rect]) -> float:
    """Length of the polyline that runs under any of the rects."""
    total = 0.0
    for p, q in zip(pts, pts[1:]):
        seg = math.dist(p, q)
        steps = max(int(seg / 0.25), 1)
        for s in range(steps):
            t = (s + 0.5) / steps
            x, z = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
            if any(abs(x - r[0]) < r[2] and abs(z - r[1]) < r[3] for r in rects):
                total += seg / steps
    return total
