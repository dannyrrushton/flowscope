"""GTK4 front end: category sidebar, a large live graph for the selection, and
a list of every channel with a sparkline and current in/out rates."""
from __future__ import annotations

from collections import deque

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gdk, GLib, Gtk, Pango, PangoCairo  # noqa: E402

from .collectors import BYTES  # noqa: E402
from .monitor import (BYTE_CATEGORIES, HISTORY, Channel, Monitor,  # noqa: E402
                      fmt_bytes, fmt_rate)
from .scene3d import FlowScene  # noqa: E402

# Categorical slots 1 & 2 of the reference palette (light / dark steps).
IN_COLOR = {"light": "#2a78d6", "dark": "#3987e5"}
OUT_COLOR = {"light": "#eb6834", "dark": "#d95926"}
ALL = "All"

CSS = b"""
.rate { font-feature-settings: "tnum"; }
.chan-name { font-weight: 600; }
.swatch { min-width: 10px; min-height: 10px; border-radius: 2px; }
.swatch.in { background: #2a78d6; }
.swatch.out { background: #eb6834; }
.graph-card { padding: 12px 16px 8px 16px; }
.big-title { font-size: 1.25em; font-weight: 700; }
.note { font-style: italic; }
"""


def _rgb(hex_: str, a: float = 1.0):
    h = hex_.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (a,)


def _mode(widget: Gtk.Widget) -> str:
    fg = widget.get_color()
    return "dark" if (fg.red + fg.green + fg.blue) / 3 > 0.5 else "light"


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** (len(str(int(v))) - 1) if v >= 1 else 1
    for step in (1, 2, 2.5, 5, 10):
        if v <= step * mag:
            return step * mag
    return 10 * mag


class Graph(Gtk.DrawingArea):
    """Area/line chart of one or two series. `big` adds grid, labels, hover."""

    def __init__(self, big: bool = False, width: int = 140, height: int = 30):
        super().__init__()
        self.big = big
        self.rx: list[float] = []
        self.tx: list[float] | None = None
        self.unit = BYTES
        self.interval = 1.0
        self.hover_x: float | None = None
        if big:
            self.set_hexpand(True)
            self.set_content_height(height)
            motion = Gtk.EventControllerMotion()
            motion.connect("motion", lambda _c, x, _y: self._hover(x))
            motion.connect("leave", lambda _c: self._hover(None))
            self.add_controller(motion)
        else:
            self.set_content_width(width)
            self.set_content_height(height)
            self.set_valign(Gtk.Align.CENTER)
        self.set_draw_func(self._draw)

    def _hover(self, x):
        self.hover_x = x
        self.queue_draw()

    def set_data(self, rx, tx, unit, interval):
        self.rx, self.tx, self.unit, self.interval = list(rx), (
            list(tx) if tx is not None else None), unit, interval
        self.queue_draw()

    def _draw(self, _area, cr, w, h):
        mode = _mode(self)
        fg = self.get_color()
        ink = (fg.red, fg.green, fg.blue)
        n = len(self.rx)
        if n < 2:
            return
        series = [(self.rx, IN_COLOR[mode])]
        if self.tx is not None:
            series.append((self.tx, OUT_COLOR[mode]))
        peak = max(max(s) for s, _ in series)
        top = _nice_max(peak * 1.05) if self.big else max(peak, 1e-9)
        right_pad = 88 if self.big else 0
        pw, ph = w - right_pad, h - (18 if self.big else 2)
        y0 = 1 if not self.big else 4

        def xy(i, v):
            return (i * pw / (n - 1), y0 + ph - (v / top) * (ph - y0) if top else y0 + ph)

        if self.big:  # recessive grid with value labels on the right
            cr.set_line_width(1)
            layout = self.create_pango_layout("")
            for frac in (0, 0.5, 1):
                y = round(y0 + ph - frac * (ph - y0)) + 0.5
                cr.set_source_rgba(*ink, 0.12 if frac else 0.3)
                cr.move_to(0, y)
                cr.line_to(pw, y)
                cr.stroke()
                layout.set_text(fmt_rate(top * frac, self.unit), -1)
                cr.set_source_rgba(*ink, 0.6)
                cr.move_to(pw + 8, y - 8)
                PangoCairo.show_layout(cr, layout)
            secs = (n - 1) * self.interval
            for label, x in ((f"−{secs:.0f} s", 0), ("now", pw)):
                layout.set_text(label, -1)
                lw = layout.get_pixel_size()[0]
                cr.move_to(min(max(x - lw / 2, 0), pw - lw), h - 16)
                PangoCairo.show_layout(cr, layout)

        for values, color in series:
            if max(values) <= 0 and not self.big:
                continue
            cr.move_to(*xy(0, values[0]))
            for i, v in enumerate(values[1:], 1):
                cr.line_to(*xy(i, v))
            cr.set_source_rgba(*_rgb(color))
            cr.set_line_width(2 if self.big else 1.5)
            cr.set_line_join(1)  # round
            cr.stroke_preserve()
            cr.line_to(pw, y0 + ph)
            cr.line_to(0, y0 + ph)
            cr.close_path()
            cr.set_source_rgba(*_rgb(color, 0.14))
            cr.fill()

        if self.big and self.hover_x is not None and 0 <= self.hover_x <= pw:
            i = round(self.hover_x / pw * (n - 1))
            x = i * pw / (n - 1)
            cr.set_source_rgba(*ink, 0.35)
            cr.set_line_width(1)
            cr.move_to(round(x) + 0.5, y0)
            cr.line_to(round(x) + 0.5, y0 + ph)
            cr.stroke()
            parts = [f"−{(n - 1 - i) * self.interval:.0f} s"]
            for (values, color), label in zip(series, ("In", "Out")):
                cx, cy = xy(i, values[i])
                cr.set_source_rgba(*_rgb(color))
                cr.arc(cx, cy, 4, 0, 6.2832)
                cr.fill()
                parts.append(f"{label} {fmt_rate(values[i], self.unit)}")
            layout = self.create_pango_layout("   ".join(parts))
            tw, th = layout.get_pixel_size()
            bx = min(max(x + 10, 0), pw - tw - 12) if x + tw + 22 > pw else x + 10
            bx = max(bx, 0)
            bg = (0.1, 0.1, 0.1, 0.92) if mode == "dark" else (1, 1, 1, 0.95)
            cr.set_source_rgba(*bg)
            cr.rectangle(bx, y0 + 4, tw + 12, th + 8)
            cr.fill()
            cr.set_source_rgba(*ink, 1)
            cr.move_to(bx + 6, y0 + 8)
            PangoCairo.show_layout(cr, layout)


class ChannelRow(Gtk.ListBoxRow):
    def __init__(self, ch: Channel):
        super().__init__()
        self.key = ch.key
        self.category = ch.category
        grid = Gtk.Box(spacing=12, margin_top=6, margin_bottom=6,
                       margin_start=12, margin_end=12)
        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        self.name = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.name.add_css_class("chan-name")
        self.detail = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.detail.add_css_class("dim-label")
        self.detail.add_css_class("caption")
        text.append(self.name)
        text.append(self.detail)
        self.spark = Graph(width=150, height=30)
        rates = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER)
        rates.set_size_request(118, -1)
        self.rx_box, self.rx = self._rate_line("in")
        self.tx_box, self.tx = self._rate_line("out")
        self.note = Gtk.Label(xalign=1)
        self.note.add_css_class("dim-label")
        self.note.add_css_class("note")
        for wdg in (self.rx_box, self.tx_box, self.note):
            rates.append(wdg)
        for wdg in (text, self.spark, rates):
            grid.append(wdg)
        self.set_child(grid)

    @staticmethod
    def _rate_line(kind):
        box = Gtk.Box(spacing=6, halign=Gtk.Align.END)
        sw = Gtk.Box(valign=Gtk.Align.CENTER)
        sw.add_css_class("swatch")
        sw.add_css_class(kind)
        sw.set_tooltip_text("Inbound" if kind == "in" else "Outbound")
        lbl = Gtk.Label(xalign=1)
        lbl.add_css_class("rate")
        box.append(sw)
        box.append(lbl)
        return box, lbl

    def update(self, ch: Channel, interval: float, show_category: bool):
        self.name.set_text(ch.name)
        detail = f"{ch.category} · {ch.detail}" if show_category else ch.detail
        self.detail.set_text(detail)
        self.set_tooltip_text(f"{ch.name}\n{ch.category} · {ch.detail}")
        self.rx_box.set_visible(ch.has_rx)
        self.tx_box.set_visible(ch.has_tx)
        self.rx.set_text(fmt_rate(ch.rx_rate, ch.unit))
        self.tx.set_text(fmt_rate(ch.tx_rate, ch.unit))
        self.note.set_text(ch.note)
        self.note.set_visible(bool(ch.note) and not (ch.has_rx or ch.has_tx))
        if self.get_mapped():
            self.spark.set_data(ch.rx_hist, ch.tx_hist if ch.has_tx else None,
                                ch.unit, interval)


class SidebarRow(Gtk.ListBoxRow):
    def __init__(self, category: str):
        super().__init__()
        self.category = category
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                      margin_top=8, margin_bottom=8, margin_start=12, margin_end=12)
        top = Gtk.Box(spacing=6)
        self.title = Gtk.Label(label=category, xalign=0, hexpand=True)
        self.title.add_css_class("chan-name")
        self.count = Gtk.Label(xalign=1)
        self.count.add_css_class("dim-label")
        top.append(self.title)
        top.append(self.count)
        self.rate = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.rate.add_css_class("dim-label")
        self.rate.add_css_class("caption")
        self.rate.add_css_class("rate")
        self.spark = Graph(width=180, height=22)
        self.spark.set_halign(Gtk.Align.FILL)
        for wdg in (top, self.rate, self.spark):
            box.append(wdg)
        self.set_child(box)


class Window(Gtk.ApplicationWindow):
    def __init__(self, app, interval: float, view: str = "graphs", renderer: str = "auto",
                 power: bool = False):
        super().__init__(application=app, title="Flowscope")
        self.set_default_size(1200, 780)
        self.monitor = Monitor()
        self.interval = interval
        self.paused = False
        self.rows: dict[str, ChannelRow] = {}
        self.cat_hist: dict[str, tuple[deque, deque, str]] = {}
        self.search = ""
        self.show_idle = False
        self.selected_key: str | None = None
        self.category = ALL

        # header bar
        header = Gtk.HeaderBar()
        self.pause_btn = Gtk.ToggleButton(icon_name="media-playback-pause-symbolic",
                                          tooltip_text="Pause sampling")
        self.pause_btn.connect("toggled", self._on_pause)
        header.pack_start(self.pause_btn)
        interval_dd = Gtk.DropDown.new_from_strings(["0.5 s", "1 s", "2 s", "5 s"])
        interval_dd.set_tooltip_text("Sampling interval")
        choices = [0.5, 1.0, 2.0, 5.0]
        interval_dd.set_selected(choices.index(interval) if interval in choices else 1)
        interval_dd.connect("notify::selected",
                            lambda dd, _p: self._set_interval(choices[dd.get_selected()]))
        header.pack_start(interval_dd)
        entry = Gtk.SearchEntry(placeholder_text="Filter channels")
        entry.connect("search-changed", self._on_search)
        header.pack_end(entry)
        self.sort_dd = Gtk.DropDown.new_from_strings(["Most active", "Name"])
        self.sort_dd.set_tooltip_text("Sort order")
        self.sort_dd.connect("notify::selected", lambda *_: self.list.invalidate_sort())
        header.pack_end(self.sort_dd)
        idle = Gtk.ToggleButton(label="Idle", tooltip_text=(
            "Show channels with no traffic yet in the All view "
            "(category views always list every channel)"))
        idle.connect("toggled", self._on_idle)
        header.pack_end(idle)
        self.power_btn = Gtk.ToggleButton(label="Power", tooltip_text=(
            "Show power draw on the 3D view in red: CPU (RAPL), GPU (NVML) and "
            "estimated USB budgets"))
        self.power_btn.connect("toggled", lambda b: self.scene.set_power_layer(b.get_active()))
        header.pack_end(self.power_btn)
        self.set_titlebar(header)

        # sidebar
        self.sidebar = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.sidebar.add_css_class("navigation-sidebar")
        self.side_rows: dict[str, SidebarRow] = {}
        for cat in [ALL] + self.monitor.categories:
            row = SidebarRow(cat)
            self.side_rows[cat] = row
            self.sidebar.append(row)
        self.sidebar.select_row(self.side_rows[ALL])
        self.sidebar.connect("row-selected", self._on_category)
        side_scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER,
                                         child=self.sidebar)
        side_scroll.set_size_request(230, -1)

        # graph card
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        card.add_css_class("graph-card")
        title_row = Gtk.Box(spacing=12)
        titles = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        self.big_title = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.big_title.add_css_class("big-title")
        self.big_sub = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.big_sub.add_css_class("dim-label")
        titles.append(self.big_title)
        titles.append(self.big_sub)
        title_row.append(titles)
        self.legend_tx = None
        legend = Gtk.Box(spacing=14, valign=Gtk.Align.CENTER)
        for kind, text in (("in", "In"), ("out", "Out")):
            b, lbl = ChannelRow._rate_line(kind)
            lbl.set_text(text)
            legend.append(b)
            if kind == "out":
                self.legend_tx = b
        title_row.append(legend)
        card.append(title_row)
        self.big = Graph(big=True, height=190)
        card.append(self.big)
        self.stats = Gtk.Label(xalign=0)
        self.stats.add_css_class("dim-label")
        self.stats.add_css_class("rate")
        card.append(self.stats)
        self.errors = Gtk.Label(xalign=0, wrap=True)
        self.errors.add_css_class("error")
        card.append(self.errors)

        # channel list
        self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.list.add_css_class("rich-list")
        self.list.set_filter_func(self._filter)
        self.list.set_sort_func(self._sort)
        self.list.connect("row-selected", self._on_channel)
        self.empty = Gtk.Label(label="No channels match", margin_top=24)
        self.empty.add_css_class("dim-label")
        self.list.set_placeholder(self.empty)
        list_scroll = Gtk.ScrolledWindow(vexpand=True, child=self.list)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        right.append(card)
        right.append(Gtk.Separator())
        right.append(list_scroll)

        self.scene = FlowScene(renderer)
        self.scene.on_select = self._on_scene_select
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.stack.add_titled(right, "graphs", "Graphs")
        self.stack.add_titled(self.scene, "inside", "Inside")
        self.stack.set_visible_child_name(view)
        header.set_title_widget(Gtk.StackSwitcher(stack=self.stack))
        self.stack.connect("notify::visible-child-name", lambda st, _p: self.power_btn.set_visible(
            st.get_visible_child_name() == "inside"))
        self.power_btn.set_visible(view == "inside")
        self.power_btn.set_active(power)

        paned = Gtk.Paned(start_child=side_scroll, end_child=self.stack,
                          shrink_start_child=False, resize_start_child=False)
        self.set_child(paned)

        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.connect("close-request", self._on_close)
        self._tick()
        self.timer = GLib.timeout_add(int(self.interval * 1000), self._tick)

    # ---------------------------------------------------------------- events
    def _set_interval(self, v):
        self.interval = v
        GLib.source_remove(self.timer)
        self.timer = GLib.timeout_add(int(v * 1000), self._tick)

    def _on_pause(self, btn):
        self.paused = btn.get_active()
        btn.set_icon_name("media-playback-start-symbolic" if self.paused
                          else "media-playback-pause-symbolic")
        btn.set_tooltip_text("Resume sampling" if self.paused else "Pause sampling")

    def _on_search(self, entry):
        self.search = entry.get_text().lower()
        self.list.invalidate_filter()

    def _on_idle(self, btn):
        self.show_idle = btn.get_active()
        self.list.invalidate_filter()

    def _on_category(self, _lb, row):
        self.category = row.category if row else ALL
        self.selected_key = None
        self.scene.set_category(self.category)
        self.list.unselect_all()
        self.list.invalidate_filter()
        self._refresh_graph()

    def _on_channel(self, _lb, row):
        self.selected_key = row.key if row else None
        self.scene.set_selected(self.selected_key)
        self._refresh_graph()

    def _on_scene_select(self, key):
        """A pipe was clicked in the 3D view: select its channel everywhere."""
        self.selected_key = key
        self.list.invalidate_filter()  # keeps the selected row visible
        row = self.rows.get(key) if key else None
        if row is not None:
            self.list.select_row(row)
        else:
            self.list.unselect_all()
        self._refresh_graph()

    def _on_close(self, *_):
        GLib.source_remove(self.timer)
        self.monitor.close()
        self.scene.close()
        return False

    # ------------------------------------------------------------- list ops
    def _filter(self, row: ChannelRow) -> bool:
        ch = self.monitor.channels.get(row.key)
        if ch is None:
            return False
        if self.category != ALL and ch.category != self.category:
            return False
        if self.search and self.search not in f"{ch.name} {ch.detail} {ch.category}".lower():
            return False
        if self.category == ALL and not self.show_idle and not self.search:
            return ch.ever_active or row.key == self.selected_key
        return True

    def _sort(self, a: ChannelRow, b: ChannelRow) -> int:
        ca, cb = self.monitor.channels.get(a.key), self.monitor.channels.get(b.key)
        if ca is None or cb is None:
            return 0
        if self.sort_dd.get_selected() == 0:
            ra, rb = _norm_rate(ca), _norm_rate(cb)
            if ra != rb:
                return -1 if ra > rb else 1
        ka, kb = (ca.category, ca.name.lower(), ca.key), (cb.category, cb.name.lower(), cb.key)
        return (ka > kb) - (ka < kb)

    # --------------------------------------------------------------- update
    def _tick(self):
        if self.paused:
            return True
        self.monitor.sample()
        chans = self.monitor.channels

        for key in [k for k in self.rows if k not in chans]:
            self.list.remove(self.rows.pop(key))
            if self.selected_key == key:
                self.selected_key = None
        for key, ch in chans.items():
            row = self.rows.get(key)
            if row is None:
                row = self.rows[key] = ChannelRow(ch)
                self.list.append(row)
            row.update(ch, self.interval, self.category == ALL)

        self._update_sidebar()
        self.scene.update(self.monitor)
        self.list.invalidate_filter()
        if self.sort_dd.get_selected() == 0:
            self.list.invalidate_sort()
        self._refresh_graph()
        errs = "; ".join(f"{k}: {v}" for k, v in self.monitor.errors.items())
        self.errors.set_text(errs)
        self.errors.set_visible(bool(errs))
        return True

    def _update_sidebar(self):
        totals = {}
        for cat in self.monitor.categories:
            rx, tx, unit = self.monitor.category_totals(cat)
            totals[cat] = (rx, tx, unit)
        totals[ALL] = (sum(totals[c][0] for c in BYTE_CATEGORIES if c in totals),
                       sum(totals[c][1] for c in BYTE_CATEGORIES if c in totals), BYTES)
        for cat, (rx, tx, unit) in totals.items():
            hist = self.cat_hist.get(cat)
            if hist is None:
                hist = self.cat_hist[cat] = (deque([0.0] * HISTORY, HISTORY),
                                             deque([0.0] * HISTORY, HISTORY), unit)
            hist[0].append(rx)
            hist[1].append(tx)
            has_tx = unit == BYTES
            row = self.side_rows[cat]
            n = sum(1 for c in self.monitor.channels.values()
                    if cat == ALL or c.category == cat)
            row.count.set_text(str(n))
            row.rate.set_text(f"↓ {fmt_rate(rx, unit)}   ↑ {fmt_rate(tx, unit)}"
                              if has_tx else fmt_rate(rx, unit))
            row.spark.set_data(hist[0], hist[1] if has_tx else None, unit, self.interval)

    def _refresh_graph(self):
        ch = self.monitor.channels.get(self.selected_key) if self.selected_key else None
        if ch is not None:
            self.big_title.set_text(ch.name)
            self.big_sub.set_text(f"{ch.category} · {ch.detail}"
                                  + (f" · {ch.note}" if ch.note else ""))
            rx, tx, unit = ch.rx_hist, (ch.tx_hist if ch.has_tx else None), ch.unit
            totals = []
            if ch.has_rx:
                totals.append(f"In: now {fmt_rate(ch.rx_rate, unit)} · peak "
                              f"{fmt_rate(max(rx), unit)} · total {_fmt_total(ch.rx_total, unit)}")
            if ch.has_tx:
                totals.append(f"Out: now {fmt_rate(ch.tx_rate, unit)} · peak "
                              f"{fmt_rate(max(tx), unit)} · total {_fmt_total(ch.tx_total, unit)}")
            self.stats.set_text("      ".join(totals) or ch.note)
        else:
            hist = self.cat_hist.get(self.category)
            if hist is None:
                return
            rx, tx, unit = hist
            tx = tx if unit == BYTES else None
            if self.category == ALL:
                self.big_title.set_text("All byte channels")
                self.big_sub.set_text("Network + Serial + Audio + Bluetooth + Storage "
                                      "· select a channel below to inspect it")
            else:
                self.big_title.set_text(self.category)
                self.big_sub.set_text("Sum of all channels in this category "
                                      "· select a channel below to inspect it")
            self.stats.set_text(f"In peak {fmt_rate(max(rx), unit)}"
                                + (f"      Out peak {fmt_rate(max(tx), unit)}" if tx else ""))
        self.legend_tx.set_visible(tx is not None)
        self.big.set_data(rx, tx, unit, self.interval)


def _norm_rate(ch: Channel) -> float:
    # rank events/IRQs below byte traffic of similar magnitude
    r = ch.rx_rate + ch.tx_rate
    return r if ch.unit == BYTES else r / 100


def _fmt_total(v: float, unit: str) -> str:
    return fmt_bytes(v) if unit == BYTES else f"{v:,.0f} {unit}"


class App(Gtk.Application):
    def __init__(self, interval: float, view: str, renderer: str, power: bool):
        super().__init__(application_id="io.github.flowscope")
        self.interval = interval
        self.view = view
        self.renderer = renderer
        self.power = power

    def do_activate(self):
        win = self.get_active_window() or Window(self, self.interval, self.view, self.renderer, self.power)
        win.present()


def run(interval: float = 1.0, view: str = "graphs", renderer: str = "auto",
        power: bool = False) -> int:
    return App(interval, view, renderer, power).run([])
