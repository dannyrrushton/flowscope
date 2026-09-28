import argparse
import sys
import time


def text_mode(seconds: float, interval: float, show_all: bool) -> None:
    from .monitor import Monitor, fmt_rate

    mon = Monitor()
    mon.sample()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(interval)
        mon.sample()
    for cat in mon.categories:
        chans = sorted((c for c in mon.channels.values() if c.category == cat),
                       key=lambda c: -(c.rx_rate + c.tx_rate))
        print(f"== {cat} ({len(chans)} channels)"
              + (f"  ERROR {mon.errors[cat]}" if cat in mon.errors else ""))
        for c in chans if show_all else chans[:6]:
            rates = []
            if c.has_rx:
                rates.append("in " + fmt_rate(c.rx_rate, c.unit))
            if c.has_tx:
                rates.append("out " + fmt_rate(c.tx_rate, c.unit))
            print(f"  {c.name[:34]:34} {' | '.join(rates) or c.note:40} {c.detail[:70]}")
    from .power import fmt_watts
    power = sorted(mon.power.readings.items(), key=lambda kv: (kv[1].estimated, kv[0]))
    print(f"== Power ({len(power)} readings)")
    for key, p in power if show_all else [kv for kv in power if not kv[1].estimated]:
        value = (fmt_watts(p.watts) + (" (estimate)" if p.estimated else "")
                 if p.watts is not None else p.note)
        print(f"  {key[:34]:34} {value:40} {p.source} · {p.detail}")
    mon.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="flowscope",
                                 description="Live data-flow monitor for every I/O channel.")
    ap.add_argument("--text", action="store_true", help="print a snapshot instead of the GUI")
    ap.add_argument("--all", action="store_true", help="with --text, list every channel")
    ap.add_argument("--seconds", type=float, default=2.0, help="with --text, sampling window")
    ap.add_argument("--interval", type=float, default=1.0, help="sampling interval (s)")
    ap.add_argument("--view", choices=("graphs", "inside"), default="graphs",
                    help="start on the graphs or the 3D inside-the-machine view")
    ap.add_argument("--power", action="store_true",
                    help="start with the 3D view's power layer on")
    ap.add_argument("--renderer", choices=("auto", "optix", "cairo"), default="auto",
                    help="3D view renderer: OptiX ray tracing when built and available "
                         "(auto), or the software Cairo renderer")
    args = ap.parse_args()
    if args.text:
        text_mode(args.seconds, args.interval, args.all)
        return
    from .gui import run
    sys.exit(run(args.interval, args.view, args.renderer, args.power))


if __name__ == "__main__":
    main()
