# Flowscope

Live GTK4 monitor of data flowing through every I/O channel on a Linux box:
network, USB, serial, audio (analog, S/PDIF, HDMI/DP), Bluetooth, input devices,
storage and hardware interrupts. Pure Python + PyGObject, no other dependencies.

    python3 -m flowscope                 # GUI
    python3 -m flowscope --view inside   # GUI, opening on the 3D view
    python3 -m flowscope --text --all    # headless snapshot

The **Inside** tab is a 3D view of the machine. Every channel is a trace routed
across the motherboard, PCB-style (straight runs, 45° corners, fanned into the
chip), from its device (rear I/O ports, M.2 SSDs, headers) to its controller;
controllers feed the chipset over bus traces and the chipset feeds the CPU.
Two-way links are in/out differential pairs. Trace color is a log-scale heat
map of the data volume (100 B/s → 1 GB/s; events and IRQs 1 → 100k/s); pulses
run toward the CPU on the in trace and away from it on the out trace. Drag to
orbit, scroll to zoom, hover for rates, click a trace to select that channel.

The board is **ray-traced with NVIDIA OptiX** when the native renderer is built:
modelled parts (socket, VRM, DIMMs, slots, heatsinks, capacitors, ports, M.2
SSDs), tin-plated traces that glow navy → blue → icy white with volume,
controller, SSD and CPU lids with a thermal hotspot gradient, port LEDs that
light with traffic, and the glow bouncing off nearby parts; the OptiX AI
denoiser cleans each frame.

    make -C flowscope/optix              # needs nvcc + an NVIDIA driver with OptiX
    python3 -m flowscope --view inside   # uses OptiX automatically once built

**Power layer** (the *Power* button on the Inside view, or `--power`): parts
glow on a separate red scale by their power draw, 0.1 W → 300 W (log). The CPU
lid, VRM chokes and heatsinks show CPU package power from RAPL; a graphics card
in the x16 slot shows board power from NVIDIA NVML; USB ports show their
*declared maximum* draw, dimmer and labelled as an estimate. Hover a part for
its watts and source. NICs, audio, SSDs and the chipset expose no power
sensors, so they stay neutral. RAPL is root-only by default (CVE-2020-8694);
`contrib/60-flowscope-rapl.rules` grants read access to a `rapl` group — the
install steps are in the file. `--text --all` also lists every power reading.

Without the build, a GPU or the driver, the view falls back to a software Cairo
renderer (force either with `--renderer optix|cairo`). The OptiX headers are
vendored in `flowscope/optix/third_party/optix` (NVIDIA licence alongside);
Python still needs nothing beyond PyGObject.

| Channel    | Source                                   | Notes |
|------------|------------------------------------------|-------|
| Network    | `/proc/net/dev`                          | per interface, bytes in/out |
| USB        | sysfs tree + usbmon                      | without usbmon, a device's traffic is the sum of what it hosts (NIC, disk, audio, BT, HID) |
| Serial     | `/sys/class/tty`, `/proc/tty/driver/serial` | byte counters are root-only; USB serial has no kernel counters |
| Audio      | `/proc/asound/card*/pcm*/sub*`           | bytes = hardware pointer × frame size; S/PDIF / HDMI tagged by PCM name |
| Bluetooth  | `HCIGETDEVINFO` ioctl                    | per controller byte + ACL/SCO packet counters |
| Input      | `/dev/input/event*`                      | needs `input` group; only event counts are kept, never contents |
| Storage    | `/sys/block/*/stat`                      | dm/md devices are shown but excluded from totals |
| Interrupts | `/proc/interrupts`                       | IRQ/s per line — a proxy for any device |

**Real per-device USB bus traffic:** `sudo modprobe usbmon` and run as root
(or make `/sys/kernel/debug/usb/usbmon/0u` readable).
