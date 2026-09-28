"""ctypes bridge to the OptiX renderer in flowscope/optix (build it with
`make -C flowscope/optix`). Everything here is optional: `load()` returns None
with a reason when the library, the GPU or the driver isn't there, and the 3D
view falls back to its Cairo renderer."""
from __future__ import annotations

import ctypes
import os
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "optix", "build")
LIB = os.path.join(BUILD, "libflowscope_optix.so")
MODULE = os.path.join(BUILD, "device.optixir")

# Procedural patterns, see Pattern in optix/src/shared.h.
PLAIN, PCB, BRUSHED, CHIP, FINS, GOLD, LABEL, FLOOR, GLOW, POWER = range(10)
# PipeParams flags.
HOVER, SELECTED, FADED, ESTIMATE = 1, 2, 4, 8

_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_uint_p = ctypes.POINTER(ctypes.c_uint)


def _ptr(arr: array, ctype):
    addr, _ = arr.buffer_info()
    return ctypes.cast(addr, ctypes.POINTER(ctype))


class Renderer:
    def __init__(self, lib: ctypes.CDLL, handle: int):
        self.lib = lib
        self.handle = ctypes.c_void_p(handle)
        self.err = ctypes.create_string_buffer(1024)
        self.device = lib.fs_device_name(self.handle).decode()
        self.frame = bytearray()

    def _check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(self.err.value.decode(errors="replace"))

    def set_scene(self, verts: array, tri_mat: array, materials: array, trace_verts: array,
                  seg_index: array, seg_info: array) -> None:
        """verts: 9 floats per triangle; tri_mat: material index per triangle;
        materials: 16 floats each; trace_verts: (x, y, z, radius) per trace
        vertex; seg_index: first vertex of each straight segment; seg_info:
        (pipe, trace kind, s0, s1) per segment."""
        self._check(self.lib.fs_set_scene(
            self.handle, _ptr(verts, ctypes.c_float), len(tri_mat), _ptr(tri_mat, ctypes.c_uint),
            _ptr(materials, ctypes.c_float), len(materials) // 16,
            _ptr(trace_verts, ctypes.c_float), len(trace_verts) // 4,
            _ptr(seg_index, ctypes.c_uint), _ptr(seg_info, ctypes.c_float), len(seg_index),
            self.err, len(self.err)))

    def set_pipes(self, data: array) -> None:
        self._check(self.lib.fs_set_pipes(self.handle, _ptr(data, ctypes.c_float),
                                          len(data) // 12, self.err, len(self.err)))

    def render(self, camera: list[float], w: int, h: int, spp: int = 2, depth: int = 3,
               denoise: bool = True, exposure: float = 1.0) -> bytearray:
        """Returns w×h Cairo RGB24 pixels (BGRX)."""
        if len(self.frame) != w * h * 4:
            self.frame = bytearray(w * h * 4)
        cam = array("f", camera)
        out = (ctypes.c_ubyte * len(self.frame)).from_buffer(self.frame)
        self._check(self.lib.fs_render(self.handle, _ptr(cam, ctypes.c_float), w, h, spp, depth,
                                       int(denoise), ctypes.c_float(exposure), out,
                                       self.err, len(self.err)))
        return self.frame

    def close(self) -> None:
        if self.handle:
            self.lib.fs_destroy(self.handle)
            self.handle = None


def load() -> tuple[Renderer | None, str]:
    """(renderer, "") on success, else (None, reason)."""
    if not (os.path.exists(LIB) and os.path.exists(MODULE)):
        return None, "OptiX renderer not built (run: make -C flowscope/optix)"
    try:
        lib = ctypes.CDLL(LIB)
    except OSError as e:
        return None, f"cannot load {LIB}: {e}"
    lib.fs_create.restype = ctypes.c_void_p
    lib.fs_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.fs_destroy.argtypes = [ctypes.c_void_p]
    lib.fs_device_name.restype = ctypes.c_char_p
    lib.fs_device_name.argtypes = [ctypes.c_void_p]
    lib.fs_set_scene.argtypes = [ctypes.c_void_p, _c_float_p, ctypes.c_uint, _c_uint_p,
                                 _c_float_p, ctypes.c_uint, _c_float_p, ctypes.c_uint,
                                 _c_uint_p, _c_float_p, ctypes.c_uint,
                                 ctypes.c_char_p, ctypes.c_int]
    lib.fs_set_pipes.argtypes = [ctypes.c_void_p, _c_float_p, ctypes.c_uint,
                                 ctypes.c_char_p, ctypes.c_int]
    lib.fs_render.argtypes = [ctypes.c_void_p, _c_float_p, ctypes.c_uint, ctypes.c_uint,
                              ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_float,
                              ctypes.POINTER(ctypes.c_ubyte), ctypes.c_char_p, ctypes.c_int]
    err = ctypes.create_string_buffer(1024)
    handle = lib.fs_create(MODULE.encode(), err, len(err))
    if not handle:
        return None, f"OptiX init failed: {err.value.decode(errors='replace')}"
    return Renderer(lib, handle), ""
