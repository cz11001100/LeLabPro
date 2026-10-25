# Copyright 2025 The HuggingFace Inc. library.
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Give torchcodec the FFmpeg shared libraries it asks for, without root.

torchcodec tries ``libtorchcodec_core{7,6,5,4}.so`` in order and each one has a
DT_NEEDED on that FFmpeg major's plain soname (``libavcodec.so.61`` etc.). A
fresh machine with no system FFmpeg therefore fails at import with
"Could not load libtorchcodec", even though PyAV — a hard dependency here —
already ships a complete FFmpeg 7 inside ``site-packages/av.libs``.

The bundled files are unusable as-is: their SONAMEs are hashed
(``libavcodec-7ee0753d.so.61.19.101``), so nothing matches the plain soname
torchcodec wants, and they reference each other by the hashed name. The shim
directory links *both* spellings, which is enough for the loader.

``LD_LIBRARY_PATH`` is read at process start, so this cannot repair an already
running interpreter: :mod:`lelab.utils.video_backend` still falls back to pyav
at runtime. This module exists so the launcher and any spawned training process
can pick torchcodec up instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import site
import sys
from pathlib import Path

# libavcodec-7ee0753d.so.61.19.101 / libmvec-2-583a17db.28.so
_HASHED = re.compile(r"^(lib[^-.]+)(?:-[\d.]+)?-[0-9a-f]{6,}[._](so(?:\.\d+)*)$")

SHIM_DIR_NAME = ".lelab-ffmpeg-shim"


def probe_torchcodec() -> str | None:
    """Return None when torchcodec can decode, else a short failure reason."""
    if importlib.util.find_spec("torchcodec") is None:
        return "torchcodec is not installed"
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:  # RuntimeError carrying a multi-line traceback
        return " ".join(str(exc).split())[:160]
    return None


def _av_libs_dir() -> Path | None:
    """Locate PyAV's bundled FFmpeg directory (``site-packages/av.libs``)."""
    candidates = {Path(p) / "av.libs" for p in site.getsitepackages() + [site.getusersitepackages()]}
    here = Path(sys.executable).parent.parent / "lib"
    candidates.update(here.glob("python*/site-packages/av.libs"))
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("libavcodec*")):
            return cand
    return None


def build_shim(force: bool = False) -> Path | None:
    """Materialise the shim dir; returns it, or None when there is nothing to do.

    Skipped when torchcodec already loads, because putting the shim on the
    loader path takes precedence over a working system FFmpeg and would pin the
    bundled build instead.
    """
    if not force and probe_torchcodec() is None:
        return None
    src = _av_libs_dir()
    if src is None:
        return None

    shim = src.parent / SHIM_DIR_NAME
    shim.mkdir(exist_ok=True)
    for lib in sorted(src.glob("lib*.so*")):
        if not lib.is_file():
            continue
        # Exact name: satisfies the bundled libraries' own cross-references.
        (shim / lib.name).symlink_to(lib.resolve()) if not (shim / lib.name).exists() else None
        m = _HASHED.match(lib.name)
        if not m:
            continue
        major = re.match(r"so\.(\d+)", m.group(2))
        plain = f"{m.group(1)}.so{('.' + major.group(1)) if major else ''}"
        link = shim / plain
        if not link.exists():
            link.symlink_to(lib.resolve())
    return shim if any(shim.iterdir()) else None


def loader_path(existing: str = "") -> str:
    """LD_LIBRARY_PATH value with the shim appended, if one was built.

    The shim sits directly in ``site-packages`` (see :func:`build_shim`), but
    keep matching a nested copy too so a relocated venv does not silently lose
    the wiring.
    """
    roots = {Path(p) for p in site.getsitepackages()} | {Path(sys.executable).parent.parent / "lib"}
    candidates: list[str] = []
    for base in roots:
        for shim in (base / SHIM_DIR_NAME, *base.glob(f"*/{SHIM_DIR_NAME}")):
            if shim.is_dir() and str(shim) not in candidates:
                candidates.append(str(shim))
    parts = [p for p in (existing.split(os.pathsep) if existing else []) if p]
    parts += [c for c in candidates if c not in parts]
    return os.pathsep.join(parts)


def ffmpeg_version() -> str | None:
    try:
        from torchcodec._core.ops import _get_json_ffmpeg_library_versions

        return json.loads(_get_json_ffmpeg_library_versions()).get("ffmpeg_version")
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ensure", action="store_true", help="build the shim if torchcodec needs it")
    parser.add_argument("--print-path", action="store_true", help="print LD_LIBRARY_PATH to use")
    args = parser.parse_args(argv)

    if args.ensure:
        shim = build_shim()
        print(f"shim built at {shim}" if shim else "torchcodec already loads; no shim needed")
    if args.print_path:
        print(loader_path())
    reason = probe_torchcodec()
    if reason:
        print(f"torchcodec still unavailable: {reason} (pyav fallback will be used)")
        return 1
    print(f"torchcodec OK, FFmpeg {ffmpeg_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
