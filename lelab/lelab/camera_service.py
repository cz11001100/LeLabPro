"""Standalone camera service: preview + health check, with no recording session.

LeLab's ``/camera-feed`` only streams while a recording owns the cameras, so outside
a session the UI shows nothing at all, and a black tile never tells you *why*. This
module owns the cameras whenever nothing else does:

- enumerate capture nodes together with the xHCI controller each one sits on,
- open a node on demand, share the handle between viewers, release it when idle,
- probe one node or a whole set and report fps / the concrete failure reason,
- render any failure into a JPEG frame so an ``<img>`` always shows readable text,
- refuse a doomed camera set before the recorder half-starts (:func:`preflight`).

Why the reasons matter: UVC isochronous streams are *reserved* from the camera
descriptor, not from the actual bitrate, so ``fourcc``/``fps``/resolution changes
cannot free bandwidth that the descriptor already claims. How many such reservations
one controller accepts is decided by the host controller's firmware — this AMD
(0000:05:00.4) laptop allows exactly one UVC stream per controller, while cz's Intel
box streams three cameras off one bus. So nothing here hardcodes a limit: the health
check measures it, and :func:`preflight` measures it again before a recording starts.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

ROBOTS_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "robots"
PREVIEW_FPS = 10.0
PROBE_SECONDS = 1.5
HEALTH_TTL = 60.0
READER_FPS = 15.0
IDLE_GRACE_S = 4.0
MAX_LIFE_S = 600.0
DEFAULT_SIZE = {"width": 640, "height": 480, "fps": 30}

_lock = threading.Lock()
_handles: dict[str, "_Preview"] = {}
_health_cache: dict[str, Any] = {"at": 0.0, "rows": None}


class Device:
    """One V4L2 capture node plus where it hangs off the USB tree."""

    def __init__(self, sysfs_path: str) -> None:
        self.node = os.path.basename(sysfs_path)
        self.index = int(self.node.replace("video", ""))
        self.name = _read(f"{sysfs_path}/name") or self.node
        dev = os.path.realpath(f"{sysfs_path}/device")
        self.topology = dev[dev.index("/usb") :] if "/usb" in dev else dev
        pci = [p for p in dev.split("/") if p.startswith("0000:")]
        self.controller = pci[-1] if pci else "?"
        self.by_path = next(
            (p for p in glob.glob("/dev/v4l/by-path/*-video-index0") if os.path.realpath(p).endswith(f"/{self.node}")),
            "",
        )

    @property
    def target(self) -> str:
        return f"/dev/{self.node}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node": self.target,
            "name": self.name,
            "controller": self.controller,
            "topology": self.topology,
            "by_path": self.by_path,
        }


def _read(path: str) -> str:
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def list_devices() -> list[Device]:
    """Capture nodes only (``index=0``); metadata/secondary nodes are skipped."""
    out: list[Device] = []
    for path in glob.glob("/sys/class/video4linux/video*"):
        try:
            if _read(f"{path}/index") != "0":
                continue
            out.append(Device(path))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda d: d.index)


def find_device(key: str) -> Device | None:
    for dev in list_devices():
        if str(key) in (dev.target, dev.node, str(dev.index), dev.by_path):
            return dev
    return None


def camera_specs() -> dict[str, dict[str, Any]]:
    """Camera name → config, collected from every saved robot record."""
    specs: dict[str, dict[str, Any]] = {}
    for path in sorted(ROBOTS_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for cam in record.get("cameras", []) or []:
            if isinstance(cam, dict) and cam.get("name"):
                specs.setdefault(str(cam["name"]), {**cam, "robot": record.get("name", path.stem)})
    return specs


def resolve(key: str) -> tuple[str | int | None, dict[str, Any]]:
    """Map a UI camera key to something ``cv2.VideoCapture`` accepts.

    Accepts a ``/dev/...`` path, a node name, a bare index, or a camera *name* from a
    saved robot config (then its width/height/fps/fourcc travel with it).
    """
    if not key:
        return None, {}
    if str(key).startswith("/dev/"):
        return str(key), {}
    if re.fullmatch(r"\d+", str(key)):
        return int(key), {}
    spec = camera_specs().get(str(key))
    if spec:
        target = spec.get("camera_index", spec.get("index_or_path"))
        cfg = {k: spec[k] for k in ("width", "height", "fps", "fourcc") if spec.get(k)}
        return target, cfg
    dev = find_device(str(key))
    return (dev.index, {}) if dev else (None, {})


def active_owner() -> str | None:
    """Which session already owns the cameras, if any (we must not fight it)."""
    from . import record as _record
    from . import rollout as _rollout
    from . import teleoperate as _teleop

    if getattr(_record, "recording_active", False):
        return "recording"
    if getattr(_rollout, "inference_active", False):
        return "inference"
    if getattr(_teleop, "teleoperation_active", False):
        return "teleoperation"
    return None


def bandwidth_refusals(window_s: float = 30.0) -> int:
    """Kernel-side isochronous rejections in the recent kernel log."""
    stamp = time.strftime("%H:%M:%S", time.localtime(time.time() - window_s))
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--since", stamp, "-q", "--no-pager"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if "Not enough bandwidth" in line)


def annotated_frame(lines: list[str], size: tuple[int, int] = (640, 480)) -> bytes:
    """A JPEG holding a message, so ``<img>`` never goes silently black."""
    img = np.full((size[1], size[0], 3), 26, dtype=np.uint8)
    y = 46
    for line in lines:
        cv2.putText(img, line[:80], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        y += 30
        if y > size[1] - 20:
            break
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


def _jpeg_chunk(jpeg: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


class _Preview:
    """One open device, read by a private thread, shared by every viewer.

    The thread owns the camera, not the HTTP generator: a viewer that vanishes
    without its generator ever finalising cannot hold the device hostage (that leak
    made the next camera on the same controller unopenable). The reader exits on its
    own once nobody has looked at the stream for ``IDLE_GRACE_S``, or after
    ``MAX_LIFE_S``, and releases the device in a ``finally``.
    """

    def __init__(self, target: str | int, cfg: dict[str, Any]) -> None:
        self.target = target
        self.cfg = cfg
        self.clients = 0
        self.last_seen = time.time()
        self.started_at = time.time()
        self.jpeg: bytes | None = None
        self.error_lines: list[str] = []
        self.stop = threading.Event()
        self.cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
        if cfg.get("fourcc"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(cfg["fourcc"]).upper()))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("width", DEFAULT_SIZE["width"])))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("height", DEFAULT_SIZE["height"])))
        self.cap.set(cv2.CAP_PROP_FPS, float(cfg.get("fps", DEFAULT_SIZE["fps"])))
        self.thread = threading.Thread(target=self._loop, name=f"lelab-camera-{target}", daemon=True)

    def _set_error(self, lines: list[str]) -> None:
        self.error_lines = lines
        self.jpeg = annotated_frame(lines)

    def _encode(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            self.jpeg = buf.tobytes()

    def _loop(self) -> None:
        try:
            if not self.cap.isOpened():
                self._set_error(
                    [
                        f"{self.target} 打不开",
                        "被占用：浏览器预览 / OBS / 另一个 LeLab 会话",
                        "或节点已失效（掉线重枚举后 /dev/videoN 会换号）",
                    ]
                )
            else:
                deadline = time.time() + PROBE_SECONDS
                while time.time() < deadline:
                    ok, frame = self.cap.read()
                    if ok:
                        self._encode(frame)
                        break
                    time.sleep(0.02)
                if self.jpeg is None:
                    lines = [f"{self.target} 打开成功但读不到帧", "多半是等时带宽分配被内核拒绝"]
                    refusals = bandwidth_refusals()
                    if refusals:
                        lines.append(f"内核最近 30s 记录 {refusals} 次 Not enough bandwidth")
                    lines.append("解法：每颗 xHCI 只留一只相机 / 减相机 / 换多控制器机器")
                    self._set_error(lines)

            while not self.stop.is_set():
                now = time.time()
                if now - self.last_seen > IDLE_GRACE_S:
                    break
                if now - self.started_at > MAX_LIFE_S:
                    break
                if self.error_lines:
                    time.sleep(1.0)
                    continue
                ok, frame = self.cap.read()
                if ok:
                    self._encode(frame)
                time.sleep(1.0 / READER_FPS)
        finally:
            self.cap.release()
            with _lock:
                if _handles.get(str(self.target)) is self:
                    _handles.pop(str(self.target), None)


def _attach(key: str) -> tuple[_Preview | None, list[str]]:
    """Return a live reader for this camera, starting one if needed."""
    target, cfg = resolve(key)
    if target is None:
        return None, [f"camera '{key}' not found", "use a /dev/videoN node, a bare index,", "or a camera name from a robot config"]
    node = target if isinstance(target, str) else f"/dev/video{target}"
    if not os.path.exists(node) and find_device(node) is None:
        return None, [f"{node} 不存在", "相机没插，或掉线重枚举后编号变了", "打开 /cameras 看现在的归属"]
    with _lock:
        handle = _handles.get(str(target))
        if handle is None:
            handle = _Preview(target, cfg)
            _handles[str(target)] = handle
            handle.thread.start()
        handle.clients += 1
        handle.last_seen = time.time()
        return handle, []


def preview_frames(key: str, fps: float = PREVIEW_FPS) -> Iterator[bytes]:
    """Multipart MJPEG chunks for one camera — works any time, not only while recording."""
    owner = active_owner()
    if owner:
        jpeg = annotated_frame([f"{owner} 正在占用相机", "该会话期间不接管设备", "停掉它再来预览"])
        until = time.time() + MAX_LIFE_S
        while time.time() < until and not (active_owner() is None):
            yield _jpeg_chunk(jpeg)
            time.sleep(1.0)
        return

    handle, why = _attach(key)
    if handle is None:
        jpeg = annotated_frame(why)
        until = time.time() + MAX_LIFE_S
        while time.time() < until:
            yield _jpeg_chunk(jpeg)
            time.sleep(1.0)
        return

    interval = 1.0 / max(fps, 0.5)
    try:
        deadline = time.time() + PROBE_SECONDS + 2.0
        while handle.jpeg is None and time.time() < deadline:
            time.sleep(0.05)
        while not handle.stop.is_set():
            handle.last_seen = time.time()
            if handle.jpeg is not None:
                yield _jpeg_chunk(handle.jpeg)
            time.sleep(interval)
    finally:
        with _lock:
            handle.clients = max(0, handle.clients - 1)
            handle.last_seen = time.time()


def probe(dev: Device, seconds: float = PROBE_SECONDS) -> dict[str, Any]:
    """Open one camera alone and measure it. Call serially — that is the point."""
    info = dev.as_dict()
    cap = cv2.VideoCapture(dev.index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return {**info, "ok": False, "fps": 0.0, "error": "打不开：被占用（浏览器预览/OBS）或节点失效"}
    frames, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            frames += 1
        else:
            time.sleep(0.01)
    cap.release()
    fps = frames / max(time.time() - t0, 1e-6)
    if frames == 0:
        return {**info, "ok": False, "fps": 0.0, "error": "开流失败：等时带宽被拒，或模组掉线"}
    return {**info, "ok": True, "fps": round(fps, 1), "error": ""}


def concurrent_probe(specs: list[tuple[Any, dict[str, Any]]], seconds: float = 2.5) -> list[dict[str, Any]]:
    """Open several cameras at once, then read round-robin: which ones deliver frames.

    This is the measurement behind :func:`preflight` — whether one controller can
    carry two cameras is decided by the host controller's firmware, not by anything
    in Python, so we ask the hardware instead of assuming (this AMD laptop refuses a
    second stream; cz's Intel box runs three on one bus).
    """
    caps: list[tuple[Any, cv2.VideoCapture]] = []
    for target, cfg in specs:
        cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
        if cfg.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(cfg["fourcc"]).upper()))
        if cfg.get("width"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["width"]))
        if cfg.get("height"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["height"]))
        if cfg.get("fps"):
            cap.set(cv2.CAP_PROP_FPS, float(cfg["fps"]))
        caps.append((target, cap))
    counts = {t: 0 for t, _ in caps}
    errs: dict[Any, str] = {}
    end = time.time() + seconds
    while time.time() < end:
        for t, cap in caps:
            if not cap.isOpened():
                errs.setdefault(t, "打不开：节点失效或已被占用")
                continue
            ok, _ = cap.read()
            if ok:
                counts[t] += 1
        time.sleep(0.005)
    rows = [
        {
            "index": t,
            "node": t if isinstance(t, str) else f"/dev/video{t}",
            "frames": counts[t],
            "fps": round(counts[t] / seconds, 1),
            "error": errs.get(t, ""),
        }
        for t, _ in caps
    ]
    for _, cap in caps:
        cap.release()
    return rows


def health(refresh: bool = False, concurrency: bool = False) -> dict[str, Any]:
    """Per-camera verdict plus the controller grouping that explains the failures."""
    devices = list_devices()
    groups: dict[str, list[str]] = {}
    for dev in devices:
        groups.setdefault(dev.controller, []).append(dev.target)

    owner = active_owner()
    if owner:
        return {
            "status": "busy",
            "owner": owner,
            "message": f"{owner} 正在占用相机，体检已跳过（停掉该会话后再跑）",
            "devices": [dev.as_dict() for dev in devices],
            "controllers": groups,
        }

    if concurrency:
        return {
            "status": "concurrency",
            "devices": [dev.as_dict() for dev in devices],
            "controllers": groups,
            "concurrent": concurrent_probe([(dev.index, {}) for dev in devices]),
            "kernel_bandwidth_refusals": bandwidth_refusals(120.0),
        }

    fresh = _health_cache["rows"] is not None and (time.time() - _health_cache["at"]) < HEALTH_TTL
    if refresh or not fresh:
        _health_cache["rows"] = [probe(dev) for dev in devices]
        _health_cache["at"] = time.time()
    rows = _health_cache["rows"]

    crowded = {ctrl: nodes for ctrl, nodes in groups.items() if len(nodes) > 1}
    suggested = sorted(min(nodes) for nodes in groups.values())
    verdict = (
        "每只相机各占一颗控制器。"
        if not crowded
        else "共享同一颗控制器的相机："
        + "；".join(f"{ctrl} → {' + '.join(nodes)}" for ctrl, nodes in crowded.items())
        + f"。能否同时开流由主机控制器决定（不同机器结论不同），点“并发验证”用实测回答；"
        + f"若实测有路拿不到帧，最稳的组合是每颗控制器一只：{' + '.join(suggested)}"
    )
    return {
        "status": "ok",
        "devices": rows,
        "controllers": groups,
        "working": [r["node"] for r in rows if r["ok"]],
        "broken": [{"node": r["node"], "error": r["error"]} for r in rows if not r["ok"]],
        "same_controller": crowded,
        "suggested_set": suggested,
        "kernel_bandwidth_refusals": bandwidth_refusals(120.0),
        "verdict": verdict,
        "probed_at": _health_cache["at"],
    }


PREFLIGHT_MODE = os.environ.get("LELAB_CAMERA_PREFLIGHT", "probe").strip().lower()


def preflight(cameras: dict[str, Any], seconds: float = 2.5) -> str | None:
    """Refuse a camera set that cannot stream together — measured, not guessed.

    ``cameras`` is the LeLab request mapping (name → config with ``camera_index``).
    Whether one controller can carry two UVC streams is decided by the host
    controller's firmware, so the default mode opens the requested cameras at once
    for ~2.5 s and looks at who actually delivers frames. This AMD laptop refuses a
    second stream; cz's Intel box runs three on one bus — the same code says "go"
    there and "stop" here because it asks the hardware.

    ``LELAB_CAMERA_PREFLIGHT``: ``probe`` (default), ``group`` (topology-only, the
    strict behaviour that matches this laptop), ``off`` (no check).
    """
    if PREFLIGHT_MODE == "off":
        return None
    specs: list[tuple[Any, dict[str, Any]]] = []
    groups: dict[str, list[str]] = {}
    unknown: list[str] = []
    for name, cfg in (cameras or {}).items():
        if not isinstance(cfg, dict):
            continue
        target, size = resolve(str(cfg.get("camera_index", cfg.get("index_or_path", ""))))
        if target is None:
            unknown.append(str(name))
            continue
        node = target if isinstance(target, str) else f"/dev/video{target}"
        dev = find_device(node)
        if dev is None:
            unknown.append(f"{name}({node})")
            continue
        groups.setdefault(dev.controller, []).append(f"{name}({node})")
        specs.append((dev.index, {**DEFAULT_SIZE, **size}))
    if unknown:
        return f"这些相机解析不到设备节点：{', '.join(unknown)}。编号可能已漂移，重新选一次相机（见 /cameras）。"
    if len(specs) < 2:
        return None

    crowded = {ctrl: names for ctrl, names in groups.items() if len(names) > 1}
    if PREFLIGHT_MODE == "group":
        if not crowded:
            return None
        rows = []
    else:
        rows = concurrent_probe(specs, seconds)

    silent = [r for r in rows if r.get("frames", 0) == 0]
    if PREFLIGHT_MODE != "group" and not silent:
        return None

    lines = ["这些相机无法同时开流："]
    if PREFLIGHT_MODE == "group":
        lines.append("  按拓扑分组：同一颗 USB 控制器上有多只相机，而本机按分组判定只允许 1 路。")
    else:
        for r in silent:
            lines.append(f"  {r['node']} 并发实测 0 帧（{r['error'] or '等时带宽被主机控制器拒绝'}）")
    if crowded:
        lines += [f"  共享控制器 {ctrl}：{' + '.join(names)}" for ctrl, names in crowded.items()]
    lines.append("把其中一只换到另一颗控制器的物理口、减少相机数量，或用多控制器的机器采数（详见 /cameras）。")
    return "\n".join(lines)
