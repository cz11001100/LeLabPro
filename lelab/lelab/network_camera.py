"""Network (RTSP/HTTP) camera sources for LeLab.

lerobot's `OpenCVCamera` assumes a local device: it insists that CAP_PROP_FPS and
CAP_PROP_FRAME_WIDTH writes take effect (a live RTSP stream ignores them and
reports what the camera sends), and it treats a stalled read as fatal. On the
network both assumptions break — a Wi-Fi drop takes the background read thread
down with it, and `cv2.VideoCapture` never recovers on its own.

`NetworkCamera` subclasses the OpenCV camera to adopt the stream's real
properties, keep serving the last good frame while the link is down, and reopen
the stream from a background thread so the recording loop keeps its cadence.
Registering the `network` camera type also lets the same config reach the
inference subprocess through lerobot's CLI parser.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from lerobot.cameras.configs import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)

NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "rtmps://", "http://", "https://", "udp://", "tcp://")

# How long a frozen frame may stand in for a live one before the caller is told the
# camera is really gone. Long enough to ride out a Wi-Fi roam, short enough that a
# dead camera doesn't quietly record minutes of a still image.
MAX_STALE_MS = 30_000
_RECONNECT_DELAYS_S = (1.0, 2.0, 5.0, 10.0)
_RECONNECT_MAX_ATTEMPTS = 12
# Ticks a preview feed tolerates without a frame before it closes the response
# (10 fps → ~30s, which outlasts one FFmpeg reopen attempt).
MAX_PREVIEW_MISSES = 300


class NetworkCameraError(RuntimeError):
    """A URL source could not be opened or produced no frames."""


def is_network_source(value: Any) -> bool:
    """True for a stream URL, as opposed to a cv2 index or a /dev/video path."""
    return isinstance(value, str) and value.strip().lower().startswith(NETWORK_SCHEMES)


def normalize_url(value: Any) -> str:
    """Validate a pasted stream URL and reject the `Path()` mangling.

    `Path("rtsp://host/live")` collapses the double slash to `rtsp:/host/live`,
    which OpenCV cannot open, so fail here instead of inside a live session.
    """
    if value is None:
        raise NetworkCameraError("camera URL is empty")
    url = str(value).strip().strip("'\"")
    if not url:
        raise NetworkCameraError("camera URL is empty")
    if "://" not in url:
        raise NetworkCameraError(f"`{url}` is missing a scheme — e.g. rtsp://192.168.1.20:554/stream1")
    scheme = url.split("://", 1)[0].lower() + "://"
    if scheme not in NETWORK_SCHEMES:
        raise NetworkCameraError(f"`{scheme}` is not a stream scheme — expected one of {', '.join(NETWORK_SCHEMES)}")
    if not url[len(scheme) :].strip("/"):
        raise NetworkCameraError(f"`{url}` has no host after the scheme")
    return url


def redact(url: Any) -> str:
    """Hide credentials so `rtsp://user:pass@host` never reaches a log or the UI."""
    return re.sub(r"^([a-zA-Z]+://)[^/@:]+:[^/@]+@", r"\1***@", str(url))


def ensure_stream_env() -> None:
    """Pin OpenCV's FFmpeg backend to TCP with a bounded connect timeout.

    UDP — the RTSP default — drops packets on Wi-Fi and FFmpeg renders each loss as
    grey blocks, which is fatal for a training dataset. The options are read per
    `VideoCapture` creation, so this runs before every open. A value the user
    already exported wins, since it may be tuned for a camera this default suits.
    """
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        # stimeout and max_delay are microseconds.
        "rtsp_transport;tcp|stimeout;8000000|fflags;nobuffer|max_delay;500000|flags;low_delay",
    )


def _prop(capture: Any, prop: int) -> float:
    try:
        return float(capture.get(prop))
    except Exception:
        return 0.0


_DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "rtmp": 1935, "http": 80, "https": 443}


def _stream_port(url: str) -> int:
    parsed_scheme = url.split("://", 1)[0].lower()
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.port:
        return int(parsed.port)
    return _DEFAULT_PORTS.get(parsed_scheme, 554)


def check_reachability(url: str, timeout_s: float = 4.0) -> None:
    """Open a plain TCP socket to the camera first.

    FFmpeg's socket timeouts cover the *read* phase, not the connect, so a wrong
    IP otherwise blocks a `VideoCapture` open for minutes — long enough to hang
    the picker and every retry behind it. A refused or unroutable host is also
    the most common mistake, and this names the host and port instead of leaving
    a generic FFmpeg failure.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise NetworkCameraError(f"`{redact(url)}` has no host to connect to")
    if parsed.scheme.startswith("udp"):
        return  # connectionless: nothing to probe before the stream itself starts
    port = _stream_port(url)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except OSError as e:
        raise NetworkCameraError(
            f"{host}:{port} is not reachable ({type(e).__name__}: {e.strerror or e}). "
            "Check the camera's IP and port, and that this machine is on the same network."
        ) from e


def _read_stream_properties(url: str, read_timeout_s: float) -> dict[str, Any]:
    """Open the capture and take one frame; blocking is why `probe` runs this
    off-thread."""
    import cv2

    ensure_stream_env()
    cap = cv2.VideoCapture(url, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        raise NetworkCameraError(f"could not open {redact(url)}")
    try:
        width = int(round(_prop(cap, cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = _prop(cap, cv2.CAP_PROP_FPS)

        # Some cameras answer DESCRIBE and then never send a frame, so an open
        # capture is not yet proof the stream works.
        deadline = time.monotonic() + read_timeout_s
        frame = None
        while time.monotonic() < deadline and frame is None:
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
            else:
                time.sleep(0.05)
        if frame is None:
            raise NetworkCameraError(f"{redact(url)} opened but sent no frame within {read_timeout_s:.0f}s")
        if (frame.shape[1], frame.shape[0]) != (width, height):
            width, height = int(frame.shape[1]), int(frame.shape[0])
    finally:
        cap.release()

    return {"width": width, "height": height, "fps": round(fps) if fps > 0 else None}


def probe(url: str, *, timeout_s: float = 15.0, read_timeout_s: float = 8.0) -> dict[str, Any]:
    """Open a stream once and report what it actually delivers.

    lerobot requires width/height/fps on any camera attached to a robot, and those
    numbers become the dataset's image features, so they have to match the stream:
    an IP camera won't accept a resolution it wasn't configured to send.

    The open runs on a daemon thread because `cv2.VideoCapture` can't be interrupted
    from the outside; without the cap, a camera that accepts the connection but
    never streams would block the caller indefinitely.
    """
    url = normalize_url(url)
    check_reachability(url)

    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = _read_stream_properties(url, read_timeout_s)
        except BaseException as e:  # noqa: BLE001 - relayed to the caller verbatim
            box["error"] = e

    thread = threading.Thread(target=work, name="camera-probe", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise NetworkCameraError(
            f"{redact(url)} accepted the connection but never finished opening within {timeout_s:.0f}s "
            "— another client may be holding the camera, or the stream path is wrong."
        )
    if "error" in box:
        raise box["error"]
    return box["result"]


@CameraConfig.register_subclass("network")
@dataclass
class NetworkCameraConfig(OpenCVCameraConfig):
    """An RTSP/HTTP/MJPEG stream opened through OpenCV's FFmpeg backend.

    `backend` stays ANY because none of the enumerated device backends (V4L2,
    DSHOW, ...) can open a URL; letting cv2 choose is what routes a URL to FFmpeg.
    """

    index_or_path: str
    warmup_s: int = 3
    backend: Cv2Backends = Cv2Backends.ANY

    def __post_init__(self) -> None:
        self.index_or_path = normalize_url(self.index_or_path)
        super().__post_init__()
        self.backend = Cv2Backends.ANY


class NetworkCamera(OpenCVCamera):
    """An OpenCV camera on a URL: lenient about the stream, persistent about the link."""

    def __init__(self, config: NetworkCameraConfig):
        super().__init__(config)
        self.url: str = str(config.index_or_path)
        self._held: Any = None
        self._held_at: float = 0.0
        self._reconnecting: bool = False
        self._closed: bool = False
        self._guard = threading.Lock()
        self._lost_since: float | None = None
        self.stats: dict[str, Any] = {"reconnects": 0, "outages": 0, "last_error": ""}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({redact(self.url)})"

    def connect(self, warmup: bool = True) -> None:
        self._closed = False
        ensure_stream_env()
        super().connect(warmup=warmup)

    def _read_loop(self) -> None:
        """Stop quietly when the stream dies.

        The base loop raises after ~11 consecutive failures, which for a network
        camera is only a dropped link — and an uncaught exception in a thread
        prints a traceback that looks like a crash while the reconnect guard is
        already handling it.
        """
        try:
            super()._read_loop()
        except RuntimeError as e:
            logger.warning(f"📡 NETWORK CAMERA: {self} read loop stopped ({e}) — reopening in the background")

    def disconnect(self) -> None:
        """Stop reconnecting.

        A session teardown calls this on purpose, so an in-flight reconnect must
        not drag the stream back open afterwards.
        """
        self._closed = True
        super().disconnect()

    def _configure_capture_settings(self) -> None:
        """Adopt what the stream reports instead of failing on a refused write.

        The base class raises when `set(CAP_PROP_FPS)` doesn't stick, and FFmpeg
        always refuses those writes on a live encoder — so the strict check would
        kill every RTSP connection.
        """
        import cv2

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        actual_w = int(round(_prop(self.videocapture, cv2.CAP_PROP_FRAME_WIDTH)))
        actual_h = int(round(_prop(self.videocapture, cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = _prop(self.videocapture, cv2.CAP_PROP_FPS)

        if actual_w and actual_h and (actual_w, actual_h) != (self.width, self.height):
            logger.warning(
                f"📐 NETWORK CAMERA: {self} streams {actual_w}x{actual_h}, not the configured "
                f"{self.width}x{self.height} — recording at the stream size"
            )
            self.width, self.height = actual_w, actual_h
            self.capture_width, self.capture_height = actual_w, actual_h

        if actual_fps > 0 and self.fps and not math.isclose(self.fps, actual_fps, rel_tol=0.1):
            logger.info(f"⏱️ NETWORK CAMERA: {self} stream runs at {actual_fps:.2f}fps (configured {self.fps})")
            self.fps = round(actual_fps)

    # -- frame access, with the last good frame as the fallback ----------------
    def _hold(self, frame: Any) -> Any:
        self._held = frame
        self._held_at = time.perf_counter()
        if self._lost_since is not None:
            self.stats["outages"] += 1
            logger.info(
                f"✅ NETWORK CAMERA: {redact(self.url)} resumed after "
                f"{time.perf_counter() - self._lost_since:.1f}s (reconnects: {self.stats['reconnects']})"
            )
            self._lost_since = None
        return frame

    def _stale(self, error: Exception) -> Any:
        """Serve the held frame while the link recovers, or give up once it is old."""
        if self._held is None or self._closed:
            raise error
        if self._lost_since is None:
            self._lost_since = time.perf_counter()
            self.stats["last_error"] = str(error)
            logger.warning(f"📡 NETWORK CAMERA: {redact(self.url)} went silent ({error}) — holding the last frame")
        age_ms = (time.perf_counter() - self._held_at) * 1e3
        if age_ms > MAX_STALE_MS:
            raise TimeoutError(
                f"{self} has had no frame for {age_ms / 1000:.0f}s (max {MAX_STALE_MS / 1000:.0f}s)"
            ) from error
        self._schedule_reconnect()
        return self._held

    def async_read(self, timeout_ms: float = 200) -> Any:
        try:
            return self._hold(super().async_read(timeout_ms=timeout_ms))
        except Exception as e:
            return self._stale(e)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._hold(super().read(*args, **kwargs))
        except Exception as e:
            return self._stale(e)

    def read_latest(self, max_age_ms: int = 500) -> Any:
        try:
            return self._hold(super().read_latest(max_age_ms=max_age_ms))
        except Exception as e:
            return self._stale(e)

    # -- reconnect -------------------------------------------------------------
    def _schedule_reconnect(self) -> None:
        """Reopen off the caller's thread — a connect() costs seconds, and the
        30 fps control loop cannot spend them waiting."""
        with self._guard:
            if self._reconnecting or self._closed:
                return
            self._reconnecting = True
        threading.Thread(target=self._reconnect, name="network-camera-reconnect", daemon=True).start()

    def _reconnect(self) -> None:
        # `delays * 3` so the 10s tail repeats until the attempt cap, which keeps
        # a powered-off camera recoverable once someone turns it back on.
        delays = _RECONNECT_DELAYS_S * 3
        try:
            for attempt, delay in enumerate(delays, start=1):
                if self._closed:
                    return  # the session ended while we were backing off
                if attempt > _RECONNECT_MAX_ATTEMPTS:
                    self.stats["last_error"] = "gave up reconnecting"
                    logger.error(
                        f"❌ NETWORK CAMERA: {redact(self.url)} unreachable after {attempt - 1} attempts"
                    )
                    return
                try:
                    try:
                        super().disconnect()
                    except Exception:
                        pass  # already half-dead; a successful connect() is what matters
                    self.connect()
                    self.stats["reconnects"] += 1
                    logger.info(f"🔁 NETWORK CAMERA: {redact(self.url)} reopened on attempt {attempt}")
                    return
                except Exception as e:
                    self.stats["last_error"] = str(e)
                    logger.warning(
                        f"⚠️ NETWORK CAMERA: {redact(self.url)} reopen failed (attempt {attempt}): {e} "
                        f"— retrying in {delay:.0f}s"
                    )
                    time.sleep(delay)
        finally:
            with self._guard:
                self._reconnecting = False

    def status(self) -> dict[str, Any]:
        """Health snapshot for the UI: is the stream live right now?"""
        age_s = (time.perf_counter() - self._held_at) if self._held_at else None
        return {
            "url": redact(self.url),
            "connected": bool(self.is_connected),
            "reconnecting": self._reconnecting,
            "last_frame_age_s": round(age_s, 2) if age_s is not None else None,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            **self.stats,
        }


def install_camera_factory_hook() -> None:
    """Teach lerobot's camera factory about `type: network`.

    `make_cameras_from_configs` is a hard-coded if/elif over type names, so a new
    camera type is rejected even though its config class is registered. The hook
    keeps every existing type on lerobot's own code path and only intercepts
    `network`. Robot modules import the factory by name, so already-bound
    references get re-pointed too — otherwise a robot imported before this hook
    would keep raising on a network camera.
    """
    import sys

    import lerobot.cameras.utils as camera_utils

    original = camera_utils.make_cameras_from_configs
    if getattr(original, "_lelab_network_hook", False):
        return

    def make_cameras_from_configs(camera_configs):  # type: ignore[no-redef]
        network = {k: c for k, c in camera_configs.items() if getattr(c, "type", None) == "network"}
        others = {k: c for k, c in camera_configs.items() if getattr(c, "type", None) != "network"}
        cameras = dict(original(others)) if others else {}
        for key, cfg in network.items():
            cameras[key] = NetworkCamera(cfg)
        return cameras

    make_cameras_from_configs._lelab_network_hook = True
    camera_utils.make_cameras_from_configs = make_cameras_from_configs
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, "make_cameras_from_configs", None) is original:
                module.make_cameras_from_configs = make_cameras_from_configs
        except Exception:
            continue


def camera_source(camera_data: dict) -> tuple[str, Any]:
    """Split a LeLab camera dict into (`type`, source) for lerobot's config.

    One place decides whether a camera is a device index or a stream URL, so
    recording, inference and the CLI arg builder can't drift apart on what counts
    as a network source.
    """
    source = camera_data.get("url") or camera_data.get("camera_index")
    if is_network_source(source):
        return "network", str(source).strip()
    return camera_data.get("type", "opencv"), source


def preview_stream(url: str):
    """A self-healing stand-in for `cv2.VideoCapture` on a stream URL.

    RTSP allows several concurrent client sessions, so LeLab can open its own copy
    for a live preview while a recording or inference run holds the primary one —
    unlike a UVC device, which takes a single opener. `read()`/`release()` match
    cv2's so callers don't need to know which kind of source they got.
    """

    class _Stream:
        def __init__(self, target: str) -> None:
            self.url = normalize_url(target)
            self.cap = None
            self._next_open = 0.0

        def read(self) -> tuple[bool, Any]:
            import cv2

            if self.cap is None:
                if time.monotonic() < self._next_open:
                    return False, None
                ensure_stream_env()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_ANY)
                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
                    self._next_open = time.monotonic() + 2.0
                    return False, None
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return True, frame
            # A failed read means the session is gone for good; reopen on the next call.
            self.cap.release()
            self.cap = None
            self._next_open = time.monotonic() + 1.0
            return False, None

        def release(self) -> None:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    return _Stream(url)


def preview_frames(url: str, fps: float = 10.0, max_misses: int | None = None):
    """Yield multipart-MJPEG chunks from a stream URL until the client leaves.

    Browsers render `multipart/x-mixed-replace` straight in an <img>. A failed
    read just skips a tick — the stream reopens itself, so a Wi-Fi drop shows as
    a frozen frame rather than a broken image.

    The miss cap exists because a server only learns about a closed browser tab
    between chunks: a generator that never produces one holds its request thread
    forever. So an unreadable stream ends on its own instead.
    """
    import cv2

    stream = preview_stream(url)
    interval = 1.0 / fps
    misses_allowed = MAX_PREVIEW_MISSES if max_misses is None else max_misses
    misses = 0
    try:
        while misses < misses_allowed:
            ok, frame = stream.read()
            if ok and frame is not None:
                misses = 0
                encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if encoded:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            else:
                misses += 1
            time.sleep(interval)
    finally:
        stream.release()
