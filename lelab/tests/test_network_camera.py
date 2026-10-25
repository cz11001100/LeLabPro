"""Network (RTSP/URL) camera source tests — no real camera required."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

URL = "rtsp://admin:s3cret@192.168.1.64:554/cam/realmonitor?channel=1&subtype=1"


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


def test_normalize_url_accepts_stream_schemes() -> None:
    from lelab.network_camera import normalize_url

    assert normalize_url(f" {URL} ") == URL
    assert normalize_url("http://192.168.1.20:8080/video") == "http://192.168.1.20:8080/video"


@pytest.mark.parametrize(
    "value",
    [
        "",
        None,
        "192.168.1.64:554/live",  # no scheme
        "file:///etc/passwd",  # not a stream scheme
        "rtsp:/192.168.1.64/live",  # what Path() leaves behind
        "rtsp://",  # no host
    ],
)
def test_normalize_url_rejects_non_streams(value) -> None:
    from lelab.network_camera import NetworkCameraError, normalize_url

    with pytest.raises(NetworkCameraError):
        normalize_url(value)


def test_normalize_url_survives_a_path_wrapper() -> None:
    """`Path("rtsp://host/live")` collapses the double slash and can never open."""
    from lelab.network_camera import NetworkCameraError, normalize_url

    with pytest.raises(NetworkCameraError):
        normalize_url(str(Path(URL)))


def test_redact_hides_credentials() -> None:
    from lelab.network_camera import redact

    masked = redact(URL)
    assert "s3cret" not in masked
    assert "admin" not in masked.split("://")[1].split("@")[0]
    assert masked.startswith("rtsp://***@192.168.1.64")


def test_is_network_source_separates_urls_from_indices() -> None:
    from lelab.network_camera import is_network_source

    assert is_network_source(URL)
    assert is_network_source("RTSP://host/live")
    assert not is_network_source(0)
    assert not is_network_source("/dev/video0")


def test_camera_source_splits_stream_from_device() -> None:
    from lelab.network_camera import camera_source

    assert camera_source({"type": "network", "url": URL}) == ("network", URL)
    # A URL pasted into the index field is still a network source.
    assert camera_source({"type": "opencv", "camera_index": URL})[0] == "network"
    assert camera_source({"type": "opencv", "camera_index": 2}) == ("opencv", 2)
    assert camera_source({"camera_index": 0}) == ("opencv", 0)


# ---------------------------------------------------------------------------
# Config building (recording path)
# ---------------------------------------------------------------------------


def test_build_camera_configs_uses_probed_stream_values(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.network_camera as nc
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr(nc, "probe", lambda url: {"width": 1280, "height": 720, "fps": 15})
    cameras = {"side": {"type": "network", "url": URL, "width": 640, "height": 480, "fps": 30}}

    configs = _build_camera_configs(cameras, Cv2Backends.V4L2)

    cfg = configs["side"]
    assert cfg.type == "network"
    assert cfg.index_or_path == URL
    # The platform pin (V4L2) cannot open a URL; cv2 must pick FFmpeg.
    assert cfg.backend == Cv2Backends.ANY
    assert (cfg.width, cfg.height, cfg.fps) == (1280, 720, 15)


def test_build_camera_configs_keeps_ui_fps_when_stream_reports_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.network_camera as nc
    from lelab.record import _build_camera_configs

    monkeypatch.setattr(nc, "probe", lambda url: {"width": 0, "height": 0, "fps": None})
    cameras = {"up": {"type": "network", "url": URL, "width": 640, "height": 480, "fps": 25}}

    cfg = _build_camera_configs(cameras, None)["up"]
    assert (cfg.width, cfg.height, cfg.fps) == (640, 480, 25)


def test_build_camera_configs_unreachable_stream_names_the_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.network_camera as nc
    from lelab.record import _build_camera_configs

    def boom(url):
        raise nc.NetworkCameraError("could not open rtsp://***@192.168.1.64/live")

    monkeypatch.setattr(nc, "probe", boom)
    cameras = {"side": {"type": "network", "url": URL}}

    with pytest.raises(RuntimeError) as excinfo:
        _build_camera_configs(cameras, None)

    message = str(excinfo.value)
    assert "side" in message
    assert "not reachable" in message
    # The password must not ride along into the UI's error banner.
    assert "s3cret" not in message


def test_build_camera_configs_still_handles_usb_indices() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"up": {"type": "opencv", "camera_index": 4, "width": 640, "height": 480, "fps": 30}}
    cfg = _build_camera_configs(cameras, Cv2Backends.V4L2)["up"]

    assert cfg.type == "opencv"
    assert cfg.index_or_path == 4
    assert cfg.backend == Cv2Backends.V4L2


# ---------------------------------------------------------------------------
# CLI arg building (inference subprocess path)
# ---------------------------------------------------------------------------


def test_format_cameras_arg_quotes_a_stream_url() -> None:
    from lelab.rollout import _format_cameras_arg

    arg = _format_cameras_arg({"side": {"type": "network", "url": URL, "width": 640, "height": 480, "fps": 15}})

    assert "{side: {type: network, index_or_path: " in arg
    assert f'"{URL}"' in arg
    assert "url:" not in arg.replace("rtsp://", "")  # the raw `url` key must not leak


def test_format_cameras_arg_maps_usb_index_to_index_or_path() -> None:
    from lelab.rollout import _format_cameras_arg

    arg = _format_cameras_arg({"up": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}})

    assert "index_or_path: 0" in arg
    assert "camera_index" not in arg


def test_camera_arg_in_localize_carries_a_string_url() -> None:
    from lelab.localize import camera_arg

    cams = camera_arg([{"name": "side", "type": "network", "url": URL, "width": 640, "height": 480, "fps": 15}])

    assert cams["side"]["type"] == "network"
    assert cams["side"]["index_or_path"] == URL


def test_draccus_keeps_a_network_url_a_string() -> None:
    """The `opencv` config types index_or_path `int | Path`, which silently
    rewrites `rtsp://host/live` to `rtsp:/host/live`; `network` must not."""
    import draccus
    from lelab.network_camera import install_camera_factory_hook  # registers the type

    install_camera_factory_hook()
    from lerobot.robots.so_follower import SO101FollowerConfig

    arg = '{side: {type: network, index_or_path: "%s", width: 640, height: 480, fps: 15}}' % URL
    cfg = draccus.parse(
        config_class=SO101FollowerConfig,
        args=[f"--cameras={arg}", "--port=/dev/ttyNONE", "--id=net-test"],
    )

    assert isinstance(cfg.cameras["side"].index_or_path, str)
    assert cfg.cameras["side"].index_or_path == URL


# ---------------------------------------------------------------------------
# Camera factory hook
# ---------------------------------------------------------------------------


def test_factory_hook_builds_network_camera_alongside_usb() -> None:
    from lelab.network_camera import NetworkCamera, NetworkCameraConfig, install_camera_factory_hook
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs

    install_camera_factory_hook()
    cameras = make_cameras_from_configs(
        {
            "side": NetworkCameraConfig(index_or_path=URL, width=640, height=480, fps=15),
            "up": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        }
    )

    assert isinstance(cameras["side"], NetworkCamera)
    assert type(cameras["up"]).__name__ == "OpenCVCamera"


def test_network_camera_str_never_leaks_the_password() -> None:
    from lelab.network_camera import NetworkCamera, NetworkCameraConfig

    cam = NetworkCamera(NetworkCameraConfig(index_or_path=URL, width=640, height=480, fps=15))

    assert "s3cret" not in str(cam)
    assert "s3cret" not in str(cam.status())


# ---------------------------------------------------------------------------
# Stale-frame fallback and reconnect
# ---------------------------------------------------------------------------


def _camera() -> "object":
    from lelab.network_camera import NetworkCamera, NetworkCameraConfig

    return NetworkCamera(NetworkCameraConfig(index_or_path=URL, width=640, height=480, fps=15))


def test_read_serves_the_held_frame_and_reopens(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    from lelab.network_camera import NetworkCamera

    opened = []
    monkeypatch.setattr(
        OpenCVCamera, "async_read", lambda self, timeout_ms=200: (_ for _ in ()).throw(TimeoutError("silent"))
    )
    monkeypatch.setattr(OpenCVCamera, "disconnect", lambda self: None)
    monkeypatch.setattr(NetworkCamera, "connect", lambda self, warmup=True: opened.append(self))

    cam = _camera()
    held = np.zeros((4, 4, 3), dtype=np.uint8)
    cam._held = held
    cam._held_at = time.perf_counter()

    assert cam.async_read(timeout_ms=1) is held
    assert cam.stats["last_error"] == "silent"
    # The reopen runs on its own thread so the 30 fps loop never waits for it.
    deadline = time.monotonic() + 5.0
    while not opened and time.monotonic() < deadline:
        time.sleep(0.05)
    assert opened == [cam]
    assert cam.stats["reconnects"] == 1


def test_stale_frame_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen frame is a stopgap, not a substitute for a live camera."""
    import numpy as np
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    from lelab.network_camera import MAX_STALE_MS, NetworkCamera

    monkeypatch.setattr(
        OpenCVCamera, "async_read", lambda self, timeout_ms=200: (_ for _ in ()).throw(TimeoutError("silent"))
    )
    monkeypatch.setattr(NetworkCamera, "connect", lambda self, warmup=True: None)

    cam = _camera()
    cam._held = np.zeros((4, 4, 3), dtype=np.uint8)
    cam._held_at = time.perf_counter() - (MAX_STALE_MS / 1000 + 1)

    with pytest.raises(TimeoutError):
        cam.async_read(timeout_ms=1)


def test_first_frame_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing buffered there is no fallback to offer — say so loudly."""
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    from lelab.network_camera import NetworkCamera

    monkeypatch.setattr(
        OpenCVCamera, "async_read", lambda self, timeout_ms=200: (_ for _ in ()).throw(TimeoutError("silent"))
    )
    monkeypatch.setattr(NetworkCamera, "_schedule_reconnect", lambda self: None)

    cam = _camera()
    with pytest.raises(TimeoutError):
        cam.async_read(timeout_ms=1)


def test_disconnect_stops_reconnecting(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    from lelab.network_camera import NetworkCamera

    calls = []
    monkeypatch.setattr(
        OpenCVCamera, "async_read", lambda self, timeout_ms=200: (_ for _ in ()).throw(TimeoutError("silent"))
    )
    monkeypatch.setattr(OpenCVCamera, "disconnect", lambda self: None)
    monkeypatch.setattr(NetworkCamera, "connect", lambda self, warmup=True: calls.append(1))

    cam = _camera()
    cam._held = object()
    cam._held_at = time.perf_counter()
    cam._closed = True  # session teardown

    with pytest.raises(TimeoutError):
        cam.async_read(timeout_ms=1)
    time.sleep(0.2)
    assert calls == []


def test_harden_camera_reads_wraps_only_device_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network cameras carry their own guard; wrapping them again would shadow the
    reconnect with a ladder that can't reopen a dropped stream."""
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    from lelab.network_camera import NetworkCamera, NetworkCameraConfig
    from lelab.record import _harden_camera_reads

    net_cam = NetworkCamera(NetworkCameraConfig(index_or_path=URL, width=640, height=480, fps=15))
    usb_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30))

    class FakeRobot:
        cameras = {"side": net_cam, "up": usb_cam}

    _harden_camera_reads(FakeRobot())

    # The network camera keeps its own bound method; the device camera is wrapped
    # in record.py's escalating-timeout closure.
    assert getattr(net_cam.async_read, "__self__", None) is net_cam
    assert not hasattr(usb_cam.async_read, "__self__")


# ---------------------------------------------------------------------------
# Reachability guard (probe)
# ---------------------------------------------------------------------------


def test_probe_fails_fast_on_a_refused_host() -> None:
    """Without the socket pre-check, FFmpeg blocks the connect for minutes."""
    import time

    from lelab.network_camera import NetworkCameraError, probe

    started = time.monotonic()
    with pytest.raises(NetworkCameraError) as excinfo:
        probe("rtsp://user:pw@127.0.0.1:1/live")
    elapsed = time.monotonic() - started

    assert "is not reachable" in str(excinfo.value)
    assert "pw" not in str(excinfo.value)
    assert elapsed < 5


def test_probe_skips_the_socket_check_for_udp(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.network_camera as nc

    monkeypatch.setattr(nc, "_read_stream_properties", lambda url, read_timeout_s: {"width": 1, "height": 1, "fps": None})

    assert nc.probe("udp://@239.0.0.1:5000") == {"width": 1, "height": 1, "fps": None}


# ---------------------------------------------------------------------------
# Preview endpoints
# ---------------------------------------------------------------------------


def test_register_preview_source_rejects_a_non_stream(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.network_camera as nc

    monkeypatch.setattr(nc, "probe", lambda url: {"width": 640, "height": 480, "fps": 15})

    response = client.post("/network-camera-source", json={"url": "/dev/video0"})

    assert response.status_code == 400


def test_register_preview_source_returns_probe_and_key(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.network_camera as nc

    class DeadStream:
        """Stands in for cv2's capture so the test never dials a camera."""

        def read(self):
            return False, None

        def release(self):
            return None

    monkeypatch.setattr(nc, "probe", lambda url: {"width": 1920, "height": 1080, "fps": 25})
    monkeypatch.setattr(nc, "preview_stream", lambda url: DeadStream())
    # A feed with no frames must close on its own, or the request thread is held
    # for as long as the browser keeps the <img> open.
    monkeypatch.setattr(nc, "MAX_PREVIEW_MISSES", 3)

    response = client.post("/network-camera-source", json={"url": URL})
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"], body["fps"]) == (1920, 1080, 25)
    assert "s3cret" not in body["url"]

    with client.stream("GET", f"/network-camera-feed?key={body['key']}") as feed:
        assert feed.status_code == 200
        # Reading to EOF also proves the feed ends by itself instead of holding
        # the request open for the life of the <img>.
        assert feed.read() == b""


def test_preview_feed_requires_a_registered_key(client) -> None:
    response = client.get("/network-camera-feed?key=deadbeef")

    assert response.status_code == 404
