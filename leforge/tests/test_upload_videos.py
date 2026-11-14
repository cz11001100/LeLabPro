"""Serial, retried video uploads — `upload_folder`'s thread pool is what broke."""

from __future__ import annotations

import pytest


@pytest.fixture
def restore_http_backend():
    from huggingface_hub.utils import set_client_factory
    from huggingface_hub.utils._http import default_client_factory

    yield
    set_client_factory(default_client_factory)


def test_videos_upload_one_at_a_time_and_report_each_file(tmp_path) -> None:
    from leforge.record import _push_videos_one_by_one

    files = []
    for name, size in (("file-001.mp4", 300), ("file-002.mp4", 100), ("file-003.mp4", 200)):
        p = tmp_path / "videos" / "observation.images.up" / "chunk-000" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
        files.append(p)

    seen: list[str] = []
    reported: list[tuple[int, int, str, int]] = []

    class Api:
        def upload_file(self, *, path_in_repo: str, **kwargs) -> None:
            seen.append(path_in_repo)

    _push_videos_one_by_one(
        Api(), "user/ds", tmp_path, files, lambda c, t, n, a: reported.append((c, t, n, a))
    )

    # Smallest first, so the progress bar starts moving instead of stalling on a 70MB file.
    assert seen == [
        "videos/observation.images.up/chunk-000/file-002.mp4",
        "videos/observation.images.up/chunk-000/file-003.mp4",
        "videos/observation.images.up/chunk-000/file-001.mp4",
    ]
    # attempt>0 rows name the file in flight; attempt==0 rows are the completed count.
    assert [(c, a) for c, t, n, a in reported] == [
        (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0),
    ]
    assert {t for _, t, _, _ in reported} == {3}


def test_a_flaky_video_upload_is_retried_without_losing_the_rest(tmp_path, monkeypatch, caplog) -> None:
    """The batch upload died on one file and abandoned the other 29."""
    import logging

    import leforge.record as record

    monkeypatch.setattr(record.time, "sleep", lambda _s: None)

    paths = []
    for name in ("file-001.mp4", "file-002.mp4"):
        p = tmp_path / "videos" / "chunk-000" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"y" * 10)
        paths.append(p)

    attempts: dict[str, int] = {}
    named: list[tuple[str, int]] = []

    class Flaky:
        def upload_file(self, *, path_in_repo: str, **kwargs) -> None:
            attempts[path_in_repo] = attempts.get(path_in_repo, 0) + 1
            if path_in_repo.endswith("file-001.mp4") and attempts[path_in_repo] < 3:
                raise OSError("error sending request for url (https://s3-accelerate.amazonaws.com/...)")

    def note(completed, total, name, attempt):
        if attempt:
            named.append((name, attempt))

    with caplog.at_level(logging.WARNING, logger="leforge.record"):
        record._push_videos_one_by_one(Flaky(), "user/ds", tmp_path, paths, note)

    assert attempts["videos/chunk-000/file-001.mp4"] == record._VIDEO_UPLOAD_ATTEMPTS
    assert attempts["videos/chunk-000/file-002.mp4"] == 1
    assert ("videos/chunk-000/file-001.mp4", 2) in named
    assert ("videos/chunk-000/file-001.mp4", 3) in named
    assert "retrying" in caplog.text


def test_a_video_that_never_uploads_fails_loudly(tmp_path, monkeypatch) -> None:
    """Half a dataset on the Hub is not a successful upload."""
    import leforge.record as record

    monkeypatch.setattr(record.time, "sleep", lambda _s: None)

    p = tmp_path / "videos" / "chunk-000" / "file-000.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"z" * 10)

    class Dead:
        def upload_file(self, **kwargs) -> None:
            raise OSError("connection reset")

    with pytest.raises(OSError):
        record._push_videos_one_by_one(Dead(), "user/ds", tmp_path, [p], lambda *a: None)


def test_a_stalled_connection_raises_instead_of_parking_forever(restore_http_backend) -> None:
    """huggingface_hub's client is built with `timeout=None`, so a dead proxy tunnel hangs.

    Measured on the live server: 19 minutes frozen on a 57MB video with 0 bytes sent
    and no exception for the retry loop to catch.
    """
    import httpx2

    from huggingface_hub.utils import _http
    from leforge.record import _UPLOAD_CONNECT_TIMEOUT_S, _UPLOAD_READ_TIMEOUT_S, _pin_upload_timeouts

    _pin_upload_timeouts()

    client = _http.get_session()
    assert isinstance(client, httpx2.Client)
    # Timeouts live on the client now, not on each request like the requests backend.
    assert client.timeout.connect == _UPLOAD_CONNECT_TIMEOUT_S
    assert client.timeout.read == _UPLOAD_READ_TIMEOUT_S

    def stall(request):
        raise httpx2.ReadTimeout("the proxy stopped forwarding")

    client._transport = httpx2.MockTransport(stall)
    with pytest.raises(httpx2.ReadTimeout):
        client.put("https://hf-hub-lfs.s3-accelerate.amazonaws.com/part", content=b"payload")
