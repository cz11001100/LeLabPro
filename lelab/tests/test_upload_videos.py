"""Serial, retried video uploads — `upload_folder`'s thread pool is what broke."""

from __future__ import annotations


def test_videos_upload_one_at_a_time_and_report_each_file(tmp_path) -> None:
    from lelab.record import _push_videos_one_by_one

    files = []
    for name, size in (("file-001.mp4", 300), ("file-002.mp4", 100), ("file-003.mp4", 200)):
        p = tmp_path / "videos" / "observation.images.up" / "chunk-000" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
        files.append(p)

    seen: list[str] = []
    reported: list[tuple[int, str]] = []

    class Api:
        def upload_file(self, *, path_in_repo: str, **kwargs) -> None:
            seen.append(path_in_repo)

    _push_videos_one_by_one(Api(), "user/ds", tmp_path, files, lambda done, name: reported.append((done, name)))

    # Smallest first, so the progress bar starts moving instead of stalling on a 70MB file.
    assert seen == [
        "videos/observation.images.up/chunk-000/file-002.mp4",
        "videos/observation.images.up/chunk-000/file-003.mp4",
        "videos/observation.images.up/chunk-000/file-001.mp4",
    ]
    assert [done for done, _ in reported] == [1, 2, 3]


def test_a_flaky_video_upload_is_retried_without_losing_the_rest(tmp_path, monkeypatch, caplog) -> None:
    """The batch upload died on one file and abandoned the other 29."""
    import logging

    import lelab.record as record

    monkeypatch.setattr(record.time, "sleep", lambda _s: None)

    paths = []
    for name in ("file-001.mp4", "file-002.mp4"):
        p = tmp_path / "videos" / "chunk-000" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"y" * 10)
        paths.append(p)

    attempts: dict[str, int] = {}

    class Flaky:
        def upload_file(self, *, path_in_repo: str, **kwargs) -> None:
            attempts[path_in_repo] = attempts.get(path_in_repo, 0) + 1
            if path_in_repo.endswith("file-001.mp4") and attempts[path_in_repo] < 3:
                raise OSError("error sending request for url (https://s3-accelerate.amazonaws.com/...)")

    with caplog.at_level(logging.WARNING, logger="lelab.record"):
        record._push_videos_one_by_one(Flaky(), "user/ds", tmp_path, paths, lambda done, name: None)

    assert attempts["videos/chunk-000/file-001.mp4"] == record._VIDEO_UPLOAD_ATTEMPTS
    assert attempts["videos/chunk-000/file-002.mp4"] == 1
    assert "retrying" in caplog.text


def test_a_video_that_never_uploads_fails_loudly(tmp_path, monkeypatch) -> None:
    """Half a dataset on the Hub is not a successful upload."""
    import pytest

    import lelab.record as record

    monkeypatch.setattr(record.time, "sleep", lambda _s: None)

    p = tmp_path / "videos" / "chunk-000" / "file-000.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"z" * 10)

    class Dead:
        def upload_file(self, **kwargs) -> None:
            raise OSError("connection reset")

    with pytest.raises(OSError):
        record._push_videos_one_by_one(Dead(), "user/ds", tmp_path, [p], lambda done, name: None)
