def _fake_v4l2(monkeypatch, supported=True, fail_ctrls=()):
    from lelab import v4l2

    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0
            stdout = b"auto_exposure gain white_balance_automatic" if supported else b""

        if cmd[-1].startswith("--set-ctrl="):
            ctrl = cmd[-1][len("--set-ctrl="):]
            if ctrl in fail_ctrls:
                Result.returncode = 1
        return Result()

    monkeypatch.setattr(v4l2.subprocess, "run", run)
    return calls


def _set_ctrls(calls):
    return [c[-1][len("--set-ctrl="):] for c in calls if c[-1].startswith("--set-ctrl=")]


def test_manual_exposure_and_gain_are_applied(monkeypatch) -> None:
    from lelab.v4l2 import apply_camera_controls, configure_manual_controls

    calls = _fake_v4l2(monkeypatch)
    configure_manual_controls({"wrist": {"camera_index": 6, "exposure": 10, "gain": 32}})
    apply_camera_controls("/dev/video6", "wrist")
    ctrls = _set_ctrls(calls)
    assert "auto_exposure=1" in ctrls
    assert "exposure_time_absolute=10" in ctrls
    assert "gain=32" in ctrls
    assert "auto_exposure=3" not in ctrls


def test_without_manual_values_driver_auto_exposure_is_used(monkeypatch) -> None:
    from lelab.v4l2 import apply_camera_controls, configure_manual_controls

    calls = _fake_v4l2(monkeypatch)
    configure_manual_controls({"wrist": {"camera_index": 6}})
    apply_camera_controls("/dev/video6", "wrist")
    ctrls = _set_ctrls(calls)
    assert "auto_exposure=3" in ctrls
    assert "exposure_time_absolute=10" not in ctrls


def test_invalid_manual_values_are_ignored(monkeypatch) -> None:
    from lelab.v4l2 import apply_camera_controls, configure_manual_controls

    calls = _fake_v4l2(monkeypatch)
    configure_manual_controls(
        {"wrist": {"camera_index": 6, "exposure": -1, "gain": "high"}}
    )
    apply_camera_controls("/dev/video6", "wrist")
    ctrls = _set_ctrls(calls)
    assert "auto_exposure=3" in ctrls
    assert not any(c.startswith("gain=") for c in ctrls)


def test_configure_manual_controls_none_clears_previous(monkeypatch) -> None:
    from lelab.v4l2 import apply_camera_controls, configure_manual_controls

    calls = _fake_v4l2(monkeypatch)
    configure_manual_controls({"wrist": {"camera_index": 6, "exposure": 10}})
    configure_manual_controls(None)
    apply_camera_controls("/dev/video6", "wrist")
    assert "auto_exposure=3" in _set_ctrls(calls)


def test_camera_without_auto_exposure_support_gets_no_exposure_ctrls(monkeypatch) -> None:
    from lelab.v4l2 import apply_camera_controls, configure_manual_controls

    calls = _fake_v4l2(monkeypatch, supported=False)
    configure_manual_controls({"wrist": {"camera_index": 6, "exposure": 10}})
    apply_camera_controls("/dev/video6", "wrist")
    ctrls = _set_ctrls(calls)
    assert "white_balance_automatic=1" in ctrls
    assert not any(c.startswith("auto_exposure=") for c in ctrls)
    assert not any(c.startswith("exposure_time_absolute=") for c in ctrls)


def test_failed_manual_pin_does_not_fall_back_to_auto(monkeypatch) -> None:
    from lelab.v4l2 import apply_camera_controls, configure_manual_controls

    calls = _fake_v4l2(monkeypatch, fail_ctrls=("auto_exposure=1",))
    configure_manual_controls({"wrist": {"camera_index": 6, "exposure": 10}})
    apply_camera_controls("/dev/video6", "wrist")
    ctrls = _set_ctrls(calls)
    assert "exposure_time_absolute=10" in ctrls
    assert "auto_exposure=3" not in ctrls
