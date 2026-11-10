"""EMA smoothing must be rate-independent, or the fast loop filters nothing."""

from __future__ import annotations

import math


def test_alpha_matches_the_documented_weight_at_30fps() -> None:
    from lerobot.robots.so_follower.so_follower import ema_step

    # At the nominal control rate the knob means exactly what the UI says.
    assert ema_step(prev=10.0, raw=20.0, alpha=0.5, dt=1 / 30) == 15.0
    assert ema_step(prev=10.0, raw=20.0, alpha=1.0, dt=1 / 30) == 20.0
    assert ema_step(prev=10.0, raw=20.0, alpha=0.0, dt=1 / 30) == 10.0


def test_one_thirty_fps_frame_equals_thirty_nine_hundred_hz_frames() -> None:
    from lerobot.robots.so_follower.so_follower import ema_step

    alpha = 0.5
    # 30 ticks of 1/900 s is exactly one 30 fps frame of smoothing.
    fast = 0.0
    for _ in range(30):
        fast = ema_step(fast, 100.0, alpha, 1 / 900)
    slow = ema_step(0.0, 100.0, alpha, 1 / 30)

    assert math.isclose(fast, slow, rel_tol=1e-9)


def test_a_kihertz_loop_actually_smooths_over_visible_time() -> None:
    from lerobot.robots.so_follower.so_follower import ema_step

    # A 10 Hz wobble through a loop running at 1 kHz: after 20 ms the output must
    # still be near the mean, not chasing the instantaneous sample.
    alpha = 0.5
    value = 0.0
    for tick in range(400):  # 0.4 s
        raw = 10.0 if (tick // 50) % 2 == 0 else -10.0  # 10 Hz square wave
        value = ema_step(value, raw, alpha, 1 / 1000)
    assert abs(value) < 7.0, f"per-tick alpha would have followed the wave, got {value}"


def test_send_action_filters_the_commanded_position(monkeypatch) -> None:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SO101Follower(SO101FollowerConfig(port="/dev/null", ema_alpha=0.5))
    sent: list[dict[str, float]] = []

    class Bus:
        def sync_write(self, name, values):
            sent.append(dict(values))

    robot.bus = Bus()
    monkeypatch.setattr(SOFollower, "is_connected", property(lambda self: True))

    robot.send_action({"shoulder_pan.pos": 0.0})
    robot.send_action({"shoulder_pan.pos": 10.0})

    # First tick seeds the filter, second one is pulled toward the new target but
    # must not jump to it: back-to-back ticks are microseconds apart.
    assert sent[0]["shoulder_pan"] == 0.0
    assert 0.0 < sent[1]["shoulder_pan"] < 10.0

