"""几何工具测试。"""

import math

from app.planner.geometry import (
    Interval,
    merge_intervals,
    normalize_azimuth,
    slit_windows,
)


def test_normalize_azimuth():
    assert normalize_azimuth(0) == 0
    assert normalize_azimuth(360) == 0
    assert normalize_azimuth(-10) == 350
    assert abs(normalize_azimuth(725) - 5) < 1e-9


def test_slit_windows_single_lift():
    cable = Interval(-180, 180)
    wins = slit_windows(10.0, 4.0, cable)
    # 窗口 [8,12] 完全在线缆内
    assert len(wins) == 1
    iv, lift = wins[0]
    assert lift == 0
    assert abs(iv.lo - 8) < 1e-9
    assert abs(iv.hi - 12) < 1e-9


def test_slit_windows_clipped_by_cable():
    cable = Interval(0, 90)
    wins = slit_windows(0.0, 20.0, cable)
    assert len(wins) == 1
    iv, lift = wins[0]
    assert abs(iv.lo - 0) < 1e-9
    assert abs(iv.hi - 10) < 1e-9


def test_slit_windows_multiple_lifts():
    # 宽线缆 + 目标 0°，窄狭缝：lift -1、0、1 的窗口都可在线缆内
    cable = Interval(-400, 400)
    wins = slit_windows(0.0, 10.0, cable)
    lifts = sorted(lift for _, lift in wins)
    assert lifts == [-1, 0, 1]


def test_slit_windows_full_width():
    cable = Interval(-50, 50)
    wins = slit_windows(123.0, 360.0, cable)
    assert len(wins) == 1
    assert (wins[0][0].lo, wins[0][0].hi) == (-50, 50)


def test_slit_windows_no_intersection():
    cable = Interval(0, 10)
    assert slit_windows(180.0, 20.0, cable) == []


def test_merge_intervals():
    out = merge_intervals([Interval(0, 5), Interval(4, 9), Interval(20, 25)])
    assert len(out) == 2
    assert (out[0].lo, out[0].hi) == (0, 9)
    assert (out[1].lo, out[1].hi) == (20, 25)
