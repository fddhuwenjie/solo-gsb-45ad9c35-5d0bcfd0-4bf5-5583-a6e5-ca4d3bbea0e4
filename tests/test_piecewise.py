"""分段线性工具测试：多分量转移的跳变保持与连续合并。"""

import pytest

from app.planner.geometry import Interval
from app.planner.piecewise import PL, const_pl, evolve, lower_envelope, Piece


def cover_value(covers, y):
    """在覆盖列表上按“逐点取 min”语义求值。"""
    vals = [g.value(y) for d, g in covers if d.lo - 1e-9 <= y <= d.hi + 1e-9]
    return min(vals) if vals else float("inf")


def test_evolve_preserves_jump_between_disconnected_components():
    # 取自回归实例 t=3 的两个前驱分量：lift -2 与 lift -1。
    # 在 y = -209.75 处 lift -1 分量恰好进入可达半径 R=97.5，
    # 价值函数从 138 向下跳到 97.5；跳变不得被桥接。
    f_far = PL([-472.25, -467.0, -331.875, -272.25], [0.0, 1.0, -1.0], 0.0)
    f_near = PL([-112.25, -107.0, 28.125, 87.75], [0.0, 1.0, -1.0], 0.0)
    prev = [
        (Interval(-472.25, -272.25), f_far),
        (Interval(-112.25, 87.75), f_near),
    ]
    covers = evolve(prev, Interval(-287.0, -87.0), 97.5)

    # 跳变点两侧必须分属不同覆盖（不允许拼成单个连续 PL）
    assert len(covers) >= 2
    # 跳变点左侧仍走高代价分量：y=-230 处为 117.75，且趋近跳变点时趋于 138
    assert cover_value(covers, -230.0) == pytest.approx(117.75, abs=1e-7)
    assert cover_value(covers, -209.75 - 1e-6) == pytest.approx(138.0, abs=1e-5)
    # 跳变点处立即取到新分量的低值 97.5，随后降到 0（可达 lift -1 的零代价区）
    assert cover_value(covers, -209.75) == pytest.approx(97.5, abs=1e-7)
    assert cover_value(covers, -112.25) == pytest.approx(0.0, abs=1e-7)
    assert cover_value(covers, -107.0) == pytest.approx(0.0, abs=1e-7)
    assert cover_value(covers, -87.0) == pytest.approx(20.0, abs=1e-7)
    # 左端只由 lift -2 分量决定
    assert cover_value(covers, -287.0) == pytest.approx(90.25, abs=1e-7)
    assert cover_value(covers, -272.25) == pytest.approx(75.5, abs=1e-7)


def test_evolve_merges_continuous_segments_into_single_cover():
    # 单个前驱分量：转移结果整体连续，应合并为一个覆盖
    f = const_pl(Interval(-50.0, 50.0), 0.0)
    covers = evolve([(Interval(-50.0, 50.0), f)], Interval(-100.0, 100.0), 30.0)
    assert len(covers) == 1
    dom, g = covers[0]
    assert (dom.lo, dom.hi) == (-80.0, 80.0)
    # 锥形：中心 0 代价，向外斜率 ±1
    assert g.value(0.0) == pytest.approx(0.0, abs=1e-9)
    assert g.value(80.0) == pytest.approx(30.0, abs=1e-7)
    assert g.value(-80.0) == pytest.approx(30.0, abs=1e-7)


def test_evolve_leaves_unreachable_gap_uncovered():
    # 两个前驱分量的可达域在目标窗口内留有缝隙：缝隙处不得有覆盖，
    # 也不得跨缝隙外推
    f_a = const_pl(Interval(-100.0, -90.0), 1.0)
    f_b = const_pl(Interval(90.0, 100.0), 2.0)
    prev = [
        (Interval(-100.0, -90.0), f_a),
        (Interval(90.0, 100.0), f_b),
    ]
    covers = evolve(prev, Interval(-200.0, 200.0), 10.0)
    assert len(covers) == 2
    (d0, g0), (d1, g1) = covers
    assert (d0.lo, d0.hi) == (-110.0, -80.0)
    assert (d1.lo, d1.hi) == (80.0, 110.0)
    assert cover_value(covers, 0.0) == float("inf")  # 缝隙不可达
    assert g0.value(-100.0) == pytest.approx(1.0, abs=1e-9)
    assert g1.value(100.0) == pytest.approx(2.0, abs=1e-9)


def test_evolve_keeps_isolated_point_cover():
    # 前驱分量仅在目标窗口的孤立点可达（可达半径与窗口边缘相切）：
    # 该点覆盖必须保留，否则反向传播会错误地报告不可行
    f = const_pl(Interval(0.0, 10.0), 3.0)
    covers = evolve([(Interval(0.0, 10.0), f)], Interval(20.0, 30.0), 10.0)
    # 只有 y=20 处可达（x=10, |20-10|=R=10）
    assert len(covers) == 1
    dom, g = covers[0]
    assert dom.lo == pytest.approx(20.0, abs=1e-9)
    assert dom.hi == pytest.approx(20.0, abs=1e-9)
    assert cover_value(covers, 20.0) == pytest.approx(13.0, abs=1e-7)
    assert cover_value(covers, 20.5) == float("inf")


def test_lower_envelope_continuous_case_exact():
    # 连通域上的连续包络：min(两条交叉仿射片) 为 V 形
    pieces = [Piece(0.0, 10.0, -1.0, 10.0), Piece(0.0, 10.0, 1.0, 0.0)]
    env = lower_envelope(pieces)
    assert env is not None
    assert env.value(0.0) == pytest.approx(0.0, abs=1e-9)
    assert env.value(5.0) == pytest.approx(5.0, abs=1e-7)
    assert env.value(10.0) == pytest.approx(0.0, abs=1e-9)
