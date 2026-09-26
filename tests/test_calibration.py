"""同步标定算法（最短弧展开、时延枚举、离群剔除、偏差归一化）测试。"""

import math

import pytest

from app.planner.calibration import (
    AmbiguousJumpError,
    calibrate_sync,
    normalize_bias,
    shortest_arc_step,
    unwrap_shortest_arc,
)


def reference(t: float) -> float:
    """平滑非线性参考方位（度）：正弦摆动叠加匀速，使 lag 可辨识。"""
    return 120.0 * math.sin(0.8 * t) + 30.0 * t


RT = [i * 0.2 for i in range(26)]                    # 0.0 .. 5.0
ET = [round(0.37 + 0.31 * i, 10) for i in range(16)]  # 与参考异步的编码器时刻
LAG = 0.5
BIAS = 30.0


def encoder_signal(t: float, lag: float = LAG, bias: float = BIAS) -> float:
    # bias = reference - encoder（残差定义），故 encoder = R(t-lag) - bias
    return reference(t - lag) - bias


def make_encoder(**kw) -> list[float]:
    return [encoder_signal(t, **kw) for t in ET]


def test_shortest_arc_step_basic():
    assert shortest_arc_step(350.0) == pytest.approx(-10.0)
    assert shortest_arc_step(-350.0) == pytest.approx(10.0)
    assert shortest_arc_step(170.0) == pytest.approx(170.0)
    assert shortest_arc_step(-170.0) == pytest.approx(-170.0)
    assert shortest_arc_step(360.0) == pytest.approx(0.0)
    assert shortest_arc_step(730.0) == pytest.approx(10.0)


def test_shortest_arc_step_rejects_exactly_180():
    with pytest.raises(AmbiguousJumpError):
        shortest_arc_step(180.0)
    with pytest.raises(AmbiguousJumpError):
        shortest_arc_step(-180.0)
    with pytest.raises(AmbiguousJumpError):
        shortest_arc_step(540.0)


def test_unwrap_wraps_across_zero():
    # 350 -> 10 -> 30：逐步 +20，展开为 [350, 370, 390]
    out = unwrap_shortest_arc([350.0, 10.0, 30.0])
    assert out == pytest.approx([350.0, 370.0, 390.0])
    # 反向穿越：10 -> 350 -> 330
    out = unwrap_shortest_arc([10.0, 350.0, 330.0])
    assert out == pytest.approx([10.0, -10.0, -30.0])


def test_unwrap_rejects_180_jump_with_context():
    with pytest.raises(AmbiguousJumpError, match="reference_angles"):
        unwrap_shortest_arc([0.0, 180.0, 10.0], "reference_angles")


def test_normalize_bias():
    assert normalize_bias(0.0) == 0.0
    assert normalize_bias(30.0) == pytest.approx(30.0)
    assert normalize_bias(185.0) == pytest.approx(-175.0)
    assert normalize_bias(-181.0) == pytest.approx(179.0)
    assert normalize_bias(360.0) == pytest.approx(0.0)
    # 上界 180 归入 -180
    assert normalize_bias(180.0) == pytest.approx(-180.0)


def test_calibration_async_sampling_recovers_lag_and_bias():
    # 参考/编码器时间戳完全异步，编码器在编码器时刻按 R(t-lag)-bias 取值
    ra = [reference(t) for t in RT]
    ea = [v % 360.0 for v in make_encoder()]  # 编码器只给包装角
    out = calibrate_sync(RT, ra, ET, ea, -1.0, 1.0, 0.1, 3.0, 12)
    assert out.calibrated
    assert out.lag == pytest.approx(0.5, abs=1e-9)
    # 展开以首样本为锚点，偏差可能相差整圈；归一化后应为 30°
    assert normalize_bias(out.bias) == pytest.approx(30.0, abs=0.2)
    assert out.rmse == pytest.approx(0.14, abs=0.05)
    assert len(out.inliers) >= 12
    assert out.outliers == []


def test_calibration_negative_lag():
    # encoder(t) = R(t+0.4) - (-12)：编码器超前参考 0.4s，bias=-12
    ra = [reference(t) for t in RT]
    ea = [(reference(t + 0.4) + 12.0) % 360.0 for t in ET]
    out = calibrate_sync(RT, ra, ET, ea, -1.0, 1.0, 0.1, 3.0, 12)
    assert out.calibrated
    assert out.lag == pytest.approx(-0.4, abs=1e-9)
    assert normalize_bias(out.bias) == pytest.approx(-12.0, abs=0.2)


def test_calibration_bias_wraps_to_half_open_interval():
    ra = [reference(t) for t in RT]
    # bias = 185° -> 归一化为 -175°
    ea = [(reference(t - LAG) - 185.0) % 360.0 for t in ET]
    out = calibrate_sync(RT, ra, ET, ea, -1.0, 1.0, 0.1, 3.0, 12)
    assert out.calibrated
    assert out.lag == pytest.approx(0.5, abs=1e-9)
    assert normalize_bias(out.bias) == pytest.approx(-175.0, abs=0.2)


def test_calibration_outliers_are_detected_and_excluded():
    ra = [reference(t) for t in RT]
    ea = [v % 360.0 for v in make_encoder()]
    ea[4] = (ea[4] + 130.0) % 360.0
    ea[12] = (ea[12] - 100.0) % 360.0
    out = calibrate_sync(RT, ra, ET, ea, -1.0, 1.0, 0.1, 5.0, 10)
    assert out.calibrated
    assert out.lag == pytest.approx(0.5, abs=1e-9)
    assert out.outliers == [4, 12]
    assert 4 not in out.inliers and 12 not in out.inliers
    assert len(out.inliers) == 13
    assert out.rmse == pytest.approx(0.14, abs=0.05)


def test_calibration_tie_breaks_by_abs_lag_then_smaller_lag():
    # 常量序列：任意 lag 的残差均为常数，RMSE 恒为 0，触发决胜规则
    rt = [0.0, 1.0, 2.0, 3.0]
    ra = [10.0, 10.0, 10.0, 10.0]
    et = [0.5, 1.5, 2.5]
    ea = [5.0, 5.0, 5.0]
    # 候选 -1, 0, 1：RMSE 全 0，先取 |lag| 最小 -> 0
    out = calibrate_sync(rt, ra, et, ea, -1.0, 1.0, 1.0, 1.0, 1)
    assert out.calibrated and out.lag == 0.0
    # 候选 -1, 1, 3：|lag| 相同的 -1/1 并列，取较小者 -1
    out = calibrate_sync(rt, ra, et, ea, -1.0, 3.0, 2.0, 1.0, 1)
    assert out.calibrated and out.lag == -1.0


def test_calibration_no_overlap_gives_reason():
    out = calibrate_sync(
        [0.0, 1.0], [10.0, 20.0],
        [10.0, 11.0], [10.0, 20.0],
        0.0, 1.0, 0.5, 5.0, 1,
    )
    assert not out.calibrated
    assert "无重叠样本" in out.reason


def test_calibration_insufficient_inliers_gives_reason():
    out = calibrate_sync(
        [0.0, 1.0, 2.0, 3.0], [10.0, 10.0, 10.0, 10.0],
        [0.5, 1.5, 2.5], [5.0, 5.0, 5.0],
        0.0, 0.0, 1.0, 0.0001, 5,
    )
    assert not out.calibrated
    assert "内点不足" in out.reason
    assert out.details["max_inliers"] == 3
    assert out.details["min_inliers"] == 5


def test_calibration_indices_are_original_zero_based():
    ra = [reference(t) for t in RT]
    ea = [v % 360.0 for v in make_encoder()]
    # 首个编码器时刻平移后落在参考区间外 -> 索引 0 非重叠，不出现在任一列表
    out = calibrate_sync(RT, ra, ET, ea, -1.0, 1.0, 0.1, 3.0, 10)
    assert out.calibrated
    assert out.inliers == sorted(out.inliers)
    assert set(out.inliers).isdisjoint(out.outliers)
    assert all(0 <= i < len(ET) for i in out.inliers + out.outliers)
