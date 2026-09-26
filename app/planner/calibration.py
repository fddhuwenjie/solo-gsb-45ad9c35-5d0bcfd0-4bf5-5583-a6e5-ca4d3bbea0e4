"""同步标定：由参考方位与圆顶编码器样本估计响应时延与零位偏差。

模型
----
设参考方位序列为 R(t)，编码器在时刻 t 的读数对应参考时刻 ``t - lag``
（**正时延**表示编码器滞后于参考），且编码器存在零位偏差 ``bias``：

    encoder(t) ≈ R(t - lag) - bias，即 bias ≈ R(t - lag) - encoder(t)。

两组样本各自的相邻角度按**最短弧**展开（unwrapped）：增量取
(-180, 180) 内的等价值；最短弧恰为 180° 的跳变方向有歧义，直接拒绝。

枚举每个候选时延后，把编码器时刻平移为 ``t - lag`` 并在**重叠样本**上
对参考序列线性插值；残差 = 插值参考角 − 展开后的编码器角。先以残差
中位数初估偏差，按阈值剔除离群点（残差偏离初估超过阈值者），再以内点
残差均值重估偏差并在内点集上计算 RMSE。内点数不足的候选无效。

选择规则：RMSE 最小；并列时 ``|lag|`` 较小；再并列时 ``lag`` 较小。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from statistics import median

from .geometry import EPS

RMSE_TIE_TOL = 1e-9
MAX_LAG_CANDIDATES = 100_000


class AmbiguousJumpError(ValueError):
    """相邻角度的最短弧恰为 180°，展开方向存在歧义。"""


def shortest_arc_step(delta: float) -> float:
    """把原始角度增量折算为最短弧增量（度，(-180, 180]）。

    最短弧恰为 ±180°（容差 EPS）时抛 :class:`AmbiguousJumpError`。
    """
    d = delta % 360.0
    if d > 180.0:
        d -= 360.0
    if abs(abs(d) - 180.0) <= EPS:
        raise AmbiguousJumpError("最短弧恰为 180° 的跳变存在方向歧义")
    return d


def unwrap_shortest_arc(angles: list[float], series_name: str = "angles") -> list[float]:
    """以首样本为锚点，按最短弧逐步展开角度序列。

    相邻样本最短弧恰为 180° 时抛出带序列名与相邻索引的
    :class:`AmbiguousJumpError`。
    """
    unwrapped = [angles[0]]
    for i in range(1, len(angles)):
        try:
            step = shortest_arc_step(angles[i] - angles[i - 1])
        except AmbiguousJumpError:
            raise AmbiguousJumpError(
                f"{series_name}[{i - 1}]={angles[i - 1]} 与 "
                f"{series_name}[{i}]={angles[i]} 之间的最短弧恰为 180°，"
                "无法确定展开方向，请检查或剔除该样本"
            ) from None
        unwrapped.append(unwrapped[-1] + step)
    return unwrapped


def normalize_bias(bias: float) -> float:
    """把零位偏差归一化到 [-180, 180)。"""
    return (bias + 180.0) % 360.0 - 180.0


def _interp_linear(
    t: float, ts: list[float], vs: list[float]
) -> float | None:
    """ts 严格递增时对 vs 线性插值；t 落在 [ts[0], ts[-1]] 之外返回 None。"""
    first, last = ts[0], ts[-1]
    if t < first:
        return vs[0] if first - t <= EPS else None
    if t > last:
        return vs[-1] if t - last <= EPS else None
    j = bisect_left(ts, t)
    if j == 0:
        return vs[0]
    if j >= len(ts):
        return vs[-1]
    t0, t1 = ts[j - 1], ts[j]
    f = (t - t0) / (t1 - t0)
    return vs[j - 1] + f * (vs[j] - vs[j - 1])


@dataclass
class LagCandidate:
    lag: float
    overlap: int                           # 重叠样本数
    inliers: list[int] = field(default_factory=list)
    outliers: list[int] = field(default_factory=list)
    bias: float = 0.0                      # 内点残差均值
    rmse: float = float("inf")


@dataclass
class CalibrationOutcome:
    calibrated: bool
    lag: float = 0.0
    bias: float = 0.0
    rmse: float = 0.0
    inliers: list[int] = field(default_factory=list)
    outliers: list[int] = field(default_factory=list)
    reason: str = ""
    details: dict = field(default_factory=dict)


def _evaluate_lag(
    lag: float,
    ref_times: list[float],
    ref_unwrapped: list[float],
    enc_times: list[float],
    enc_unwrapped: list[float],
    outlier_threshold: float,
) -> LagCandidate:
    """评估单个时延候选：插值 → 中位数初估偏差 → 剔除离群点 → 均值/RMSE。"""
    residuals: list[tuple[int, float]] = []
    for i, t in enumerate(enc_times):
        ref_value = _interp_linear(t - lag, ref_times, ref_unwrapped)
        if ref_value is None:
            continue  # 平移后落在参考时间区间之外：非重叠样本
        residuals.append((i, ref_value - enc_unwrapped[i]))

    cand = LagCandidate(lag=lag, overlap=len(residuals))
    if not residuals:
        return cand

    bias0 = median(r for _, r in residuals)
    inlier_res: list[float] = []
    for i, r in residuals:
        if abs(r - bias0) > outlier_threshold:
            cand.outliers.append(i)
        else:
            cand.inliers.append(i)
            inlier_res.append(r)

    if not cand.inliers:
        cand.bias = bias0
        return cand

    cand.bias = sum(inlier_res) / len(inlier_res)
    cand.rmse = math.sqrt(
        sum((r - cand.bias) ** 2 for r in inlier_res) / len(inlier_res)
    )
    return cand


def _candidate_better(cand: LagCandidate, best: LagCandidate) -> bool:
    """RMSE 最小；RMSE 并列时 |lag| 较小；再并列时 lag 较小。"""
    if cand.rmse < best.rmse - RMSE_TIE_TOL:
        return True
    if cand.rmse > best.rmse + RMSE_TIE_TOL:
        return False
    if abs(cand.lag) + EPS < abs(best.lag):
        return True
    if abs(cand.lag) > abs(best.lag) + EPS:
        return False
    return cand.lag < best.lag - EPS


def calibrate_sync(
    reference_times: list[float],
    reference_angles: list[float],
    encoder_times: list[float],
    encoder_angles: list[float],
    lag_min: float,
    lag_max: float,
    lag_step: float,
    outlier_threshold: float,
    min_inliers: int,
) -> CalibrationOutcome:
    """枚举时延候选并返回最优标定结果；无有效候选时给出原因与诊断。"""
    ref_u = unwrap_shortest_arc(reference_angles, "reference_angles")
    enc_u = unwrap_shortest_arc(encoder_angles, "encoder_angles")

    best: LagCandidate | None = None
    evaluated = 0
    max_overlap = 0
    max_inliers = 0

    k = 0
    while True:
        lag = lag_min + k * lag_step
        if lag > lag_max + EPS:
            break
        evaluated += 1
        cand = _evaluate_lag(
            lag, reference_times, ref_u, encoder_times, enc_u, outlier_threshold
        )
        max_overlap = max(max_overlap, cand.overlap)
        max_inliers = max(max_inliers, len(cand.inliers))
        if len(cand.inliers) >= min_inliers and (
            best is None or _candidate_better(cand, best)
        ):
            best = cand
        k += 1

    common_details = {
        "evaluated_candidates": evaluated,
        "max_overlap_samples": max_overlap,
        "max_inliers": max_inliers,
        "min_inliers": min_inliers,
        "outlier_threshold": outlier_threshold,
    }

    if best is None:
        if evaluated == 0:
            reason = "时延搜索区间为空：需满足 lag_min <= lag_max 且 lag_step > 0"
        elif max_overlap == 0:
            reason = (
                "所有时延候选均无重叠样本：编码器时刻平移后均落在参考时间区间"
                "之外，请放宽 lag_min/lag_max 或检查两组样本的时间范围"
            )
        else:
            reason = (
                f"内点不足：{evaluated} 个候选的内点数均小于 min_inliers"
                f"（最多重叠 {max_overlap} 个样本、最多内点 {max_inliers} 个），"
                "可适当增大 outlier_threshold 或减小 min_inliers"
            )
        return CalibrationOutcome(
            calibrated=False, reason=reason, details=common_details
        )

    return CalibrationOutcome(
        calibrated=True,
        lag=best.lag,
        bias=best.bias,
        rmse=best.rmse,
        inliers=best.inliers,
        outliers=best.outliers,
        details=common_details,
    )
