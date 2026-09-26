"""几何工具：方位角归一化、狭缝窗口（目标提升）、线缆区间求交与区间合并。

所有角度均以“度”为单位，圆顶状态使用**未展开角度**（unwrapped angle，
可取任意实数，不限于 0..360）。狭缝以当前圆顶角为中心，半宽为
``slit_width / 2``；目标方位经整数圈平移后（不同 lift）落入狭缝窗口即可。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EPS = 1e-9
FULL_TURN = 360.0


@dataclass(frozen=True)
class Interval:
    """闭区间 [lo, hi]。"""

    lo: float
    hi: float

    @property
    def empty(self) -> bool:
        return self.lo > self.hi + EPS

    def intersect(self, other: "Interval") -> "Interval":
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))


def normalize_azimuth(az: float) -> float:
    """把方位角归一化到 [0, 360)。"""
    a = az % FULL_TURN
    if a < 0:
        a += FULL_TURN
    if a >= FULL_TURN - EPS:
        a = 0.0
    return a


def slit_windows(
    target_az: float,
    slit_width: float,
    cable: Interval,
    initial_angle: float | None = None,
) -> list[tuple[Interval, int]]:
    """返回某目标方位对应的狭缝可行区间（与线缆区间求交后）。

    每个返回项为 ``(interval, lift)``，其中 ``lift`` 为整数圈数，
    该区间内的圆顶角 ``d`` 满足
    ``d - slit_width/2 <= target_az + 360*lift <= d + slit_width/2``。
    ``lift`` 直接以输入方位为基准（不做归一化），因此上式对任意实数
    方位都精确成立，响应中的 ``lift`` 与 ``dome_angle`` 自洽。

    当 slit_width >= 360 时，任意圆顶角都能看到目标，直接返回线缆区间。
    其它情况下，线缆区间（圆顶工程上约为 ±270° 量级）最多与几个相邻
    lift 的窗口相交，故生成少量候选即可。
    """
    half = slit_width / 2.0

    if slit_width >= FULL_TURN - EPS:
        return [] if cable.empty else [(cable, 0)]

    # 不归一化：窗口中心直接取 target_az + 360*lift，保证 lift 语义
    # 与文档恒等式一致（归一化只会平移 lift 标号，窗口集合不变）。
    base = target_az

    # 估计需要考察的 lift 范围：窗口 [center-half, center+half] 与
    # [cable.lo, cable.hi] 相交，即 center 在 [cable.lo-half, cable.hi+half]。
    k_lo = int(math.floor((cable.lo - half - base) / FULL_TURN)) - 1
    k_hi = int(math.ceil((cable.hi + half - base) / FULL_TURN)) + 1

    out: list[tuple[Interval, int]] = []
    for lift in range(k_lo, k_hi + 1):
        center = base + FULL_TURN * lift
        window = Interval(center - half, center + half)
        clipped = window.intersect(cable)
        if not clipped.empty and clipped.hi >= clipped.lo - EPS:
            out.append((Interval(clipped.lo, clipped.hi), lift))
    return out


def merge_intervals(
    intervals: list[Interval], tol: float = 1e-7
) -> list[Interval]:
    """合并相交或几乎相接的区间，按 lo 排序；输入允许为空。"""
    items = sorted(
        (Interval(float(i.lo), float(i.hi)) for i in intervals if not i.empty),
        key=lambda i: i.lo,
    )
    merged: list[Interval] = []
    for iv in items:
        if merged and iv.lo <= merged[-1].hi + tol:
            if iv.hi > merged[-1].hi:
                merged[-1] = Interval(merged[-1].lo, iv.hi)
        else:
            merged.append(iv)
    return merged
