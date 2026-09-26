"""圆顶狭缝跟踪：精确分段线性动态规划求解器。

问题（角度单位：度；时间单位：秒）
----------------------------------
给定目标方位序列 a_1..a_n（对应时刻 t_1..t_n）、圆顶初始未展开角度 d_0、
最大转速 v_max（度/秒）、狭缝角宽 w、线缆角度区间 [c_lo, c_hi]，为每个
时刻选择未展开角度 d_t，使得：

  1. 狭缝可见：|d_t - (a_t + 360·k)| <= w/2（某个整数圈 k）；
  2. 转速限制：|d_t - d_{t-1}| <= v_max · (t_t - t_{t-1})；
  3. 线缆边界：c_lo <= d_t <= c_hi；
  4. 最小化累计转角 Σ|d_t - d_{t-1}|；
  5. 代价相同的序列中取**字典序最小**者（d_1 最小，并列时 d_2 最小……）。

算法
----
每时刻可行域 F_t 是若干闭区间（各 lift 窗口与线缆区间求交）。代价函数
（“从此刻到末尾的最小剩余转角”）用 :mod:`piecewise` 中的 PL 函数表示，
自末时刻反向递推；再从 d_0 起正向重建，每一步在可行区间内取使
“立即动作代价 + 剩余代价”最小的**最左点**，由此得到唯一的字典序最小
最优序列。前向单独做一次纯区间可达性传播，用于无解时定位最早失效时刻
及未满足的约束。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import EPS, Interval, merge_intervals, slit_windows
from .piecewise import PL, add_abs, const_pl, evolve

OPT_TOL = 1e-7


@dataclass
class Problem:
    target_azimuths: list[float]
    times: list[float]
    initial_angle: float
    max_speed: float          # 度/秒
    slit_width: float         # 度
    cable_lo: float
    cable_hi: float


@dataclass
class Action:
    index: int
    time: float
    dome_angle: float        # 未展开角度
    target_azimuth: float
    lift: int                # 目标所在的整数圈
    move_delta: float        # 相对上一时刻的转角（度）
    move_speed: float        # 本步转速（度/秒）
    cable_low: float
    cable_high: float


@dataclass
class Failure:
    index: int               # 最早失效时刻（1 基）
    time: float
    target_azimuth: float
    unmet: list[str]
    details: dict = field(default_factory=dict)


@dataclass
class Solution:
    feasible: bool
    actions: list[Action] = field(default_factory=list)
    total_rotation: float = 0.0
    failure: Failure | None = None


def _lifts_for(problem: Problem, target_az: float) -> list[tuple[Interval, int]]:
    cable = Interval(problem.cable_lo, problem.cable_hi)
    return slit_windows(target_az, problem.slit_width, cable)


def _reach_bounds(intervals: list[Interval], R: float) -> list[Interval]:
    return [Interval(iv.lo - R, iv.hi + R) for iv in intervals]


def _forward(problem: Problem) -> tuple[list[list[Interval]], list[list[tuple[Interval, int]]] | None]:
    """前向纯区间可达性传播。

    返回 (forward_sets, target_components)：
      forward_sets[t-1] 为时刻 t 的可达可行域（区间并，已合并）；
      一旦某时刻为空即截断返回（此时 target_components 为该时刻的窗口）。
    """
    initial_iv = Interval(problem.initial_angle, problem.initial_angle)

    forward: list[list[Interval]] = []
    prev = [initial_iv]
    last_time = 0.0
    for t, (time_s, az) in enumerate(
        zip(problem.times, problem.target_azimuths), start=1
    ):
        dt = time_s - last_time
        R = problem.max_speed * dt
        windows = _lifts_for(problem, az)
        reach = _reach_bounds(prev, R)
        current: list[Interval] = []
        for win, _lift in windows:
            for rc in reach:
                inter = win.intersect(rc)
                if not inter.empty and inter.hi >= inter.lo - EPS:
                    current.append(inter)
        current = merge_intervals(current)
        if not current:
            return forward + [[]], windows
        forward.append(current)
        prev = current
        last_time = time_s
    return forward, None


def _diagnose(
    problem: Problem,
    forward: list[list[Interval]],
    windows: list[tuple[Interval, int]],
) -> Failure:
    """前向在时刻 t 失败时，区分线缆 / 转速两类未满足约束。"""
    t = len(forward)  # 失败时刻 1-based（forward 末项为空）
    time_s = problem.times[t - 1]
    az = problem.target_azimuths[t - 1]
    unmet: list[str] = []
    details: dict = {}

    if not windows:
        unmet.extend(["slit_visibility", "cable_limit"])
        details["reason"] = (
            "目标方位在任意整数圈平移后，狭缝窗口都与线缆角度区间无交集"
        )
        details["cable_interval"] = [problem.cable_lo, problem.cable_hi]
        details["slit_width"] = problem.slit_width
        return Failure(t, time_s, az, unmet, details)

    details["candidate_windows"] = [
        {
            "lift": lift,
            "low": round(iv.lo, 9),
            "high": round(iv.hi, 9),
        }
        for iv, lift in windows
    ]

    if t == 1:
        prev = [Interval(problem.initial_angle, problem.initial_angle)]
        last_time = 0.0
    else:
        prev = forward[t - 2]
        last_time = problem.times[t - 2]
    dt = time_s - last_time
    R = problem.max_speed * dt
    reach = merge_intervals(_reach_bounds(prev, R))

    best_gap = float("inf")
    nearest = None
    for win, lift in windows:
        for rc in reach:
            gap = max(rc.lo - win.hi, win.lo - rc.hi, 0.0)
            if gap < best_gap:
                best_gap = gap
                nearest = (win, lift, rc)

    # 线缆本身永远允许窗口存在（windows 非空），失败只能由可达性造成。
    unmet.append("speed_limit")
    win, lift, rc = nearest
    required_speed = (best_gap + R) / dt if dt > 0 else float("inf")
    details["reason"] = (
        "从可达角度集合到最近狭缝窗口的最小间隙超过本步最大转角"
    )
    details["time_step_seconds"] = dt
    details["max_rotation_this_step"] = R
    details["required_extra_rotation"] = round(best_gap, 9)
    details["required_min_speed_deg_per_s"] = (
        round(required_speed, 9) if math.isfinite(required_speed) else None
    )
    details["reachable_set"] = [
        [round(i.lo, 9), round(i.hi, 9)] for i in reach
    ]
    details["nearest_window"] = {
        "lift": lift,
        "low": round(win.lo, 9),
        "high": round(win.hi, 9),
    }
    return Failure(t, time_s, az, unmet, details)


def _backward(
    problem: Problem,
    components: list[list[tuple[Interval, int]]],
) -> list[list[list[tuple[Interval, PL]]]]:
    """反向值函数：W_t(y) = 从 y 出发完成 t..n 的最小剩余转角。

    返回与 components 对齐的列表：每个窗口对应一个覆盖列表
    ``[(区间, PL), ...]``，整体语义为逐点取 min。同一窗口内可能有多个
    互不连通的可达分量；即便分量域相接，价值函数在分量恰好进入可达半径
    的位置也可能向下跳变，因此覆盖按跳变点切分（见 piecewise.evolve）。
    无法完成后续时刻的窗口对应空列表。
    """
    n = len(components)
    W: list[list[list[tuple[Interval, PL]]]] = [None] * n  # type: ignore[list-item]
    W[n - 1] = [[(iv, const_pl(iv, 0.0))] for iv, _ in components[n - 1]]
    for t in range(n - 2, -1, -1):
        R = problem.max_speed * (problem.times[t + 1] - problem.times[t])
        next_cover = [
            (dom, g)
            for covers in W[t + 1]
            for dom, g in covers
        ]
        row: list[list[tuple[Interval, PL]]] = []
        for iv, _lift in components[t]:
            row.append(evolve(next_cover, iv, R))
        W[t] = row
    return W


def _leftmost_argmin(
    start: float,
    windows: list[tuple[Interval, int]],
    W: list[list[tuple[Interval, PL]]],
    R: float,
) -> tuple[float, int, float]:
    """选择使 h(y)=|y-start| + W(y) 最小的最左 y（y 位于当前时刻窗口内）。

    ``W`` 与 windows 对齐，为该窗口的连通覆盖列表（已含本步 |y-x| 及后续
    全部代价）；代价相同取较小 y（字典序）。返回 (y, lift, 最优值)。
    """
    best_y: float | None = None
    best_lift = 0
    best_val = float("inf")

    def consider(x: float, v: float, lift: int) -> None:
        nonlocal best_y, best_lift, best_val
        if v < best_val - OPT_TOL or (
            abs(v - best_val) <= OPT_TOL
            and (best_y is None or x < best_y - EPS)
        ):
            best_val, best_y, best_lift = v, x, lift

    for (iv, lift), covers in zip(windows, W):
        for dom, w in covers:
            h = add_abs(w, start, dom, R)
            if h is None:
                continue
            vals = h.values()

            if len(h.knots) == 1:
                consider(h.knots[0], h.v0, lift)
                continue

            consider(h.knots[0], vals[0], lift)
            for j, s in enumerate(h.slopes):
                l, r = h.knots[j], h.knots[j + 1]
                vl, vr = vals[j], vals[j + 1]
                if s < -1e-7:
                    consider(r, vr, lift)   # 递减段：最小在右端
                elif s > 1e-7:
                    consider(l, vl, lift)   # 递增段：最小在左端
                else:
                    consider(l, vl, lift)   # 平坦段：整段同值，取左端

    return best_y, best_lift, best_val


def solve(problem: Problem) -> Solution:
    """求解完整问题。"""
    n = len(problem.target_azimuths)
    if n == 0:
        return Solution(feasible=True)

    cable = Interval(problem.cable_lo, problem.cable_hi)
    if problem.initial_angle < cable.lo - EPS or \
            problem.initial_angle > cable.hi + EPS:
        return Solution(
            feasible=False,
            failure=Failure(
                index=0,
                time=0.0,
                target_azimuth=float("nan"),
                unmet=["cable_limit"],
                details={
                    "reason": "初始未展开角度不在线缆角度区间内",
                    "initial_angle": problem.initial_angle,
                    "cable_interval": [problem.cable_lo, problem.cable_hi],
                },
            ),
        )

    forward, failed_windows = _forward(problem)
    if failed_windows is not None:
        failure = _diagnose(problem, forward, failed_windows)
        return Solution(feasible=False, failure=failure)

    # 每时刻的可行分量（lift 窗口与线缆求交；前向已证明整体可行）
    components = [_lifts_for(problem, az) for az in problem.target_azimuths]
    W = _backward(problem, components)

    actions: list[Action] = []
    d = problem.initial_angle
    total = 0.0
    last_time = 0.0
    for t in range(n):
        time_s = problem.times[t]
        az = problem.target_azimuths[t]
        dt = time_s - last_time
        R = problem.max_speed * dt
        w_next = W[t] if t < n else []
        y, lift, _val = _leftmost_argmin(d, components[t], w_next, R)
        if y is None:
            # 理论上不可达（前向已保证可行），防御性处理
            return Solution(
                feasible=False,
                failure=Failure(
                    index=t + 1,
                    time=time_s,
                    target_azimuth=az,
                    unmet=["speed_limit"],
                    details={"reason": "内部求解错误：路径重建失败"},
                ),
            )
        delta = y - d
        speed = delta / dt if dt > 0 else 0.0
        total += abs(delta)
        actions.append(
            Action(
                index=t + 1,
                time=time_s,
                dome_angle=y,
                target_azimuth=az,
                lift=lift,
                move_delta=delta,
                move_speed=speed,
                cable_low=problem.cable_lo,
                cable_high=problem.cable_hi,
            )
        )
        d = y
        last_time = time_s

    return Solution(
        feasible=True,
        actions=actions,
        total_rotation=total,
    )
