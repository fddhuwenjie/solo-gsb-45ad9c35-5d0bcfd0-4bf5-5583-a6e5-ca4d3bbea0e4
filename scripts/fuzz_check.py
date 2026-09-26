"""对拍脚本：网格对齐的随机小实例上比较精确 PL-DP 与暴力 DP。

所有参数对齐到 0.25° 网格：目标中心、狭缝半宽、线缆边界、初角、每步最大
转角均为网格倍数。L1 最短路经过区间约束时，其最优点只可能是区间端点、
目标中心或两约束点的对称中点，因此全部落在网格点上，可逐坐标比较
（最优代价 + 字典序）。

直接运行：`python3 scripts/fuzz_check.py`。
"""

from __future__ import annotations

import collections
import itertools
import random
import sys

sys.path.insert(0, "/workspace")

from app.planner.geometry import Interval, slit_windows  # noqa: E402
from app.planner.solver import Problem, solve  # noqa: E402

G = 0.0625


def q(x: float) -> float:
    return round(x / G) * G


def grid_points(problem: Problem, az: float) -> list[float]:
    cable = Interval(problem.cable_lo, problem.cable_hi)
    pts = set()
    for iv, _ in slit_windows(az, problem.slit_width, cable):
        k_lo = round(iv.lo / G)
        k_hi = round(iv.hi / G)
        for k in range(k_lo, k_hi + 1):
            pts.add(q(k * G))
    return sorted(pts)


def brute(problem: Problem):
    """网格 DP（单调队列，O(n·G)），返回 (最小代价, 字典序最小序列)。

    B[t][i] = min_{j 可达} |x_j - x_i| + B[t+1][j]，两侧分别用单调队列
    维护窗口内的 min(B-q)（q<=p 侧）与 min(B+q)（q>=p 侧）。
    """
    grids = [grid_points(problem, az) for az in problem.target_azimuths]
    n = len(grids)

    B = [None] * n
    B[n - 1] = [0.0] * len(grids[n - 1])
    for t in range(n - 2, -1, -1):
        R = problem.max_speed * (problem.times[t + 1] - problem.times[t])
        cur, nxt = grids[t], grids[t + 1]
        bn = B[t + 1]
        b = [float("inf")] * len(cur)

        # 左侧 q ∈ [p-R, p]：cost = p + (Bn[j] - q)
        k = 0
        dq = collections.deque()
        for i, p in enumerate(cur):
            while k < len(nxt) and nxt[k] <= p + 1e-9:
                v = bn[k] - nxt[k]
                while dq and dq[-1][0] >= v - 1e-12:
                    dq.pop()
                dq.append((v, k))
                k += 1
            while dq and nxt[dq[0][1]] < p - R - 1e-9:
                dq.popleft()
            if dq:
                b[i] = min(b[i], p + dq[0][0])

        # 右侧 q ∈ [p, p+R]：cost = -p + (Bn[j] + q)
        k = len(nxt) - 1
        dq.clear()
        for i in range(len(cur) - 1, -1, -1):
            p = cur[i]
            while k >= 0 and nxt[k] >= p - 1e-9:
                v = bn[k] + nxt[k]
                while dq and dq[-1][0] >= v - 1e-12:
                    dq.pop()
                dq.append((v, k))
                k -= 1
            while dq and nxt[dq[0][1]] > p + R + 1e-9:
                dq.popleft()
            if dq:
                b[i] = min(b[i], -p + dq[0][0])
        B[t] = b

    # 前向重建：每步取使 |y-start|+B[t] 最小的最左网格点
    seq = []
    start = problem.initial_angle
    last_t = 0.0
    total = 0.0
    for t in range(n):
        R = problem.max_speed * (problem.times[t] - last_t)
        best_j, best_v = None, float("inf")
        for j, y in enumerate(grids[t]):
            if abs(y - start) > R + 1e-9 or B[t][j] == float("inf"):
                continue
            v = abs(y - start) + B[t][j]
            if v < best_v - 1e-9 or (abs(v - best_v) <= 1e-9 and
                                     (best_j is None or j < best_j)):
                best_v, best_j = v, j
        if best_j is None:
            return float("inf"), None
        total += abs(grids[t][best_j] - start)
        seq.append(grids[t][best_j])
        start = grids[t][best_j]
        last_t = problem.times[t]
    return total, seq


def random_instance(rng: random.Random) -> Problem:
    n = rng.randint(1, 6)
    # 时间步长取 0.5 秒的整数倍；speed 取 0.5 的整数倍，
    # 使 R=speed*dt 落在 0.0625 网格
    dts = [0.5 * rng.randint(1, 8) for _ in range(n)]
    times = []
    acc = 0.0
    for d in dts:
        acc += d
        times.append(acc)
    azs = [q(rng.randint(0, 1440) * 0.25) for _ in range(n)]
    slit = q(rng.choice([1.0, 2.0, 5.0, 10.0, 30.0, 60.0]))
    speed = 0.5 * rng.randint(1, 120)
    clo = q(rng.choice([-400.0, -270.0, -180.0, -90.0, -20.0, 0.0]))
    chi = q(rng.choice([180.0, 90.0, 20.0, 270.0, 360.0, 400.0]))
    if clo > chi:
        clo, chi = chi, clo
    if chi - clo < 40:
        clo, chi = -90.0, 90.0
    d0 = q(rng.randint(int(clo / G) + 2, int(chi / G) - 2) * G)
    return Problem(azs, times, d0, speed, slit, clo, chi)


def main() -> int:
    rng = random.Random(20260919)
    trials = 1200
    fails = 0
    for trial in range(trials):
        p = random_instance(rng)
        sol = solve(p)
        bc, bseq = brute(p)
        brute_feasible = bseq is not None

        if not brute_feasible:
            if sol.feasible:
                print(f"[{trial}] 暴力无解但精确解有解")
                fails += 1
            continue
        if not sol.feasible:
            print(f"[{trial}] 暴力有解({bc:.3f})但精确解报告无解: "
                  f"{sol.failure.unmet if sol.failure else None}")
            fails += 1
            continue

        seq = [q(a.dome_angle) for a in sol.actions]
        raw = [a.dome_angle for a in sol.actions]
        if any(abs(x - gx) > 1e-5 for x, gx in zip(raw, seq)):
            print(f"[{trial}] 解偏离网格: {raw}")
            fails += 1
            continue
        if abs(sol.total_rotation - bc) > 1e-5:
            print(f"[{trial}] 代价不一致: 精确={sol.total_rotation:.5f} "
                  f"暴力={bc:.5f}; 精确序列={seq}")
            fails += 1
            continue
        if seq != bseq:
            print(f"[{trial}] 字典序不一致:\n  精确={seq}\n  暴力={bseq}")
            fails += 1

    print(f"完成 {trials} 组随机对拍，失败 {fails} 组")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
