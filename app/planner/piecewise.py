"""分段线性（PL）函数与“距离变换”转移算子。

本模块实现圆顶跟踪问题的核心数学工具。代价函数表示为定义在区间
``[lo, hi]`` 上的连续分段线性函数（断点 ``knots`` 与每段斜率 ``slopes``）。

关键性质：从初始的常数代价出发，对**单个连通前驱分量**反复施加转移

    g(y) = min_x { |y - x| + f(x) }，  约束 |y - x| <= R，
                                        x∈dom(f)， y∈target 窗口，

所得函数始终连续，且每段斜率属于 {-1, 0, 1}。

注意：当前一时刻存在**多个互不连通的 lift 分量**时，完整的价值函数是各
分量转移结果逐点取 min 的函数，一般**不连续**——在某个分量恰好进入可达
半径 R 的位置，可完成的代价会向下跳变。因此 :func:`evolve` 不把不同分量
的结果拼接成单个连续 PL，而是返回若干 ``(区间, PL)`` 覆盖，整体语义为
“逐点取 min”（见 :func:`evolve` 文档）。

本实现不做网格离散，也不依赖 LP 求解器：对 f 的每一个线性段
（斜率 s∈{-1,0,1}），上述最小值可由结构固定的 4 条仿射候选片精确给出
（段内截断片、左/右可达边界片、两个端点锥），对所有候选片取**下包络**
即得到 g。候选集对 g 的每个线性段都是充分的，多余的片不影响下包络结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import EPS, Interval

SLOPE_TOL = 1e-7
COORD_TOL = 1e-8


@dataclass
class Piece:
    """仿射片：在 [lo, hi] 上取值 a*x + b。"""

    lo: float
    hi: float
    a: float  # 斜率
    b: float  # 截距

    def value(self, x: float) -> float:
        return self.a * x + self.b


@dataclass
class PL:
    """连续分段线性函数。

    knots 为严格递增边界点，slopes[j] 为 [knots[j], knots[j+1]] 上的斜率；
    在 knots[0] 处取值 v0。单点域（knots 长度 1）以 slopes=[] 表示。
    """

    knots: list[float]
    slopes: list[float]
    v0: float

    @property
    def domain(self) -> Interval:
        return Interval(self.knots[0], self.knots[-1])

    def values(self) -> list[float]:
        """返回所有断点处的函数值。"""
        vals = [self.v0]
        v = self.v0
        for j, s in enumerate(self.slopes):
            v += s * (self.knots[j + 1] - self.knots[j])
            vals.append(v)
        return vals

    def value(self, x: float) -> float:
        v = self.v0
        for j, s in enumerate(self.slopes):
            l, r = self.knots[j], self.knots[j + 1]
            if x <= r + EPS:
                return v + s * (x - l)
            v += s * (r - l)
        return v

    def to_pieces(self) -> list[Piece]:
        vals = self.values()
        out = []
        for j, s in enumerate(self.slopes):
            l = self.knots[j]
            out.append(Piece(l, self.knots[j + 1], s, vals[j] - s * l))
        return out


def const_pl(iv: Interval, value: float) -> PL:
    return PL([iv.lo, iv.hi], [0.0], value)


def _canon_slope(s: float) -> float:
    if s > SLOPE_TOL:
        return 1.0
    if s < -SLOPE_TOL:
        return -1.0
    return 0.0


def lower_envelope(pieces: list[Piece]) -> PL | None:
    """把若干仿射片在其域之并上取最小值，重建为连续 PL。

    前置条件：片的并集是一个连通区间，且真包络在该区间上**连续**（例如
    所有片共同覆盖同一区间）。满足时包络每段斜率仍属于 {-1, 0, 1}，
    此时由首值 + 斜率即可精确重建各断点取值。

    断点取自所有片端点与两片的有效交点；在这些点上最小值必为线性，
    故采样即可精确重建，随后压缩共线断点。

    注意：若各片只覆盖互不相交的子域（不连通 lift 分量），真包络在分量
    边界处会发生跳变，本函数无权跨跳变点桥接——那种情形必须由
    :func:`evolve` 先按分量域端点切分后再分别调用本函数。
    """
    active = [p for p in pieces if p.hi >= p.lo - EPS]
    if not active:
        return None

    lo = min(p.lo for p in active)
    hi = max(p.hi for p in active)

    breaks: set[float] = {lo, hi}
    for p in active:
        breaks.add(min(max(p.lo, lo), hi))
        breaks.add(min(max(p.hi, lo), hi))
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            p, q = active[i], active[j]
            if abs(p.a - q.a) > SLOPE_TOL:
                x = (q.b - p.b) / (p.a - q.a)
                il = max(p.lo, q.lo)
                ir = min(p.hi, q.hi)
                if il - 1e-7 <= x <= ir + 1e-7:
                    breaks.add(min(max(x, lo), hi))

    xs = sorted(b for b in breaks if lo - 1e-7 <= b <= hi + 1e-7)

    def eval_min(x: float) -> float:
        best = float("inf")
        for p in active:
            if p.lo - COORD_TOL <= x <= p.hi + COORD_TOL:
                v = p.value(x)
                if v < best:
                    best = v
        return best

    pts = [xs[0]]
    vals = [eval_min(xs[0])]
    for x in xs[1:]:
        if abs(x - pts[-1]) <= EPS:
            continue
        y = eval_min(x)
        pts.append(x)
        vals.append(y)

    # 压缩共线断点
    knots = [pts[0]]
    kvals = [vals[0]]
    for i in range(1, len(pts)):
        x, y = pts[i], vals[i]
        if len(knots) >= 2:
            x0, x1 = knots[-2], knots[-1]
            y0, y1 = kvals[-2], kvals[-1]
            s01 = (y1 - y0) / (x1 - x0)
            s12 = (y - y1) / (x - x1)
            if abs(_canon_slope(s01) - _canon_slope(s12)) < SLOPE_TOL and \
                    abs(s01 - s12) < 1e-6:
                knots[-1] = x
                kvals[-1] = y
                continue
        knots.append(x)
        kvals.append(y)

    slopes = [
        _canon_slope((kvals[i + 1] - kvals[i]) / (knots[i + 1] - knots[i]))
        for i in range(len(knots) - 1)
    ]
    return PL(knots, slopes, kvals[0])


def _segment_pieces(
    xl: float,
    xr: float,
    s: float,
    fl: float,
    R: float,
    target: Interval,
) -> list[Piece]:
    """单个线性段 f(x)=fl+s*(x-xl)，x∈[xl,xr]，贡献的候选仿射片。

    g(y) = min_{x∈[xl,xr], |y-x|<=R} |y-x| + fl + s*(x-xl)

    四类候选（裁剪到 target 窗口）：
      1. 段内解 x=clamp(y, xl, xr)；
      2. 左可达边界 x=y+R（当 y+R∈[xl,xr]）；
      3. 右可达边界 x=y-R（当 y-R∈[xl,xr]）；
      4. 端点锥 x=xl / x=xr（在其可达半径内）。
    """
    out: list[Piece] = []
    ylo, yhi = target.lo, target.hi

    def add(a: float, b: float, lo: float, hi: float) -> None:
        lo = max(lo, ylo)
        hi = min(hi, yhi)
        if hi >= lo - EPS:
            out.append(Piece(lo, hi, a, b))

    # 1) 段内解
    #   y∈[xl,xr]: g = f(y) = s*y + (fl - s*xl)
    add(s, fl - s * xl, xl, xr)
    #   y∈[xr, xr+R]: x=xr, g = y - xr + fr
    fr = fl + s * (xr - xl)
    add(1.0, fr - xr, xr, xr + R)
    #   y∈[xl-R, xl]: x=xl, g = xl - y + fl
    add(-1.0, fl + xl, xl - R, xl)

    # 2) 左边界 x=y+R ∈ [xl,xr] => y∈[xl-R, xr-R]
    #    g = |y-(y+R)| + f(y+R) = R + fl + s*(y+R-xl)
    add(s, R + fl + s * (R - xl), xl - R, xr - R)

    # 3) 右边界 x=y-R ∈ [xl,xr] => y∈[xl+R, xr+R]
    #    g = R + fl + s*(y-R-xl)
    add(s, R + fl + s * (-R - xl), xl + R, xr + R)

    # 4) 端点锥（与上面在 s=±1 时部分重合，平坦段时不可缺）
    add(1.0, fl - xl, xl, xl + R)
    add(-1.0, fl + xl, xl - R, xl)
    add(1.0, fr - xr, xr, xr + R)
    add(-1.0, fr + xr, xr - R, xr)

    return out


def transition(f: PL, target: Interval, R: float) -> PL | None:
    """从单个前驱 PL 构造转移函数 g（在 target∩可达域 上）。

    返回 None 表示 target 中没有任何点能在 R 内到达 f 的域。
    """
    d_lo, d_hi = f.knots[0], f.knots[-1]
    reach_y = Interval(d_lo - R, d_hi + R)
    clipped = target.intersect(reach_y)
    if clipped.empty or clipped.hi < clipped.lo - EPS:
        return None

    pieces: list[Piece] = []
    if len(f.knots) == 1:
        x0 = f.knots[0]
        v = f.v0
        pieces.append(Piece(max(x0, clipped.lo), min(x0 + R, clipped.hi),
                            1.0, v - x0))
        pieces.append(Piece(max(x0 - R, clipped.lo), min(x0, clipped.hi),
                            -1.0, v + x0))
    else:
        vals = f.values()
        for j, s in enumerate(f.slopes):
            pieces.extend(
                _segment_pieces(
                    f.knots[j],
                    f.knots[j + 1],
                    _canon_slope(s),
                    vals[j],
                    R,
                    clipped,
                )
            )
    g = lower_envelope(pieces)
    return g


def add_abs(
    f: PL,
    center: float,
    domain: Interval,
    R: float,
) -> PL | None:
    """返回在 domain 上的 |y - center| + f(y)，其中 f 按其断点线性外推，
    结果裁剪到 |y-center| <= R 与 f 定义域及 domain 的交集。

    f PL，|y-center| 斜率 ±1，故和的斜率属于 {-2,-1,0,1,2}，
    断点为两者断点之并（含 center）。
    """
    lo = max(domain.lo, f.knots[0], center - R)
    hi = min(domain.hi, f.knots[-1], center + R)
    if hi < lo - EPS:
        return None

    breaks = {lo, hi}
    if lo < center < hi:
        breaks.add(center)
    for k in f.knots:
        if lo - 1e-8 <= k <= hi + 1e-8:
            breaks.add(min(max(k, lo), hi))
    xs = sorted(breaks)

    def fval(x: float) -> float:
        xx = min(max(x, f.knots[0]), f.knots[-1])
        return f.value(xx)

    vals = [fval(x) + abs(x - center) for x in xs]

    # 压缩共线断点（按实际斜率判定）
    knots = [xs[0]]
    kvals = [vals[0]]
    for i in range(1, len(xs)):
        x, y = xs[i], vals[i]
        if abs(x - knots[-1]) <= EPS:
            continue
        if len(knots) >= 2:
            x0, x1 = knots[-2], knots[-1]
            y0, y1 = kvals[-2], kvals[-1]
            s01 = (y1 - y0) / (x1 - x0)
            s12 = (y - y1) / (x - x1)
            if abs(s01 - s12) < 1e-6:
                knots[-1] = x
                kvals[-1] = y
                continue
        knots.append(x)
        kvals.append(y)

    slopes = [
        (kvals[i + 1] - kvals[i]) / (knots[i + 1] - knots[i])
        for i in range(len(knots) - 1)
    ]
    return PL(knots, slopes, kvals[0])


def _restrict(f: PL, lo: float, hi: float) -> PL:
    """把连续 PL 限制到子区间 [lo, hi]（f 的定义域必须覆盖 [lo, hi]）。"""
    f_lo, f_hi = f.knots[0], f.knots[-1]
    lo = min(max(lo, f_lo), f_hi)
    hi = min(max(hi, f_lo), f_hi)
    fvals = f.values()

    def val_at(x: float) -> float:
        for j, s in enumerate(f.slopes):
            l, r = f.knots[j], f.knots[j + 1]
            if x <= r + EPS:
                return fvals[j] + s * (x - l)
        return fvals[-1]

    knots = [lo]
    vals = [val_at(lo)]
    for j in range(len(f.slopes)):
        k = f.knots[j + 1]
        if lo + EPS < k < hi - EPS:
            knots.append(k)
            vals.append(fvals[j + 1])
    knots.append(hi)
    vals.append(val_at(hi))
    slopes = [
        _canon_slope((vals[i + 1] - vals[i]) / (knots[i + 1] - knots[i]))
        for i in range(len(knots) - 1)
    ]
    return PL(knots, slopes, vals[0])


def evolve(
    prev: list[tuple[Interval, PL]],
    target: Interval,
    max_delta: float,
) -> list[tuple[Interval, PL]]:
    """从前一时刻多个可行分量（区间+代价 PL）转移到目标窗口。

    每个前驱分量分别求转移函数 g_i（定义在 target 的某子区间上）。不同
    分量的可达子域之间可能互不相交；即使在子域相交处，逐点 min 的包络
    也可能在某个分量恰好进入可达半径的位置**向下跳变**。因此完整的价值
    函数一般不是连续 PL，这里返回若干 ``(区间, PL)`` 覆盖，整体语义为
    逐点取 min：

      - 以所有可达子域端点为切点，把窗口切成基本区间；每个基本区间上
        活跃分量集固定，包络连续，可安全用 :func:`lower_envelope` 合并；
      - 相邻基本区间仅在拼接处取值连续时才拼成单个 PL，跳变处保持分裂；
      - 缝隙（任何分量都不可达的区间）不产生覆盖，不做线性外推；
      - 零宽（点）覆盖单独保留——它可能是窗口内唯一可达的位置。

    下游（反向传播与最左 argmin 重建）对所有覆盖逐点取 min，故该表示
    在跳变点处也是精确的。
    """
    cover: list[tuple[Interval, PL]] = []
    for dom, f in prev:
        g = transition(f, target, max_delta)
        if g is not None:
            cover.append((g.domain, g))
    if not cover:
        return []

    points = [(d, g) for d, g in cover if d.hi - d.lo <= EPS]
    spans = [(d, g) for d, g in cover if d.hi - d.lo > EPS]

    cuts = sorted({d.lo for d, _ in spans} | {d.hi for d, _ in spans})
    segs: list[tuple[Interval, PL]] = []
    for lo, hi in zip(cuts, cuts[1:]):
        if hi - lo <= EPS:
            continue
        active = [
            g for d, g in spans
            if d.lo <= lo + COORD_TOL and d.hi >= hi - COORD_TOL
        ]
        if not active:
            continue  # 缝隙：该区间内任何前驱分量都不可达
        if len(active) == 1:
            segs.append((Interval(lo, hi), _restrict(active[0], lo, hi)))
            continue
        pieces: list[Piece] = []
        for g in active:
            for pc in g.to_pieces():
                c_lo = max(pc.lo, lo)
                c_hi = min(pc.hi, hi)
                if c_hi >= c_lo - EPS:
                    pieces.append(Piece(c_lo, c_hi, pc.a, pc.b))
        env = lower_envelope(pieces)
        if env is not None:
            segs.append((Interval(lo, hi), env))

    segs.extend(points)
    segs.sort(key=lambda c: (c[0].lo, c[0].hi))

    # 相邻且拼接处取值连续的段合并为单个 PL；跳变处保持分裂
    out: list[tuple[Interval, PL]] = []
    for dom, g in segs:
        if out:
            pdom, pg = out[-1]
            if abs(pdom.hi - dom.lo) <= COORD_TOL and \
                    abs(pg.value(pdom.hi) - g.value(dom.lo)) <= 1e-7:
                out[-1] = (
                    Interval(pdom.lo, dom.hi),
                    PL(pg.knots + g.knots[1:], pg.slopes + g.slopes, pg.v0),
                )
                continue
        out.append((dom, g))
    return out
