"""安全联锁审计：按分段常值重放事件流，推导天气风险状态与关闭义务履行情况。

事件类型
--------
- ``wind`` / ``rain``：风速 / 雨量更新，两次更新之间信号**分段常值**保持；
- ``close_command``：关顶命令（人工或自动发出）；
- ``shutter``：快门状态 ``open → closing → closed → opening → open``，
  允许停留原态或沿循环前进一步，缺初值（全程无快门事件）或跳态均拒绝。

所有事件按 ``(time, 输入数组下标)`` 重放：时刻非递减，**同刻事件严格按输入
顺序处理**。

天气状态机
----------
- ``safe``：初始在收到首个天气更新之前为 ``unknown``；
- 触发：风速 ≥ ``wind_trip`` **或** 雨量 ≥ ``rain_trip``（任一已上报通道越限
  即算），且该条件连续保持 ``trip_duration`` → ``unsafe``；
- 复位：风速 < ``wind_reset`` **且** 雨量 < ``rain_reset``（两项都已上报且
  连续低于各自复位阈值）保持 ``reset_duration`` → ``safe``；
- 中断：距最近一次任意天气更新的间隔严格超过 ``stale_after`` → ``unknown``；
  中断后来新更新立即回到 ``safe`` 并重新计时。

进入 ``unsafe`` / ``unknown`` 即产生**关闭义务**：须在 ``command_deadline``
内发出关顶命令、在 ``close_deadline`` 内确认 ``closed``。义务时限在进入时刻
钉死，之后天气恢复或重复关顶命令都不会取消或延后它。``unsafe`` 与
``unknown`` 之间若无 ``safe`` 间隔，视为同一段连续风险（共享最早进入时刻的
义务）；从 ``safe`` 再次进入则产生新义务。

义务归属按 **(时刻, 原始事件索引)** 的全序判定：每个状态入口在该序中有确定
位置——计时器在同刻事件组**之前**成熟的触发/复位落在 ``(t, -1)``（同刻所有
输入事件都在入口之后，可履行本次义务）；``trip_duration=0`` 的当刻触发落在
触发天气事件自身的位置（其后的同刻事件可履行，其前的不可履行）；中断因要求
“严格超过”，在随后一组中回拨到 ``stale_bound`` 并位于该时刻全部输入事件
**之后** ``(stale_bound, +∞)``（恰在边界到达的命令不算履行）。只有严格位于
入口之后的关顶命令与 ``closed`` 能履行本次义务；严格在入口之前已生效的
``closed`` 才设置 ``already_closed_at_entry``。

末个事件之后，引擎会把已挂起的计时（触发/复位/中断）按发生时刻**投影**到
时间线中（``projected=true``），不再有新事件可履行这些未来义务。
"""

from __future__ import annotations

from dataclasses import dataclass, field

SAFE = "safe"
UNSAFE = "unsafe"
UNKNOWN = "unknown"

# 快门循环：只允许停留原态或沿此序列前进一步
SHUTTER_CYCLE = ("open", "closing", "closed", "opening")

WIND = "wind"
RAIN = "rain"
COMMAND = "close_command"
SHUTTER = "shutter"

# 状态入口在 (时刻, 序号) 全序中的序号哨兵：
# - BEFORE_GROUP：计时器在同刻事件组之前成熟（触发/复位在边界时刻即成熟）；
# - AFTER_GROUP：中断在随后一组中回拨到 stale_bound，严格超过才成立，故位于
#   该时刻全部输入事件之后（恰在边界到达的命令不算履行）。
BEFORE_GROUP = -1
AFTER_GROUP = float("inf")


class InterlockValidationError(ValueError):
    """事件流语义非法（快门缺初值 / 跳态等），应映射为 HTTP 422。"""

    def __init__(self, message: str, event_index: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.event_index = event_index


@dataclass
class Event:
    index: int
    time: float
    kind: str
    value: float | None = None
    state: str | None = None


@dataclass
class StatusSegment:
    """天气状态时间段：[start, end)，end 为 None 表示持续到末尾（开口）。"""

    start: float
    end: float | None
    state: str
    cause_event_indices: list[int] = field(default_factory=list)
    projected: bool = False


@dataclass
class ShutterSegment:
    """快门状态时间段。同刻多次转换会产生 start == end 的零长段（保留以反映
    同刻输入顺序），最后一段 end 为 None。"""

    start: float
    end: float | None
    state: str
    event_index: int


@dataclass
class CommandRecord:
    time: float
    event_index: int


@dataclass
class RiskInterval:
    """一段极大的连续非安全（unsafe/unknown 无缝衔接）区间及其关闭义务审计。"""

    start: float
    end: float | None
    states: list[str]
    entry_event_indices: list[int]
    projected: bool
    command_due: float
    close_due: float
    command_time: float | None
    command_event_index: int | None
    closed_time: float | None
    closed_event_index: int | None
    already_closed_at_entry: bool
    command_overdue: float
    close_overdue: float


@dataclass
class InterlockResult:
    status_timeline: list[StatusSegment]
    risk_intervals: list[RiskInterval]
    close_commands: list[CommandRecord]
    shutter_timeline: list[ShutterSegment]
    horizon: float  # 末个事件时刻：开口义务的超期量截算到此


def _trip_condition(
    wind: float | None, rain: float | None, wind_trip: float, rain_trip: float
) -> bool:
    return (wind is not None and wind >= wind_trip) or (
        rain is not None and rain >= rain_trip
    )


def _reset_condition(
    wind: float | None, rain: float | None, wind_reset: float, rain_reset: float
) -> bool:
    # 复位要求两项都已上报且严格低于各自复位阈值（迟滞）
    return (
        wind is not None
        and rain is not None
        and wind < wind_reset
        and rain < rain_reset
    )


def audit_interlock(
    events: list[Event],
    *,
    wind_trip: float,
    wind_reset: float,
    rain_trip: float,
    rain_reset: float,
    trip_duration: float,
    reset_duration: float,
    stale_after: float,
    command_deadline: float,
    close_deadline: float,
) -> InterlockResult:
    """对已通过参数校验的事件流执行分段常值重放。"""
    if not events:
        raise InterlockValidationError("events 不得为空")

    t0 = events[0].time
    t_last = events[-1].time

    # ---- 天气状态机运行时状态 ----
    weather = UNKNOWN
    wind: float | None = None
    rain: float | None = None
    wind_idx: int | None = None
    rain_idx: int | None = None
    run: tuple[str, float] | None = None  # ("trip"|"reset", run_start)
    run_fire: float | None = None  # 计时成熟时刻
    stale_bound: float | None = None  # 中断严格超过该时刻 → unknown

    segments: list[dict] = []
    cur: dict = {
        "start": t0,
        "end": None,
        "state": UNKNOWN,
        "cause": [],
        "projected": False,
        # 初始 unknown 段始于首个事件时刻、早于该时刻任何输入事件
        "entry_pos": (t0, BEFORE_GROUP),
    }

    commands: list[CommandRecord] = []
    shutter_points: list[tuple[float, int, str]] = []

    def held_indices() -> list[int]:
        return [k for k in (wind_idx, rain_idx) if k is not None]

    def change_state(
        new: str,
        t: float,
        cause: list[int],
        projected: bool,
        entry_pos: tuple[float, float],
    ) -> None:
        nonlocal cur
        if cur["state"] == new:
            return
        if cur["start"] == t:
            # 同刻翻转：前一状态零时长，直接覆盖（不产生区间/义务）
            cur["state"] = new
            cur["cause"] = list(cause)
            cur["projected"] = projected
            cur["entry_pos"] = entry_pos
            return
        cur["end"] = t
        segments.append(cur)
        cur = {
            "start": t,
            "end": None,
            "state": new,
            "cause": list(cause),
            "projected": projected,
            "entry_pos": entry_pos,
        }

    def fire_run(projected: bool, entry_order: float = BEFORE_GROUP) -> None:
        nonlocal weather, run, run_fire
        assert run is not None and run_fire is not None
        kind, _ = run
        # 计时器成熟时刻先于/在同刻事件组边界；组内当刻触发（duration==0）
        # 由调用方传入触发事件的原始索引作为入口序号
        entry_pos = (run_fire, entry_order)
        if kind == "trip":
            change_state(UNSAFE, run_fire, held_indices(), projected, entry_pos)
            weather = UNSAFE
            # 越限保持中，复位条件不可能同时成立，无需立即起复位计时
        else:
            change_state(SAFE, run_fire, held_indices(), projected, entry_pos)
            weather = SAFE
        run = None
        run_fire = None

    def fire_stale(projected: bool) -> None:
        nonlocal weather, run, run_fire, stale_bound
        assert stale_bound is not None
        # 严格超过 stale_after 才中断：回拨到 stale_bound 时位于该时刻全部
        # 输入事件之后（恰在边界到达的更新会取消它，命令则不能履行本义务）
        change_state(
            UNKNOWN,
            stale_bound,
            held_indices(),
            projected,
            (stale_bound, AFTER_GROUP),
        )
        weather = UNKNOWN
        run = None
        run_fire = None
        stale_bound = None

    def apply_wind_or_rain(ev: Event) -> None:
        nonlocal wind, rain, wind_idx, rain_idx, run, run_fire, stale_bound, weather
        if ev.kind == WIND:
            wind, wind_idx = ev.value, ev.index
        else:
            rain, rain_idx = ev.value, ev.index
        t = ev.time

        # 任何天气更新都刷新中断计时（取消旧的、重排新的）
        stale_bound = t + stale_after

        if weather == UNKNOWN:
            # 恢复观测：立即回到 safe，触发/复位计时重新开始。入口位置就是
            # 本天气事件（同刻更早的命令不能履行已结束的 unknown 义务）
            change_state(SAFE, t, [ev.index], False, (t, ev.index))
            weather = SAFE

        if weather == SAFE:
            if _trip_condition(wind, rain, wind_trip, rain_trip):
                if run is None or run[0] != "trip":
                    run = ("trip", t)
                    run_fire = t + trip_duration
                # duration == 0：本次更新当刻进入 unsafe（先于同刻后续事件，
                # 但晚于同刻更早的命令/快门事件）
                if run_fire <= t:
                    fire_run(False, entry_order=ev.index)
            else:
                run, run_fire = None, None
        elif weather == UNSAFE:
            if _reset_condition(wind, rain, wind_reset, rain_reset):
                if run is None or run[0] != "reset":
                    run = ("reset", t)
                    run_fire = t + reset_duration
                if run_fire <= t:
                    fire_run(False, entry_order=ev.index)
            else:
                run, run_fire = None, None

    def apply_shutter(ev: Event) -> None:
        assert ev.state is not None
        if shutter_points:
            prev = shutter_points[-1][2]
            wanted = SHUTTER_CYCLE[(SHUTTER_CYCLE.index(prev) + 1) % len(SHUTTER_CYCLE)]
            if ev.state != prev and ev.state != wanted:
                raise InterlockValidationError(
                    f"快门事件[{ev.index}] 非法跳变：{prev} → {ev.state}，"
                    f"仅允许停留原态或转换为 {wanted}",
                    event_index=ev.index,
                )
        shutter_points.append((ev.time, ev.index, ev.state))

    # ---- 按时刻分组重放 ----
    pos = 0
    n = len(events)
    while pos < n:
        group_t = events[pos].time
        group: list[Event] = []
        while pos < n and events[pos].time == group_t:
            group.append(events[pos])
            pos += 1

        # 先处理在本组之前/当刻成熟的计时事件：
        # - 触发/复位计时在边界时刻（含相等）即成熟，先于同刻输入事件；
        # - 中断要求“严格超过”，仅当 stale_bound < group_t 才触发；
        #   恰在边界到达的天气更新会取消它（下方事件处理中重排）。
        while True:
            due_run = run_fire if run_fire is not None and run_fire <= group_t else None
            due_stale = (
                stale_bound
                if stale_bound is not None and stale_bound < group_t
                else None
            )
            if due_run is None and due_stale is None:
                break
            # 同刻时先跑触发/复位再跑中断；零长段会被 change_state 合并掉
            if due_stale is None or (
                due_run is not None and due_run <= due_stale
            ):
                fire_run(False)
            else:
                fire_stale(False)

        # 同刻事件严格按输入顺序处理
        for ev in group:
            if ev.kind in (WIND, RAIN):
                apply_wind_or_rain(ev)
            elif ev.kind == COMMAND:
                commands.append(CommandRecord(ev.time, ev.index))
            elif ev.kind == SHUTTER:
                apply_shutter(ev)

    # 缺快门初值（全程无 shutter 事件）拒绝
    if not shutter_points:
        raise InterlockValidationError("事件流缺少快门初值：全程无 shutter 事件")

    # ---- 末个事件之后：投影挂起的成熟计时（假定此后无更新/命令到达）----
    while run is not None or stale_bound is not None:
        if stale_bound is None or (
            run_fire is not None and run_fire <= stale_bound
        ):
            if run is None:
                break
            fire_run(True)
        else:
            fire_stale(True)

    segments.append(cur)
    status_timeline = [
        StatusSegment(
            start=s["start"],
            end=s["end"],
            state=s["state"],
            cause_event_indices=s["cause"],
            projected=s["projected"],
        )
        for s in segments
    ]

    # ---- 快门时间线（保留同刻零长段以反映输入顺序）----
    shutter_timeline: list[ShutterSegment] = []
    shutter_cur: dict | None = None
    for t, idx, state in shutter_points:
        if shutter_cur is not None:
            shutter_cur["end"] = t
            shutter_timeline.append(ShutterSegment(**shutter_cur))
        shutter_cur = {"start": t, "end": None, "state": state, "event_index": idx}
    if shutter_cur is not None:
        shutter_timeline.append(ShutterSegment(**shutter_cur))

    # ---- 极大连续非安全区间（unsafe/unknown 无缝衔接合并）----
    risk_intervals: list[RiskInterval] = []
    i = 0
    while i < len(segments):
        if segments[i]["state"] == SAFE:
            i += 1
            continue
        j = i
        states: list[str] = []
        while j < len(segments) and segments[j]["state"] != SAFE:
            if not states or states[-1] != segments[j]["state"]:
                states.append(segments[j]["state"])
            j += 1
        first, last = segments[i], segments[j - 1]
        start, end = first["start"], last["end"]
        # 入口在 (时刻, 原始事件索引) 全序中的位置：只有严格位于其后的
        # 命令 / closed 能履行本次义务，严格在其前已 closed 才算进入即已关闭
        entry_pos: tuple[float, float] = first["entry_pos"]

        command_due = start + command_deadline
        close_due = start + close_deadline
        horizon = end if end is not None else t_last

        cmd = next(
            (c for c in commands if (c.time, c.event_index) > entry_pos),
            None,
        )
        command_overdue = (
            max(0.0, cmd.time - command_due)
            if cmd is not None
            else max(0.0, horizon - command_due)
        )

        prior = [
            p for p in shutter_points if (p[0], p[1]) < entry_pos
        ]
        already_closed = bool(prior and prior[-1][2] == "closed")
        closed_time: float | None
        closed_idx: int | None
        if already_closed:
            # 进入义务前快门本已 closed：视为在进入时刻即确认
            closed_time, closed_idx = start, prior[-1][1]
            close_overdue = 0.0
        else:
            fut = next(
                (
                    p
                    for p in shutter_points
                    if (p[0], p[1]) > entry_pos and p[2] == "closed"
                ),
                None,
            )
            if fut is not None:
                closed_time, closed_idx = fut[0], fut[1]
                close_overdue = max(0.0, closed_time - close_due)
            else:
                closed_time = closed_idx = None
                close_overdue = max(0.0, horizon - close_due)

        risk_intervals.append(
            RiskInterval(
                start=start,
                end=end,
                states=states,
                entry_event_indices=list(first["cause"]),
                projected=first["projected"],
                command_due=command_due,
                close_due=close_due,
                command_time=cmd.time if cmd is not None else None,
                command_event_index=cmd.event_index if cmd is not None else None,
                closed_time=closed_time,
                closed_event_index=closed_idx,
                already_closed_at_entry=already_closed,
                command_overdue=command_overdue,
                close_overdue=close_overdue,
            )
        )
        i = j

    return InterlockResult(
        status_timeline=status_timeline,
        risk_intervals=risk_intervals,
        close_commands=commands,
        shutter_timeline=shutter_timeline,
        horizon=t_last,
    )
