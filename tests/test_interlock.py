"""POST /api/v1/audit-safety-interlock 的边界回归测试（含引擎直调）。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planner.interlock import (
    Event,
    InterlockValidationError,
    audit_interlock,
)

client = TestClient(app)

_ENGINE_PARAMS = dict(
    wind_trip=20.0,
    wind_reset=15.0,
    rain_trip=10.0,
    rain_reset=5.0,
    trip_duration=5.0,
    reset_duration=5.0,
    stale_after=30.0,
    command_deadline=60.0,
    close_deadline=120.0,
)


def _payload(events, **overrides):
    body = {"events": events, **_ENGINE_PARAMS}
    body.update(overrides)
    return body


def _post(payload):
    return client.post("/api/v1/audit-safety-interlock", json=payload)


# ---------------------------------------------------------------------------
# 合法请求：200 + 状态时间线 / 关闭义务 / 风险区间 / 超期量 / 原始事件索引
# ---------------------------------------------------------------------------


def test_audit_happy_path_full_timeline():
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 1.0, "value": 10.0},           # 1 -> safe
        {"kind": "rain", "time": 2.0, "value": 1.0},            # 2
        {"kind": "wind", "time": 3.0, "value": 25.0},           # 3 -> 越限计时
        {"kind": "close_command", "time": 10.0},                # 4
        {"kind": "shutter", "time": 12.0, "state": "closing"},  # 5
        {"kind": "shutter", "time": 14.0, "state": "closed"},   # 6
        {"kind": "wind", "time": 20.0, "value": 10.0},          # 7 -> 复位计时
        {"kind": "shutter", "time": 30.0, "state": "opening"},  # 8
        {"kind": "shutter", "time": 32.0, "state": "open"},     # 9
    ]
    r = _post(_payload(events))
    assert r.status_code == 200, r.text
    body = r.json()

    # 状态时间线：8=3+5（触发成熟）、25=20+5（复位成熟）、50=20+30（中断投影）
    timeline = body["status_timeline"]
    assert [(s["start"], s["end"], s["state"]) for s in timeline] == [
        (0.0, 1.0, "unknown"),
        (1.0, 8.0, "safe"),
        (8.0, 25.0, "unsafe"),
        (25.0, 50.0, "safe"),
        (50.0, None, "unknown"),
    ]
    assert [s["cause_event_indices"] for s in timeline] == [
        [],
        [1],
        [3, 2],
        [7, 2],
        [7, 2],
    ]
    assert [s["projected"] for s in timeline] == [False] * 4 + [True]

    # 关闭义务与超期量（含初始 unknown 段产生的义务）
    risks = body["risk_intervals"]
    assert [(i["start"], i["end"], i["states"]) for i in risks] == [
        (0.0, 1.0, ["unknown"]),
        (8.0, 25.0, ["unsafe"]),
        (50.0, None, ["unknown"]),
    ]
    assert risks[1]["entry_event_indices"] == [3, 2]
    assert risks[1]["command_due"] == 68.0   # 8 + 60
    assert risks[1]["close_due"] == 128.0    # 8 + 120
    assert risks[1]["command_time"] == 10.0
    assert risks[1]["command_event_index"] == 4
    assert risks[1]["closed_time"] == 14.0
    assert risks[1]["closed_event_index"] == 6
    assert risks[1]["already_closed_at_entry"] is False
    assert risks[1]["command_overdue"] == 0.0
    assert risks[1]["close_overdue"] == 0.0
    # 投影产生的开口区间：义务时限钉死但截算到 horizon，未履行
    assert risks[2]["projected"] is True
    assert risks[2]["command_time"] is None
    assert risks[2]["closed_time"] is None
    assert risks[2]["command_overdue"] == 0.0  # max(0, 32 - 110)

    assert body["close_commands"] == [{"time": 10.0, "event_index": 4}]
    shutter = body["shutter_timeline"]
    assert [(s["start"], s["end"], s["state"], s["event_index"])
            for s in shutter] == [
        (0.0, 12.0, "open", 0),
        (12.0, 14.0, "closing", 5),
        (14.0, 30.0, "closed", 6),
        (30.0, 32.0, "opening", 8),
        (32.0, None, "open", 9),
    ]
    assert body["horizon"] == 32.0


def test_audit_overdue_amounts_and_merged_interval():
    # trip_duration=0：当刻进入 unsafe；命令/关闭均超期；unsafe 末段与投影
    # unknown 无 safe 间隔，合并为同一段连续风险
    events = [
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 0 -> safe@0
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 1
        {"kind": "wind", "time": 1.0, "value": 25.0},           # 2 -> unsafe@1
        {"kind": "close_command", "time": 20.0},                # 3
        {"kind": "shutter", "time": 25.0, "state": "closing"},  # 4
        {"kind": "shutter", "time": 30.0, "state": "closed"},   # 5
        {"kind": "wind", "time": 40.0, "value": 5.0},           # 6（rain 缺通道，不复位）
    ]
    r = _post(_payload(events, trip_duration=0.0, reset_duration=2.0,
                       stale_after=100.0, command_deadline=5.0,
                       close_deadline=10.0))
    assert r.status_code == 200, r.text
    body = r.json()

    assert [(s["start"], s["end"], s["state"], s["projected"])
            for s in body["status_timeline"]] == [
        (0.0, 1.0, "safe", False),
        (1.0, 140.0, "unsafe", False),
        (140.0, None, "unknown", True),
    ]
    risks = body["risk_intervals"]
    assert len(risks) == 1
    ri = risks[0]
    assert ri["start"] == 1.0 and ri["end"] is None
    assert ri["states"] == ["unsafe", "unknown"]
    assert ri["entry_event_indices"] == [2]
    assert ri["command_due"] == 6.0 and ri["close_due"] == 11.0
    assert ri["command_event_index"] == 3
    assert ri["closed_event_index"] == 5
    assert ri["command_overdue"] == 14.0  # 20 - 6
    assert ri["close_overdue"] == 19.0    # 30 - 11
    assert body["horizon"] == 40.0


def test_audit_already_closed_at_entry():
    events = [
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 0 -> safe@0
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 1
        {"kind": "shutter", "time": 1.0, "state": "closing"},   # 2
        {"kind": "shutter", "time": 2.0, "state": "closed"},    # 3
        {"kind": "wind", "time": 5.0, "value": 30.0},           # 4 -> unsafe@5
    ]
    r = _post(_payload(events, trip_duration=0.0, stale_after=1000.0))
    assert r.status_code == 200, r.text
    risks = r.json()["risk_intervals"]
    assert len(risks) == 1
    ri = risks[0]
    assert ri["start"] == 5.0
    # 进入风险时快门已 closed：视为进入时刻确认，不超期
    assert ri["already_closed_at_entry"] is True
    assert ri["closed_time"] == 5.0
    assert ri["closed_event_index"] == 3
    assert ri["close_overdue"] == 0.0
    # 从未发出关顶命令：超期量截算到 horizon（5），未超期
    assert ri["command_time"] is None
    assert ri["command_overdue"] == 0.0


def test_audit_stale_boundary_strictly_greater():
    # 间隔恰为 stale_after 不中断；严格超过才中断
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 1.0, "value": 5.0},            # 1
        {"kind": "wind", "time": 11.0, "value": 5.0},           # 2 间隔=10 不中断
        {"kind": "shutter", "time": 21.5, "state": "closing"},  # 3 间隔 10.5 -> 21 中断
        {"kind": "wind", "time": 22.0, "value": 5.0},           # 4 恢复观测
    ]
    r = _post(_payload(events, stale_after=10.0))
    assert r.status_code == 200, r.text
    timeline = r.json()["status_timeline"]
    assert [(s["start"], s["end"], s["state"]) for s in timeline] == [
        (0.0, 1.0, "unknown"),
        (1.0, 21.0, "safe"),     # 关键点：11 处未中断，21 处才中断
        (21.0, 22.0, "unknown"),
        (22.0, 32.0, "safe"),
        (32.0, None, "unknown"),
    ]
    assert timeline[-1]["projected"] is True


def test_audit_same_tick_order_and_zero_length_shutter_segments():
    # 顺序 A：同刻先以 trip_duration=0 的天气事件进入 unsafe，随后才提交
    # close_command / closing / closed —— 后到的 closed 是入口**之后**的
    # 履行，不得标 already_closed_at_entry；同刻事件严格按输入顺序处理，
    # 同刻多次快门转换仍产生零长段
    events = [
        {"kind": "shutter", "time": 5.0, "state": "open"},      # 0
        {"kind": "wind", "time": 5.0, "value": 30.0},           # 1 -> 当刻 unsafe
        {"kind": "close_command", "time": 5.0},                 # 2
        {"kind": "shutter", "time": 5.0, "state": "closing"},   # 3
        {"kind": "shutter", "time": 5.0, "state": "closed"},    # 4
    ]
    r = _post(_payload(events, trip_duration=0.0, stale_after=100.0,
                       command_deadline=10.0, close_deadline=20.0))
    assert r.status_code == 200, r.text
    body = r.json()

    # 首个事件时刻即翻转为 unsafe（初始 unknown 被同刻覆盖，不产生区间）
    assert [(s["start"], s["end"], s["state"])
            for s in body["status_timeline"]] == [
        (5.0, 105.0, "unsafe"),
        (105.0, None, "unknown"),
    ]
    # 同刻零长段保留以反映输入顺序
    assert [(s["start"], s["end"], s["state"], s["event_index"])
            for s in body["shutter_timeline"]] == [
        (5.0, 5.0, "open", 0),
        (5.0, 5.0, "closing", 3),
        (5.0, None, "closed", 4),
    ]
    ri = body["risk_intervals"][0]
    assert ri["start"] == 5.0
    assert ri["entry_event_indices"] == [1]
    # 命令在入口（天气事件 1）之后：履行本次义务
    assert ri["command_time"] == 5.0
    assert ri["command_event_index"] == 2
    assert ri["command_overdue"] == 0.0
    # closed 也在入口之后：属于履行而非“进入时已关闭”
    assert ri["already_closed_at_entry"] is False
    assert ri["closed_time"] == 5.0
    assert ri["closed_event_index"] == 4
    assert ri["close_overdue"] == 0.0
    # 命令全量列表不受归属判断影响
    assert body["close_commands"] == [{"time": 5.0, "event_index": 2}]


def test_audit_same_tick_command_before_weather_entry():
    # 顺序 B（相反顺序）：同刻先提交 close_command / closing / closed，
    # 触发天气事件排在其后 —— 义务尚未产生时的命令不得计作履行；入口前
    # 已生效的 closed 才设置 already_closed_at_entry
    events = [
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 0 -> safe@0
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 1
        {"kind": "close_command", "time": 5.0},                 # 2（入口前）
        {"kind": "shutter", "time": 5.0, "state": "closing"},   # 3（入口前）
        {"kind": "shutter", "time": 5.0, "state": "closed"},    # 4（入口前）
        {"kind": "wind", "time": 5.0, "value": 30.0},           # 5 -> 当刻 unsafe
        {"kind": "wind", "time": 10.0, "value": 30.0},          # 6（拉远 horizon）
    ]
    r = _post(_payload(events, trip_duration=0.0, stale_after=1000.0,
                       command_deadline=2.0, close_deadline=10.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"], s["projected"])
            for s in body["status_timeline"]] == [
        (0.0, 5.0, "safe", False),
        (5.0, 1010.0, "unsafe", False),
        (1010.0, None, "unknown", True),
    ]
    risks = body["risk_intervals"]
    assert len(risks) == 1
    ri = risks[0]
    assert ri["start"] == 5.0 and ri["end"] is None
    assert ri["entry_event_indices"] == [5]
    # 唯一命令在入口之前：不履行；超期量截算到 horizon（10 − 7 = 3）
    assert ri["command_time"] is None
    assert ri["command_event_index"] is None
    assert ri["command_overdue"] == 3.0
    # closed 在入口之前已生效：视为进入时刻确认，索引为入口前的事件 4
    assert ri["already_closed_at_entry"] is True
    assert ri["closed_time"] == 5.0
    assert ri["closed_event_index"] == 4
    assert ri["close_overdue"] == 0.0
    # 全量命令列表仍保留入口前的命令
    assert body["close_commands"] == [{"time": 5.0, "event_index": 2}]


def test_audit_timer_matures_before_same_tick_group():
    # trip_duration > 0：计时器在同刻事件组之前成熟（run_fire 恰为组时刻），
    # 入口位于 (t, -1)，该时刻全部输入事件都在入口之后、可履行义务
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 1 -> safe@0
        {"kind": "wind", "time": 10.0, "value": 30.0},          # 2 -> 计时至 15
        {"kind": "close_command", "time": 15.0},                # 3（入口后）
        {"kind": "shutter", "time": 15.0, "state": "closing"},  # 4（入口后）
        {"kind": "shutter", "time": 15.0, "state": "closed"},   # 5（入口后）
        {"kind": "wind", "time": 20.0, "value": 30.0},          # 6（horizon）
    ]
    r = _post(_payload(events, trip_duration=5.0, stale_after=1000.0,
                       command_deadline=0.0, close_deadline=0.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"], s["projected"])
            for s in body["status_timeline"]] == [
        (0.0, 15.0, "safe", False),
        (15.0, 1020.0, "unsafe", False),
        (1020.0, None, "unknown", True),
    ]
    ri = body["risk_intervals"][0]
    assert ri["start"] == 15.0
    assert ri["entry_event_indices"] == [2]
    assert ri["projected"] is False
    assert ri["command_time"] == 15.0
    assert ri["command_event_index"] == 3
    assert ri["command_overdue"] == 0.0
    assert ri["already_closed_at_entry"] is False
    assert ri["closed_time"] == 15.0
    assert ri["closed_event_index"] == 5
    assert ri["close_overdue"] == 0.0


def test_audit_stale_boundary_command_not_credited():
    # 中断要求“严格超过”：中断回拨到 stale_bound=10 时位于该时刻全部输入
    # 事件之后 (10, +∞)，恰在边界到达的命令不能履行 unknown 义务；
    # 超期量截算到闭区间末端 20（10 + deadline 5 → 5 秒）
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 1 -> safe@0
        {"kind": "close_command", "time": 10.0},                # 2（恰在边界）
        {"kind": "wind", "time": 20.0, "value": 5.0},           # 3 -> 20 恢复
    ]
    r = _post(_payload(events, stale_after=10.0,
                       command_deadline=5.0, close_deadline=5.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"]) for s in body["status_timeline"]] == [
        (0.0, 10.0, "safe"),
        (10.0, 20.0, "unknown"),
        (20.0, 30.0, "safe"),
        (30.0, None, "unknown"),
    ]
    # 初始段在 wind@0 被同刻覆盖；第一段风险是回拨的 unknown[10,20)
    ri = next(i for i in body["risk_intervals"] if i["start"] == 10.0)
    assert ri["end"] == 20.0
    assert ri["command_time"] is None
    assert ri["command_event_index"] is None
    assert ri["command_overdue"] == 5.0   # 20 − 15
    assert ri["already_closed_at_entry"] is False
    assert ri["closed_time"] is None
    assert ri["closed_event_index"] is None
    assert ri["close_overdue"] == 5.0     # 20 − 15
    # 全量命令列表仍记录边界命令
    assert body["close_commands"] == [{"time": 10.0, "event_index": 2}]


def test_audit_projected_entry_ignores_last_tick_actions():
    # 末尾投影：触发计时在末事件之后（15 > 10）成熟，投影入口 (15, -1)。
    # 末时刻 10 的命令 / closed 都在入口之前：命令不计履行（截算到 horizon
    # 10，未超 due=15 故为 0）；入口前已 closed → already_closed_at_entry，
    # 视为入口时刻 15 确认并返回入口前的事件索引
    events = [
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 0 -> safe@0
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 1
        {"kind": "wind", "time": 10.0, "value": 30.0},          # 2 -> 计时至 15
        {"kind": "close_command", "time": 10.0},                # 3（投影入口前）
        {"kind": "shutter", "time": 10.0, "state": "closing"},  # 4（投影入口前）
        {"kind": "shutter", "time": 10.0, "state": "closed"},   # 5（投影入口前）
    ]
    r = _post(_payload(events, trip_duration=5.0, stale_after=1000.0,
                       command_deadline=0.0, close_deadline=0.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"], s["projected"])
            for s in body["status_timeline"]] == [
        (0.0, 15.0, "safe", False),
        (15.0, 1010.0, "unsafe", True),
        (1010.0, None, "unknown", True),
    ]
    ri = body["risk_intervals"][0]
    assert ri["start"] == 15.0 and ri["end"] is None
    assert ri["projected"] is True
    assert ri["entry_event_indices"] == [2]
    assert ri["command_due"] == 15.0 and ri["close_due"] == 15.0
    assert ri["command_time"] is None
    assert ri["command_event_index"] is None
    assert body["horizon"] == 10.0
    assert ri["command_overdue"] == 0.0   # max(0, 10 − 15)
    assert ri["already_closed_at_entry"] is True
    assert ri["closed_time"] == 15.0
    assert ri["closed_event_index"] == 5
    assert ri["close_overdue"] == 0.0


def test_audit_shutter_full_cycle_and_stay_allowed():
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},
        {"kind": "shutter", "time": 1.0, "state": "closing"},
        {"kind": "shutter", "time": 2.0, "state": "closed"},
        {"kind": "shutter", "time": 3.0, "state": "opening"},
        {"kind": "shutter", "time": 4.0, "state": "open"},   # 循环回绕
        {"kind": "shutter", "time": 5.0, "state": "open"},   # 停留原态
        {"kind": "wind", "time": 6.0, "value": 1.0},
    ]
    r = _post(_payload(events))
    assert r.status_code == 200, r.text
    assert [s["state"] for s in r.json()["shutter_timeline"]] == [
        "open", "closing", "closed", "opening", "open", "open",
    ]


def test_audit_deadline_exact_boundary_not_overdue():
    # 命令/关闭恰在时限时刻：超期量为 0
    events = [
        {"kind": "wind", "time": 0.0, "value": 5.0},            # 0 -> safe@0
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 1
        {"kind": "wind", "time": 1.0, "value": 30.0},           # 2 -> unsafe@1
        {"kind": "close_command", "time": 11.0},                # 3 恰在 1+10
        {"kind": "shutter", "time": 11.0, "state": "closing"},  # 4
        {"kind": "shutter", "time": 21.0, "state": "closed"},   # 5 恰在 1+20
    ]
    r = _post(_payload(events, trip_duration=0.0, stale_after=1000.0,
                       command_deadline=10.0, close_deadline=20.0))
    assert r.status_code == 200, r.text
    ri = r.json()["risk_intervals"][0]
    assert ri["command_time"] == 11.0 and ri["closed_time"] == 21.0
    assert ri["command_overdue"] == 0.0
    assert ri["close_overdue"] == 0.0


def test_audit_projected_pending_trip_timer():
    # 末事件后触发计时才成熟：投影出 unsafe 段，义务无法履行
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 1.0, "value": 30.0},           # 1 -> 越限计时
    ]
    r = _post(_payload(events, trip_duration=10.0, stale_after=1000.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"], s["projected"])
            for s in body["status_timeline"]] == [
        (0.0, 1.0, "unknown", False),
        (1.0, 11.0, "safe", False),
        (11.0, 1001.0, "unsafe", True),
        (1001.0, None, "unknown", True),
    ]
    ri = body["risk_intervals"][-1]
    assert ri["start"] == 11.0 and ri["end"] is None
    assert ri["projected"] is True
    assert ri["states"] == ["unsafe", "unknown"]
    assert ri["command_time"] is None and ri["closed_time"] is None
    # 开口义务超期量截算到 horizon（末事件时刻 1）
    assert body["horizon"] == 1.0
    assert ri["command_overdue"] == 0.0
    assert ri["close_overdue"] == 0.0


# ---------------------------------------------------------------------------
# 语义非法（引擎拒绝）→ 422
# ---------------------------------------------------------------------------


def test_audit_missing_shutter_initial_state_422():
    r = _post(_payload([{"kind": "wind", "time": 1.0, "value": 5.0}]))
    assert r.status_code == 422
    assert "快门" in r.json()["detail"]


def test_audit_shutter_skip_intermediate_state_422():
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},
        {"kind": "shutter", "time": 1.0, "state": "closed"},  # 跳过 closing
    ]
    r = _post(_payload(events))
    assert r.status_code == 422
    body = r.json()
    assert "跳变" in body["detail"]
    assert body["event_index"] == 1


def test_audit_shutter_reverse_step_422():
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},
        {"kind": "shutter", "time": 1.0, "state": "closing"},
        {"kind": "shutter", "time": 2.0, "state": "closed"},
        {"kind": "shutter", "time": 3.0, "state": "closing"},  # 应为 opening
    ]
    r = _post(_payload(events))
    assert r.status_code == 422
    assert r.json()["event_index"] == 3


# ---------------------------------------------------------------------------
# 参数与时序校验（请求模型拒绝）→ 422
# ---------------------------------------------------------------------------


def test_audit_events_time_must_be_non_decreasing():
    events = [
        {"kind": "shutter", "time": 2.0, "state": "open"},
        {"kind": "wind", "time": 1.0, "value": 5.0},
    ]
    assert _post(_payload(events)).status_code == 422


def test_audit_same_tick_events_accepted():
    events = [
        {"kind": "shutter", "time": 1.0, "state": "open"},
        {"kind": "wind", "time": 1.0, "value": 5.0},
    ]
    assert _post(_payload(events)).status_code == 200


def test_audit_reset_threshold_must_not_exceed_trip():
    events = [{"kind": "shutter", "time": 0.0, "state": "open"}]
    assert _post(_payload(events, wind_reset=21.0)).status_code == 422
    assert _post(_payload(events, rain_reset=11.0)).status_code == 422
    # reset == trip（零迟滞）合法
    assert _post(_payload(events, wind_reset=20.0)).status_code == 200


def test_audit_negative_weather_value_422():
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},
        {"kind": "wind", "time": 1.0, "value": -1.0},
    ]
    assert _post(_payload(events)).status_code == 422


def test_audit_duration_and_stale_constraints():
    events = [{"kind": "shutter", "time": 0.0, "state": "open"}]
    assert _post(_payload(events, trip_duration=-1.0)).status_code == 422
    assert _post(_payload(events, reset_duration=-1.0)).status_code == 422
    assert _post(_payload(events, stale_after=0.0)).status_code == 422
    assert _post(_payload(events, command_deadline=-1.0)).status_code == 422
    assert _post(_payload(events, close_deadline=-1.0)).status_code == 422
    # 零时长合法（当刻触发/复位、立即到期）
    assert _post(_payload(events, trip_duration=0.0, reset_duration=0.0,
                          command_deadline=0.0, close_deadline=0.0)
                 ).status_code == 200


def test_audit_empty_events_422():
    assert _post(_payload([])).status_code == 422


def test_audit_unknown_event_kind_422():
    events = [{"kind": "hail", "time": 1.0, "value": 5.0}]
    assert _post(_payload(events)).status_code == 422


def test_audit_missing_shutter_state_field_422():
    events = [{"kind": "shutter", "time": 0.0}]
    assert _post(_payload(events)).status_code == 422


# ---------------------------------------------------------------------------
# 引擎直调回归：首次天气状态切换不再抛 UnboundLocalError
# ---------------------------------------------------------------------------


def test_engine_direct_first_weather_switch_no_unbound_local():
    events = [
        Event(index=0, time=0.0, kind="shutter", state="open"),
        Event(index=1, time=1.0, kind="wind", value=5.0),
    ]
    result = audit_interlock(events, **_ENGINE_PARAMS)
    assert [s.state for s in result.status_timeline][:2] == ["unknown", "safe"]
    assert result.status_timeline[1].cause_event_indices == [1]


def test_engine_direct_missing_shutter_initial_state_raises():
    with pytest.raises(InterlockValidationError):
        audit_interlock(
            [Event(index=0, time=1.0, kind="wind", value=5.0)],
            **_ENGINE_PARAMS,
        )


def test_engine_direct_empty_events_raises():
    with pytest.raises(InterlockValidationError):
        audit_interlock([], **_ENGINE_PARAMS)
