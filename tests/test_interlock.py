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
    # 同刻事件严格按输入顺序处理；同刻多次快门转换产生零长段
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
    assert ri["command_event_index"] == 2
    assert ri["command_overdue"] == 0.0
    # 入口在事件 1 处：同刻后到的 closed（事件 4）是履行而非“进入时已关闭”
    assert ri["already_closed_at_entry"] is False
    assert ri["closed_time"] == 5.0
    assert ri["closed_event_index"] == 4
    assert ri["close_overdue"] == 0.0


def test_audit_same_tick_command_before_trip_weather_not_counted():
    # 相反顺序：命令与 closed 排在触发天气之前（同刻）——义务尚未产生，
    # 命令不得计作履行；入口前已生效的 closed 置 already_closed_at_entry
    events = [
        {"kind": "shutter", "time": 5.0, "state": "closed"},    # 0 入口前已关闭
        {"kind": "close_command", "time": 5.0},                 # 1 义务产生前的命令
        {"kind": "wind", "time": 5.0, "value": 30.0},           # 2 -> 当刻 unsafe
        {"kind": "close_command", "time": 6.0},                 # 3 义务产生后的命令
    ]
    r = _post(_payload(events, trip_duration=0.0, stale_after=100.0,
                       command_deadline=10.0, close_deadline=20.0))
    assert r.status_code == 200, r.text
    ri = r.json()["risk_intervals"][0]
    assert ri["start"] == 5.0
    assert ri["entry_event_indices"] == [2]
    # 同刻但排在触发天气之前的命令（事件 1）不计入；首个可履行命令是事件 3
    assert ri["command_time"] == 6.0
    assert ri["command_event_index"] == 3
    assert ri["command_overdue"] == 0.0
    # 入口前（事件 0）已生效的 closed：视为进入时刻确认
    assert ri["already_closed_at_entry"] is True
    assert ri["closed_time"] == 5.0
    assert ri["closed_event_index"] == 0
    assert ri["close_overdue"] == 0.0


def test_audit_same_tick_command_before_trip_weather_no_later_command():
    # 义务产生前的唯一命令不计入：命令义务未履行，超期量截算到 horizon
    events = [
        {"kind": "shutter", "time": 5.0, "state": "open"},      # 0
        {"kind": "close_command", "time": 5.0},                 # 1 义务产生前
        {"kind": "wind", "time": 5.0, "value": 30.0},           # 2 -> 当刻 unsafe
    ]
    r = _post(_payload(events, trip_duration=0.0, stale_after=100.0,
                       command_deadline=10.0, close_deadline=20.0))
    assert r.status_code == 200, r.text
    ri = r.json()["risk_intervals"][0]
    assert ri["start"] == 5.0
    assert ri["command_time"] is None
    assert ri["command_event_index"] is None
    # 超期量截算到 horizon（末事件时刻 5）：max(0, 5 - 15) = 0
    assert ri["command_overdue"] == 0.0
    assert ri["already_closed_at_entry"] is False
    assert ri["closed_time"] is None


def test_audit_trip_timer_maturing_at_group_tick_entry_precedes_group():
    # 触发计时恰在某同刻事件组的时刻成熟：计时器先于该组事件成熟，
    # 组内命令/closed 均在入口之后，可履行本次义务（非 already_closed）
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 0.0, "value": 30.0},           # 1 -> 越限计时
        {"kind": "close_command", "time": 5.0},                 # 2 与触发成熟同刻
        {"kind": "shutter", "time": 5.0, "state": "closing"},   # 3
        {"kind": "shutter", "time": 5.0, "state": "closed"},    # 4
    ]
    r = _post(_payload(events, trip_duration=5.0, stale_after=1000.0,
                       command_deadline=10.0, close_deadline=20.0))
    assert r.status_code == 200, r.text
    body = r.json()
    # 触发计时在 0+5=5 成熟，先于同刻事件组
    assert [(s["start"], s["end"], s["state"])
            for s in body["status_timeline"]][:2] == [
        (0.0, 5.0, "safe"),
        (5.0, 1000.0, "unsafe"),
    ]
    ri = body["risk_intervals"][0]
    assert ri["start"] == 5.0
    assert ri["command_time"] == 5.0
    assert ri["command_event_index"] == 2
    assert ri["command_overdue"] == 0.0
    assert ri["already_closed_at_entry"] is False
    assert ri["closed_time"] == 5.0
    assert ri["closed_event_index"] == 4
    assert ri["close_overdue"] == 0.0


def test_audit_stale_boundary_same_tick_events_precede_entry():
    # 中断要求严格超过：与 stale_bound 同刻的事件仍在旧状态下处理，
    # 其命令不得履行中断产生的新义务
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 10.0, "value": 5.0},           # 1 -> safe@10
        {"kind": "close_command", "time": 40.0},                # 2 恰在中断边界
        {"kind": "wind", "time": 45.0, "value": 5.0},           # 3 恢复观测
    ]
    r = _post(_payload(events, stale_after=30.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"])
            for s in body["status_timeline"]] == [
        (0.0, 10.0, "unknown"),
        (10.0, 40.0, "safe"),
        (40.0, 45.0, "unknown"),   # 10+30=40 处中断
        (45.0, 75.0, "safe"),
        (75.0, None, "unknown"),
    ]
    risks = body["risk_intervals"]
    # 中断区间的入口在同刻命令（事件 2）之后：命令未履行该义务
    stale_ri = risks[1]
    assert (stale_ri["start"], stale_ri["end"]) == (40.0, 45.0)
    assert stale_ri["command_time"] is None
    assert stale_ri["command_event_index"] is None
    assert stale_ri["command_overdue"] == 0.0  # max(0, 45 - 100)
    # 同一条命令对初始 unknown 段的义务（入口在首个事件之前）仍然有效
    assert risks[0]["command_time"] == 40.0
    assert risks[0]["command_event_index"] == 2


def test_audit_projected_stale_entry_after_last_tick_events():
    # 末尾投影：stale_bound 恰等于末事件时刻，投影入口在全部同刻事件之后——
    # 同刻命令不得履行投影义务，同刻已生效的 closed 仍属入口之前
    events = [
        {"kind": "shutter", "time": 0.0, "state": "open"},      # 0
        {"kind": "wind", "time": 10.0, "value": 5.0},           # 1 -> safe@10
        {"kind": "close_command", "time": 40.0},                # 2 与中断边界同刻
        {"kind": "shutter", "time": 40.0, "state": "closing"},  # 3
        {"kind": "shutter", "time": 40.0, "state": "closed"},   # 4
    ]
    r = _post(_payload(events, stale_after=30.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"], s["state"], s["projected"])
            for s in body["status_timeline"]] == [
        (0.0, 10.0, "unknown", False),
        (10.0, 40.0, "safe", False),
        (40.0, None, "unknown", True),
    ]
    ri = body["risk_intervals"][-1]
    assert ri["start"] == 40.0 and ri["end"] is None
    assert ri["projected"] is True
    # 投影入口在事件 2 之后：同刻命令不得计作履行
    assert ri["command_time"] is None
    assert ri["command_event_index"] is None
    assert ri["command_overdue"] == 0.0  # max(0, 40 - 100)
    # 同刻 closed（事件 4）在投影入口之前已生效
    assert ri["already_closed_at_entry"] is True
    assert ri["closed_time"] == 40.0
    assert ri["closed_event_index"] == 4
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
