"""HTTP API 测试（基于 FastAPI TestClient）。"""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _ref(t: float) -> float:
    return 120.0 * math.sin(0.8 * t) + 30.0 * t


_REF_TIMES = [i * 0.2 for i in range(26)]
_ENC_TIMES = [round(0.37 + 0.31 * i, 10) for i in range(16)]


def _calibrate_payload(lag: float = 0.5, bias: float = 30.0, **overrides):
    body = {
        "reference_times": _REF_TIMES,
        "reference_angles": [_ref(t) % 360.0 for t in _REF_TIMES],
        "encoder_times": _ENC_TIMES,
        "encoder_angles": [
            (_ref(t - lag) - bias) % 360.0 for t in _ENC_TIMES
        ],
        "lag_min": -1.0,
        "lag_max": 1.0,
        "lag_step": 0.1,
        "outlier_threshold": 3.0,
        "min_inliers": 12,
    }
    body.update(overrides)
    return body


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "dome-slit-tracker"


def test_track_feasible():
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [0.0, 350.0],
            "times": [1.0, 2.0],
            "initial_angle": 0.0,
            "max_speed": 20.0,
            "slit_width": 10.0,
            "cable_low": -180.0,
            "cable_high": 180.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    assert len(body["actions"]) == 2
    assert body["actions"][0]["lift"] == 0
    assert body["actions"][1]["lift"] == -1
    assert body["total_rotation"] > 0


def test_track_infeasible_speed():
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [100.0],
            "times": [1.0],
            "initial_angle": 0.0,
            "max_speed": 10.0,
            "slit_width": 10.0,
            "cable_low": -180.0,
            "cable_high": 180.0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["failure"]["index"] == 1
    assert "speed_limit" in body["failure"]["unmet_constraints"]
    assert "required_extra_rotation" in body["failure"]["details"]


def test_track_infeasible_cable():
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [180.0],
            "times": [1.0],
            "initial_angle": 0.0,
            "max_speed": 1000.0,
            "slit_width": 10.0,
            "cable_low": -10.0,
            "cable_high": 10.0,
        },
    )
    body = r.json()
    assert body["feasible"] is False
    assert "cable_limit" in body["failure"]["unmet_constraints"]


def test_validation_length_mismatch():
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [1.0, 2.0],
            "times": [1.0],
            "initial_angle": 0.0,
            "max_speed": 10.0,
            "slit_width": 10.0,
            "cable_low": -180.0,
            "cable_high": 180.0,
        },
    )
    assert r.status_code == 422


def test_validation_nonpositive_speed():
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [1.0],
            "times": [1.0],
            "initial_angle": 0.0,
            "max_speed": 0,
            "slit_width": 10.0,
            "cable_low": -180.0,
            "cable_high": 180.0,
        },
    )
    assert r.status_code == 422


def test_validation_times_must_increase():
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [1.0, 2.0],
            "times": [2.0, 1.0],
            "initial_angle": 0.0,
            "max_speed": 10.0,
            "slit_width": 10.0,
            "cable_low": -180.0,
            "cable_high": 180.0,
        },
    )
    assert r.status_code == 422


def test_track_wide_slit_multiple_lifts_exact_optimum():
    # 精确回归：宽狭缝 + 多整数圈候选时，曾返回累计转角 148.5 的次优
    # 路线；应为 121.75 的序列 [-72.25, -107, -107, -107, -107, -107]。
    r = client.post(
        "/api/v1/track",
        json={
            "target_azimuths": [387.75, 533.0, -12.25, 561.75, -207.0, -59.75],
            "times": [2.0, 5.0, 7.0, 11.0, 13.0, 16.0],
            "initial_angle": 14.75,
            "max_speed": 48.75,
            "slit_width": 200.0,
            "cable_low": -900.0,
            "cable_high": 900.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    assert body["total_rotation"] == pytest.approx(121.75, abs=1e-7)
    assert [a["dome_angle"] for a in body["actions"]] == pytest.approx(
        [-72.25, -107.0, -107.0, -107.0, -107.0, -107.0], abs=1e-7
    )
    assert [a["lift"] for a in body["actions"]] == [-1, -2, 0, -2, 0, 0]
    # 响应自洽：目标提升方位 azimuth + 360*lift 落在狭缝半宽内
    for a in body["actions"]:
        center = a["target_azimuth"] + 360 * a["lift"]
        assert abs(a["dome_angle"] - center) <= 100.0 + 1e-7


def test_openapi_docs_available():
    assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/calibrate-sync
# ---------------------------------------------------------------------------


def test_calibrate_async_sampling():
    r = client.post("/api/v1/calibrate-sync", json=_calibrate_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calibrated"] is True
    assert body["lag_seconds"] == pytest.approx(0.5, abs=1e-9)
    assert body["bias_deg"] == pytest.approx(30.0, abs=0.2)
    assert body["rmse_deg"] < 0.3
    assert len(body["inlier_indices"]) >= 12
    assert body["outlier_indices"] == []
    assert body["reason"] is None


def test_calibrate_crosses_zero_with_normalized_bias():
    # 角度输入全部归一在 [0,360)，信号穿越 0° 多次；偏差 185° 归一到 -175°
    r = client.post("/api/v1/calibrate-sync", json=_calibrate_payload(bias=185.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calibrated"] is True
    assert body["lag_seconds"] == pytest.approx(0.5, abs=1e-9)
    assert -180.0 <= body["bias_deg"] < 180.0
    assert body["bias_deg"] == pytest.approx(-175.0, abs=0.2)


def test_calibrate_negative_lag():
    # encoder(t) = R(t+0.4) - (-12)：负时延
    payload = _calibrate_payload(lag=-0.4, bias=-12.0)
    r = client.post("/api/v1/calibrate-sync", json=payload)
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["calibrated"] is True
    assert body["lag_seconds"] == pytest.approx(-0.4, abs=1e-9)
    assert body["bias_deg"] == pytest.approx(-12.0, abs=0.2)


def test_calibrate_outliers_indices():
    payload = _calibrate_payload(outlier_threshold=5.0, min_inliers=10)
    angles = payload["encoder_angles"]
    angles[4] = (angles[4] + 130.0) % 360.0
    angles[12] = (angles[12] - 100.0) % 360.0
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calibrated"] is True
    assert body["lag_seconds"] == pytest.approx(0.5, abs=1e-9)
    assert body["outlier_indices"] == [4, 12]
    assert 4 not in body["inlier_indices"] and 12 not in body["inlier_indices"]
    assert body["rmse_deg"] < 0.3


def test_calibrate_tie_abs_lag_then_smaller():
    constant = {
        "reference_times": [0.0, 1.0, 2.0, 3.0],
        "reference_angles": [10.0, 10.0, 10.0, 10.0],
        "encoder_times": [0.5, 1.5, 2.5],
        "encoder_angles": [5.0, 5.0, 5.0],
        "outlier_threshold": 1.0,
        "min_inliers": 1,
    }
    r = client.post(
        "/api/v1/calibrate-sync",
        json=constant | {"lag_min": -1.0, "lag_max": 1.0, "lag_step": 1.0},
    )
    body = r.json()
    assert body["calibrated"] is True
    assert body["lag_seconds"] == 0.0  # |-1|、|1| 与 0 并列，取 |lag| 最小

    r = client.post(
        "/api/v1/calibrate-sync",
        json=constant | {"lag_min": -1.0, "lag_max": 3.0, "lag_step": 2.0},
    )
    body = r.json()
    assert body["calibrated"] is True
    assert body["lag_seconds"] == -1.0  # -1 与 1 并列，取较小 lag


def test_calibrate_no_valid_candidate_reports_reason():
    payload = _calibrate_payload(lag_min=0.0, lag_max=0.0, lag_step=1.0,
                                 outlier_threshold=0.0001, min_inliers=50)
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calibrated"] is False
    assert body["reason"]
    assert "内点不足" in body["reason"]
    assert body["details"]["evaluated_candidates"] == 1
    assert body["lag_seconds"] == 0.0
    assert body["inlier_indices"] == []


def test_calibrate_no_overlap_reports_reason():
    r = client.post(
        "/api/v1/calibrate-sync",
        json={
            "reference_times": [0.0, 1.0],
            "reference_angles": [10.0, 20.0],
            "encoder_times": [10.0, 11.0],
            "encoder_angles": [10.0, 20.0],
            "lag_min": 0.0,
            "lag_max": 1.0,
            "lag_step": 0.5,
            "outlier_threshold": 5.0,
            "min_inliers": 1,
        },
    )
    body = r.json()
    assert body["calibrated"] is False
    assert "无重叠样本" in body["reason"]


def test_calibrate_validation_rejects_180_jump():
    payload = _calibrate_payload()
    payload["reference_angles"][:3] = [0.0, 180.0, 30.0]
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 422
    assert "180" in r.text


def test_calibrate_validation_times_must_increase():
    payload = _calibrate_payload()
    payload["encoder_times"][1] = payload["encoder_times"][0]
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 422


def test_calibrate_validation_length_mismatch():
    payload = _calibrate_payload()
    payload["reference_times"].append(9.9)
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 422


def test_calibrate_validation_lag_bounds_and_step():
    payload = _calibrate_payload(lag_min=1.0, lag_max=0.0, lag_step=0.1)
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 422

    payload = _calibrate_payload(lag_step=0.0)
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 422


def test_calibrate_validation_min_inliers_positive():
    payload = _calibrate_payload(min_inliers=0)
    r = client.post("/api/v1/calibrate-sync", json=payload)
    assert r.status_code == 422
