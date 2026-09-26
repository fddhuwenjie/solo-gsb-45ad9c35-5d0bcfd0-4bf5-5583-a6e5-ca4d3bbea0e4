"""FastAPI 入口：健康检查、圆顶狭缝跟踪求解与安全联锁审计接口。"""

from __future__ import annotations

import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .models import (
    ActionItem,
    AuditSafetyInterlockRequest,
    AuditSafetyInterlockResponse,
    CalibrateSyncRequest,
    CalibrateSyncResponse,
    FailureDetail,
    HealthResponse,
    InterlockCommandRecord,
    InterlockRiskInterval,
    InterlockShutterSegment,
    InterlockStatusSegment,
    TrackRequest,
    TrackResponse,
)
from .planner.calibration import calibrate_sync, normalize_bias
from .planner.interlock import Event, InterlockValidationError, audit_interlock
from .planner.solver import Problem, solve

app = FastAPI(
    title="天文台圆顶狭缝跟踪 API",
    description=(
        "为时间序列目标方位计算圆顶动作时间线：目标始终位于狭缝内、"
        "满足转速与线缆边界约束、累计转角最小，等代价取字典序最小序列。"
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="dome-slit-tracker")


@app.exception_handler(InterlockValidationError)
async def interlock_validation_error_handler(
    _request: Request, exc: InterlockValidationError
) -> JSONResponse:
    # 事件流语义非法（快门缺初值 / 跳态等）→ HTTP 422
    return JSONResponse(
        status_code=422,
        content={"detail": exc.message, "event_index": exc.event_index},
    )


@app.post("/api/v1/track", response_model=TrackResponse, tags=["tracking"])
def track(req: TrackRequest) -> TrackResponse:
    problem = Problem(
        target_azimuths=req.target_azimuths,
        times=req.times,
        initial_angle=req.initial_angle,
        max_speed=req.max_speed,
        slit_width=req.slit_width,
        cable_lo=req.cable_low,
        cable_hi=req.cable_high,
    )
    sol = solve(problem)

    if sol.feasible:
        return TrackResponse(
            feasible=True,
            total_rotation=round(sol.total_rotation, 9),
            actions=[
                ActionItem(
                    index=a.index,
                    time=a.time,
                    dome_angle=round(a.dome_angle, 9),
                    target_azimuth=a.target_azimuth,
                    lift=a.lift,
                    move_delta=round(a.move_delta, 9),
                    move_speed=round(a.move_speed, 9),
                    cable_low=a.cable_low,
                    cable_high=a.cable_high,
                )
                for a in sol.actions
            ],
        )

    f = sol.failure
    return TrackResponse(
        feasible=False,
        failure=FailureDetail(
            index=f.index,
            time=f.time,
            target_azimuth=None if (isinstance(f.target_azimuth, float) and
                                    math.isnan(f.target_azimuth)) else f.target_azimuth,
            unmet_constraints=f.unmet,  # type: ignore[arg-type]
            details=f.details,
        ),
    )


@app.post(
    "/api/v1/calibrate-sync",
    response_model=CalibrateSyncResponse,
    tags=["calibration"],
)
def calibrate(req: CalibrateSyncRequest) -> CalibrateSyncResponse:
    outcome = calibrate_sync(
        reference_times=req.reference_times,
        reference_angles=req.reference_angles,
        encoder_times=req.encoder_times,
        encoder_angles=req.encoder_angles,
        lag_min=req.lag_min,
        lag_max=req.lag_max,
        lag_step=req.lag_step,
        outlier_threshold=req.outlier_threshold,
        min_inliers=req.min_inliers,
    )

    if not outcome.calibrated:
        return CalibrateSyncResponse(
            calibrated=False,
            reason=outcome.reason,
            details=outcome.details,
        )

    return CalibrateSyncResponse(
        calibrated=True,
        bias_deg=round(normalize_bias(outcome.bias), 9),
        lag_seconds=round(outcome.lag, 9),
        rmse_deg=round(outcome.rmse, 9),
        inlier_indices=outcome.inliers,
        outlier_indices=outcome.outliers,
        details=outcome.details,
    )


def _r9(value: float | None) -> float | None:
    return None if value is None else round(value, 9)


@app.post(
    "/api/v1/audit-safety-interlock",
    response_model=AuditSafetyInterlockResponse,
    tags=["interlock"],
)
def audit_safety_interlock(
    req: AuditSafetyInterlockRequest,
) -> AuditSafetyInterlockResponse:
    events = [
        Event(
            index=i,
            time=e.time,
            kind=e.kind,
            value=getattr(e, "value", None),
            state=getattr(e, "state", None),
        )
        for i, e in enumerate(req.events)
    ]
    result = audit_interlock(
        events,
        wind_trip=req.wind_trip,
        wind_reset=req.wind_reset,
        rain_trip=req.rain_trip,
        rain_reset=req.rain_reset,
        trip_duration=req.trip_duration,
        reset_duration=req.reset_duration,
        stale_after=req.stale_after,
        command_deadline=req.command_deadline,
        close_deadline=req.close_deadline,
    )

    return AuditSafetyInterlockResponse(
        status_timeline=[
            InterlockStatusSegment(
                start=round(s.start, 9),
                end=_r9(s.end),
                state=s.state,  # type: ignore[arg-type]
                cause_event_indices=s.cause_event_indices,
                projected=s.projected,
            )
            for s in result.status_timeline
        ],
        risk_intervals=[
            InterlockRiskInterval(
                start=round(r.start, 9),
                end=_r9(r.end),
                states=r.states,  # type: ignore[arg-type]
                entry_event_indices=r.entry_event_indices,
                projected=r.projected,
                command_due=round(r.command_due, 9),
                close_due=round(r.close_due, 9),
                command_time=_r9(r.command_time),
                command_event_index=r.command_event_index,
                closed_time=_r9(r.closed_time),
                closed_event_index=r.closed_event_index,
                already_closed_at_entry=r.already_closed_at_entry,
                command_overdue=round(r.command_overdue, 9),
                close_overdue=round(r.close_overdue, 9),
            )
            for r in result.risk_intervals
        ],
        close_commands=[
            InterlockCommandRecord(
                time=round(c.time, 9), event_index=c.event_index
            )
            for c in result.close_commands
        ],
        shutter_timeline=[
            InterlockShutterSegment(
                start=round(s.start, 9),
                end=_r9(s.end),
                state=s.state,  # type: ignore[arg-type]
                event_index=s.event_index,
            )
            for s in result.shutter_timeline
        ],
        horizon=round(result.horizon, 9),
    )
