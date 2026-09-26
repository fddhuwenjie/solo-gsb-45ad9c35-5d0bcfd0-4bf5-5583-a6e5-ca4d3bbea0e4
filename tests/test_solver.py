"""求解器端到端测试。

注：最优性 / 字典序的大规模验证由 scripts/fuzz_check.py 对拍保证；
本文件聚焦清晰场景、约束校验与无解诊断。
"""

import pytest

from app.planner.solver import Problem, solve


def make(azs, times, d0=0.0, speed=10.0, slit=10.0, clo=-180.0, chi=180.0):
    return Problem(azs, times, d0, speed, slit, clo, chi)


def assert_valid(sol, p):
    """断言解满足全部约束。"""
    assert sol.feasible
    prev, last_t = p.initial_angle, 0.0
    total = 0.0
    for a in sol.actions:
        R = p.max_speed * (a.time - last_t)
        assert abs(a.dome_angle - prev) <= R + 1e-7
        assert p.cable_lo - 1e-7 <= a.dome_angle <= p.cable_hi + 1e-7
        center = a.target_azimuth + 360 * a.lift
        assert abs(a.dome_angle - center) <= p.slit_width / 2 + 1e-7
        total += abs(a.dome_angle - prev)
        prev, last_t = a.dome_angle, a.time
    assert total == pytest.approx(sol.total_rotation, abs=1e-7)


def test_stationary_target_no_motion():
    sol = solve(make([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
    assert sol.feasible
    assert all(abs(a.move_delta) < 1e-8 for a in sol.actions)
    assert sol.total_rotation == pytest.approx(0.0, abs=1e-8)
    # 待在中心 0 是唯一的零代价序列
    assert all(a.dome_angle == pytest.approx(0.0, abs=1e-7)
               for a in sol.actions)


def test_tracks_moving_target():
    p = make([0.0, 10.0, 20.0], [1.0, 2.0, 3.0], speed=10.0, slit=10.0)
    sol = solve(p)
    assert_valid(sol, p)
    assert sol.total_rotation == pytest.approx(15.0, abs=1e-7)
    assert [a.dome_angle for a in sol.actions] == pytest.approx(
        [0.0, 5.0, 15.0], abs=1e-7
    )


def test_infeasible_speed():
    sol = solve(make([100.0], [1.0], d0=0.0, speed=10.0, slit=10.0))
    assert not sol.feasible
    f = sol.failure
    assert f.index == 1
    assert "speed_limit" in f.unmet
    # 从 0 到最近窗口端点 95，缺 85
    assert f.details["required_extra_rotation"] == pytest.approx(85.0, abs=1e-6)


def test_infeasible_cable_at_initial():
    sol = solve(make([0.0], [1.0], d0=100.0, speed=100.0,
                     slit=10.0, clo=-50.0, chi=50.0))
    assert not sol.feasible
    assert sol.failure.index == 0
    assert "cable_limit" in sol.failure.unmet


def test_infeasible_target_window_outside_cable():
    sol = solve(make([180.0], [1.0], d0=0.0, speed=1000.0,
                     slit=10.0, clo=-10.0, chi=10.0))
    assert not sol.feasible
    assert "cable_limit" in sol.failure.unmet


def test_wrap_around_chooses_shorter_unwrapped_path():
    sol = solve(make([350.0], [1.0], d0=0.0, speed=10.0, slit=10.0))
    assert sol.feasible
    a = sol.actions[0]
    assert a.lift == -1
    assert a.dome_angle == pytest.approx(-5.0, abs=1e-7)
    assert sol.total_rotation == pytest.approx(5.0, abs=1e-7)


def test_lexicographic_tie_prefers_smaller_first_angle():
    # 真正的平坦最优：d0=0，t1 窗口 [0,20]（目标 10、缝 20），
    # t2 窗口 [10,30]（目标 20、缝 20），速度足够大。
    # 任意 d1∈[0,10]、d2=10 都给出总转角 10；字典序最小取 d1=0。
    p = Problem(
        target_azimuths=[10.0, 20.0],
        times=[1.0, 2.0],
        initial_angle=0.0,
        max_speed=1000.0,
        slit_width=20.0,
        cable_lo=-180.0,
        cable_hi=180.0,
    )
    sol = solve(p)
    assert sol.feasible
    assert sol.actions[0].dome_angle == pytest.approx(0.0, abs=1e-7)
    assert sol.actions[1].dome_angle == pytest.approx(10.0, abs=1e-7)
    assert sol.total_rotation == pytest.approx(10.0, abs=1e-7)


def test_anticipatory_move_required():
    # t1 目标 0（窗 [-5,5]），t2 目标 50（窗 [45,55]），每步上限 40。
    # t1 必须预先走到 5（5->45 恰为 40），总转角 45。
    sol = solve(make([0.0, 50.0], [1.0, 2.0], d0=0.0,
                     speed=40.0, slit=10.0))
    assert sol.feasible
    assert sol.actions[0].dome_angle == pytest.approx(5.0, abs=1e-7)
    assert sol.actions[1].dome_angle == pytest.approx(45.0, abs=1e-7)
    assert sol.total_rotation == pytest.approx(45.0, abs=1e-7)


def test_general_instance_constraints():
    p = Problem(
        target_azimuths=[12.0, 350.0, 200.0],
        times=[1.0, 3.0, 4.0],
        initial_angle=5.0,
        max_speed=300.0,
        slit_width=8.0,
        cable_lo=-200.0,
        cable_hi=200.0,
    )
    assert_valid(solve(p), p)


def test_full_slit_width():
    p = make([0.0, 90.0], [1.0, 2.0], d0=0.0, speed=100.0, slit=360.0)
    sol = solve(p)
    assert sol.feasible
    # 360 宽缝：目标始终在缝内，无需转动
    assert sol.actions[0].dome_angle == pytest.approx(0.0, abs=1e-7)
    assert sol.actions[1].dome_angle == pytest.approx(0.0, abs=1e-7)
    assert sol.total_rotation == pytest.approx(0.0, abs=1e-7)


def test_lift_path_continuity():
    p = Problem(
        target_azimuths=[359.0, 1.0, 358.0],
        times=[1.0, 2.0, 3.0],
        initial_angle=-170.0,
        max_speed=200.0,
        slit_width=30.0,
        cable_lo=-270.0,
        cable_hi=270.0,
    )
    sol = solve(p)
    assert_valid(sol, p)
    # 最优解允许在相邻 lift 间切换（解绕卷），总转角 156
    assert sol.total_rotation == pytest.approx(156.0, abs=1e-6)
    # 每个动作的目标中心（azimuth + 360*lift）应与 dome 角差 ≤ 半宽
    for a in sol.actions:
        center = a.target_azimuth + 360 * a.lift
        assert abs(center - a.dome_angle) <= 15.0 + 1e-7


def test_earliest_failure_is_reported():
    # 第一步可行，第二步超出速度
    sol = solve(make([0.0, 100.0], [1.0, 2.0], d0=0.0,
                     speed=20.0, slit=10.0))
    assert not sol.feasible
    assert sol.failure.index == 2


def test_wide_slit_multiple_lifts_exact_optimum():
    # 精确回归：宽狭缝（200°）下每个时刻有多个整数圈候选，价值函数在
    # 不连通 lift 分量边界处发生跳变。曾因下包络跨跳变点桥接，返回累计
    # 转角 148.5 的次优路线 [14.75, 73, 73, 101.75, 53, 40.25]。
    # 最优为 121.75：第一步走到 lift -1 窗口（中心 27.75）左端 -72.25，
    # 第二步进入 lift -2 分量（中心 -187）的 -107 后保持不动。
    p = Problem(
        target_azimuths=[387.75, 533.0, -12.25, 561.75, -207.0, -59.75],
        times=[2.0, 5.0, 7.0, 11.0, 13.0, 16.0],
        initial_angle=14.75,
        max_speed=48.75,
        slit_width=200.0,
        cable_lo=-900.0,
        cable_hi=900.0,
    )
    sol = solve(p)
    assert_valid(sol, p)
    assert sol.total_rotation == pytest.approx(121.75, abs=1e-7)
    assert [a.dome_angle for a in sol.actions] == pytest.approx(
        [-72.25, -107.0, -107.0, -107.0, -107.0, -107.0], abs=1e-7
    )
    # lift 满足文档恒等式：目标提升方位 = azimuth + 360*lift
    assert [a.lift for a in sol.actions] == [-1, -2, 0, -2, 0, 0]
    for a in sol.actions:
        center = a.target_azimuth + 360 * a.lift
        assert abs(a.dome_angle - center) <= p.slit_width / 2 + 1e-7


def test_jump_component_entry_is_not_diluted():
    # 手工核算的小型跳变场景（旧实现返回 204.9375 的次优解）。
    # t1 只能到达 lift -2 窗口 [-437.4375, -418.75]；t3 的 lift -1 窗口
    # 仅从 t2 分量左端附近可达，价值函数在 t2 窗口内发生跳变。
    # 总转角 = -669.9375 - 2*t1，t1 越大越优，故 t1 = -418.75；
    # 随后字典序最小为 t2 = -380.5、t3 = -288，总转角 167.5625。
    p = Problem(
        target_azimuths=[226.25, 51.25, 147.0],
        times=[1.5, 4.5, 7.0],
        initial_angle=-381.9375,
        max_speed=37.0,
        slit_width=150.0,
        cable_lo=-900.0,
        cable_hi=900.0,
    )
    sol = solve(p)
    assert_valid(sol, p)
    assert sol.total_rotation == pytest.approx(167.5625, abs=1e-7)
    assert [a.dome_angle for a in sol.actions] == pytest.approx(
        [-418.75, -380.5, -288.0], abs=1e-7
    )
