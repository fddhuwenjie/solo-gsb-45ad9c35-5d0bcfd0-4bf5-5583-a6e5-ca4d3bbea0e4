# 天文台圆顶狭缝跟踪 API（Dome Slit Tracker）

使用 Python + FastAPI 实现：给定一条时间序列的**目标方位角**、圆顶**当前未展开
角度**、**最大转速**、**狭缝角宽**与**线缆角度区间**，计算圆顶在每个时刻应处
的未展开角度，使得：

1. **狭缝可见**：目标（允许跨 0°/360° 边界做整数圈平移）始终位于狭缝窗口内；
2. **转速约束**：相邻时刻转角不超过 `max_speed · Δt`；
3. **线缆边界**：圆顶角度始终位于 `[cable_low, cable_high]`；
4. **最优**：最小化累计转角 `Σ|d_t − d_{t−1}|`；
5. **平局规则**：累计转角相同时，返回**字典序最小**的角度序列
   （先比第 1 个时刻，再比第 2 个……）。

有解时返回完整动作时间线；无解时返回**最早失效时刻**与该时刻**未满足的约束**
（`slit_visibility` / `speed_limit` / `cable_limit`）及诊断细节。

## 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听在宿主机 **8000** 端口（容器内同样为 8000）。

- 健康检查：<http://localhost:8000/health>
- Swagger 交互文档：<http://localhost:8000/docs>
- OpenAPI Schema：<http://localhost:8000/openapi.json>

停止：`docker compose down`。

## 接口

### GET `/health`

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"dome-slit-tracker"}
```

### POST `/api/v1/track`

**请求字段**（角度单位：度；时间单位：秒）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `target_azimuths` | number[] | 各时刻目标方位角（长度 1..5000，与 `times` 等长） |
| `times` | number[] | 各目标的绝对时刻（秒，严格递增，首项 > 0） |
| `initial_angle` | number | 圆顶当前未展开角度（可取任意实数，须在线缆区间内） |
| `max_speed` | number | 最大转速（度/秒，> 0） |
| `slit_width` | number | 狭缝角宽（度，(0, 360]；以圆顶角为中心的对称窗口） |
| `cable_low` / `cable_high` | number | 线缆允许的未展开角度区间（度） |

**示例请求**：

```bash
curl -X POST http://localhost:8000/api/v1/track \
  -H 'Content-Type: application/json' \
  -d @examples/track_request.json
```

其中 `examples/track_request.json`：

```json
{
  "target_azimuths": [0.0, 350.0, 340.0],
  "times": [1.0, 2.0, 3.0],
  "initial_angle": 0.0,
  "max_speed": 20.0,
  "slit_width": 10.0,
  "cable_low": -180.0,
  "cable_high": 180.0
}
```

**有解响应**（`feasible: true`，返回动作时间线）：

```json
{
  "feasible": true,
  "total_rotation": 15.0,
  "actions": [
    {
      "index": 1,
      "time": 1.0,
      "dome_angle": -5.0,
      "target_azimuth": 0.0,
      "lift": 0,
      "move_delta": -5.0,
      "move_speed": -5.0,
      "cable_low": -180.0,
      "cable_high": 180.0
    },
    {
      "index": 2,
      "time": 2.0,
      "dome_angle": -15.0,
      "target_azimuth": 350.0,
      "lift": -1,
      "move_delta": -10.0,
      "move_speed": -10.0,
      "cable_low": -180.0,
      "cable_high": 180.0
    },
    {
      "index": 3,
      "time": 3.0,
      "dome_angle": -15.0,
      "target_azimuth": 340.0,
      "lift": -1,
      "move_delta": 0.0,
      "move_speed": 0.0,
      "cable_low": -180.0,
      "cable_high": 180.0
    }
  ],
  "failure": null
}
```

说明：

- `dome_angle` 为该时刻选定的圆顶**未展开角度**；
- `lift` 为目标所在的整数圈，目标提升后方位为
  `target_azimuth + 360 · lift`（例如 350° 取 `lift = -1` 即 −10°，圆顶可向
  负方向小角度移动而非正转 350°）；
- `move_delta` / `move_speed` 为相对上一时刻的带符号转角 / 转速。

**无解响应**（`feasible: false`）：

```bash
curl -X POST http://localhost:8000/api/v1/track \
  -H 'Content-Type: application/json' \
  -d @examples/track_request_infeasible.json
```

```json
{
  "feasible": false,
  "total_rotation": 0.0,
  "actions": [],
  "failure": {
    "index": 2,
    "time": 2.0,
    "target_azimuth": 300.0,
    "unmet_constraints": ["speed_limit"],
    "details": {
      "candidate_windows": [{ "lift": -1, "low": -65.0, "high": -55.0 }],
      "reason": "从可达角度集合到最近狭缝窗口的最小间隙超过本步最大转角",
      "time_step_seconds": 1.0,
      "max_rotation_this_step": 10.0,
      "required_extra_rotation": 50.0,
      "required_min_speed_deg_per_s": 60.0,
      "reachable_set": [[-5.0, 20.0]],
      "nearest_window": { "lift": -1, "low": -65.0, "high": -55.0 }
    }
  }
}
```

`unmet_constraints` 取值：

- `speed_limit`：本步从可达角度集合到最近狭缝窗口的间隙超过最大转角，
  `details` 给出所需额外转角与最低转速；
- `cable_limit`：初始角度越界，或目标狭缝窗口在任意整数圈下都不与线缆区间
  相交（后者同时标注 `slit_visibility`）。

### POST `/api/v1/calibrate-sync`

用两组**各自按时间严格递增**的样本估计圆顶编码器相对参考方位的**响应时延**
与**零位偏差**：

- 参考样本（`reference_times` / `reference_angles`）；
- 圆顶编码器样本（`encoder_times` / `encoder_angles`，可与参考异步采样）。

模型：**正时延**表示编码器在时刻 `t` 的读数对应参考序列的 `t - lag` 时刻
（编码器滞后）；零位偏差定义为 `bias = reference - encoder`，即
`encoder + bias ≈ reference`。

**请求字段**（角度单位：度；时间单位：秒）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `reference_times` | number[] | 参考样本时刻（严格递增，长度 ≥ 2） |
| `reference_angles` | number[] | 参考方位角（与 `reference_times` 等长） |
| `encoder_times` | number[] | 编码器样本时刻（严格递增，长度 ≥ 2） |
| `encoder_angles` | number[] | 编码器方位角（与 `encoder_times` 等长） |
| `lag_min` / `lag_max` | number | 时延搜索区间（秒） |
| `lag_step` | number | 时延枚举步长（秒，> 0；候选数上限 100000） |
| `outlier_threshold` | number | 离群点阈值（度，> 0） |
| `min_inliers` | integer | 有效候选所需的最少内点数（≥ 1） |

处理流程：

1. 两组角度分别以首样本为锚点，相邻角度按**最短弧**展开；最短弧恰为
   **180° 的跳变方向有歧义，直接拒绝**（HTTP 422）；
2. 对每个候选 `lag = lag_min + k·lag_step`，把编码器时刻平移为
   `t - lag`，在**重叠样本**上对参考序列线性插值，
   残差 = 插值参考角 − 展开编码器角；
3. 以残差**中位数**初估偏差，残差偏离初估超过 `outlier_threshold` 的样本
   判为离群点；再以**内点均值**重估偏差，并在内点集上计算 RMSE；
4. 内点数不足 `min_inliers` 的候选无效；
5. 选择 **RMSE 最小**的候选，并列时取 `|lag|` 较小者，再并列取 `lag`
   较小者。

**示例请求**：

```bash
curl -X POST http://localhost:8000/api/v1/calibrate-sync \
  -H 'Content-Type: application/json' \
  -d @examples/calibrate_request.json
```

**有有效候选时的响应**（偏差归一化到 `[-180, 180)`）：

```json
{
  "calibrated": true,
  "bias_deg": 29.807763,
  "lag_seconds": 0.5,
  "rmse_deg": 0.21044,
  "inlier_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "outlier_indices": [],
  "reason": null,
  "details": {
    "evaluated_candidates": 41,
    "max_overlap_samples": 15,
    "max_inliers": 15,
    "min_inliers": 10,
    "outlier_threshold": 2.0
  }
}
```

`inlier_indices` / `outlier_indices` 为样本在**编码器数组中的 0 基原始
索引**（平移后落在参考时间区间之外的非重叠样本不出现在任一列表中）。

**无有效候选时**返回 HTTP 200、`calibrated: false`，并在 `reason` 说明
原因（搜索区间为空 / 所有候选均无重叠样本 / 内点数不足），`details` 给出
枚举候选数、最大重叠样本数与最大内点数：

```json
{
  "calibrated": false,
  "bias_deg": 0.0,
  "lag_seconds": 0.0,
  "rmse_deg": 0.0,
  "inlier_indices": [],
  "outlier_indices": [],
  "reason": "内点不足：41 个候选的内点数均小于 min_inliers……",
  "details": { "evaluated_candidates": 41, "max_overlap_samples": 9, "...": "..." }
}
```

### POST `/api/v1/audit-safety-interlock`

按**分段常值**重放安全联锁事件流，推导天气风险状态时间线，并审计每段
连续风险区间上的**关闭义务**履行情况（关顶命令与快门 `closed` 确认是否
在时限内完成）。

**事件类型**（`events` 数组按 `kind` 判别，时刻必须**非递减**，同刻事件
严格按输入顺序处理）：

| `kind` | 附加字段 | 说明 |
| --- | --- | --- |
| `wind` | `value` | 风速更新（≥ 0），两次更新之间分段常值保持 |
| `rain` | `value` | 雨量更新（≥ 0），同上 |
| `close_command` | — | 关顶命令（人工或自动发出） |
| `shutter` | `state` | 快门状态 `open → closing → closed → opening → open`；允许停留原态或沿循环前进一步，**全程无快门事件（缺初值）或跳态均拒绝（HTTP 422）** |

**请求字段**（时间单位：秒）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `events` | object[] | 事件流（长度 1..5000，`time` 非递减） |
| `wind_trip` / `rain_trip` | number | 风速 / 雨量触发阈值（≥ 0） |
| `wind_reset` / `rain_reset` | number | 风速 / 雨量复位阈值（≥ 0 且 ≤ 各自触发阈值，构成迟滞） |
| `trip_duration` | number | 越限需连续保持的时长（≥ 0；0 表示当刻触发） |
| `reset_duration` | number | 复位条件需连续保持的时长（≥ 0） |
| `stale_after` | number | 距最近一次天气更新**严格超过**该时长即判中断（> 0） |
| `command_deadline` | number | 进入风险后发出关顶命令的时限（≥ 0） |
| `close_deadline` | number | 进入风险后确认快门 `closed` 的时限（≥ 0） |

天气状态机：

- 初始（收到首个天气更新前）为 `unknown`；
- 风速 ≥ `wind_trip` **或** 雨量 ≥ `rain_trip`，连续保持 `trip_duration`
  → `unsafe`；
- 风速 < `wind_reset` **且** 雨量 < `rain_reset`（两通道都已上报）连续保持
  `reset_duration` → `safe`；
- 距最近一次天气更新严格超过 `stale_after` → `unknown`，来新更新立即回到
  `safe` 并重新计时。

进入 `unsafe` / `unknown` 即产生关闭义务：须在 `command_deadline` 内发出
关顶命令、在 `close_deadline` 内确认 `closed`。义务时限在进入时刻钉死，
之后天气恢复或重复命令都不会取消或延后；`unsafe` 与 `unknown` 之间若无
`safe` 间隔，视为同一段连续风险（共享最早进入时刻的义务）。末个事件之后
仍挂起的计时按发生时刻**投影**到时间线（`projected: true`），开口义务的
超期量截算到 `horizon`（末个事件时刻）。

**示例请求**：

```bash
curl -X POST http://localhost:8000/api/v1/audit-safety-interlock \
  -H 'Content-Type: application/json' \
  -d @examples/audit_interlock_request.json
```

**响应**（HTTP 200）：`status_timeline` 为天气状态时间线（半开区间段，
`cause_event_indices` 为触发事件的 0 基原始索引）；`risk_intervals` 为
极大连续非安全区间及其关闭义务审计（`command_due` / `close_due` 为义务
时限时刻，`command_overdue` / `close_overdue` 为超期量秒数，未履行时截算
到区间末 / `horizon`）；`close_commands` 与 `shutter_timeline` 附带原始
事件索引（`event_index`，同刻多次快门转换保留零长段以反映输入顺序）：

```json
{
  "status_timeline": [
    { "start": 0.0, "end": 1.0, "state": "unknown", "cause_event_indices": [], "projected": false },
    { "start": 1.0, "end": 8.0, "state": "safe", "cause_event_indices": [1], "projected": false },
    { "start": 8.0, "end": 25.0, "state": "unsafe", "cause_event_indices": [3, 2], "projected": false },
    { "start": 25.0, "end": 50.0, "state": "safe", "cause_event_indices": [7, 2], "projected": false },
    { "start": 50.0, "end": null, "state": "unknown", "cause_event_indices": [7, 2], "projected": true }
  ],
  "risk_intervals": [
    {
      "start": 8.0, "end": 25.0, "states": ["unsafe"],
      "entry_event_indices": [3, 2], "projected": false,
      "command_due": 68.0, "close_due": 128.0,
      "command_time": 10.0, "command_event_index": 4,
      "closed_time": 14.0, "closed_event_index": 6,
      "already_closed_at_entry": false,
      "command_overdue": 0.0, "close_overdue": 0.0
    },
    {
      "start": 50.0, "end": null, "states": ["unknown"],
      "entry_event_indices": [7, 2], "projected": true,
      "command_due": 110.0, "close_due": 170.0,
      "command_time": null, "command_event_index": null,
      "closed_time": null, "closed_event_index": null,
      "already_closed_at_entry": false,
      "command_overdue": 0.0, "close_overdue": 0.0
    }
  ],
  "close_commands": [{ "time": 10.0, "event_index": 4 }],
  "shutter_timeline": [
    { "start": 0.0, "end": 12.0, "state": "open", "event_index": 0 },
    { "start": 12.0, "end": 14.0, "state": "closing", "event_index": 5 },
    { "start": 14.0, "end": 30.0, "state": "closed", "event_index": 6 },
    { "start": 30.0, "end": 32.0, "state": "opening", "event_index": 8 },
    { "start": 32.0, "end": null, "state": "open", "event_index": 9 }
  ],
  "horizon": 32.0
}
```

（上例中初始 `unknown` 段 `[0, 1)` 同样产生一段风险区间及其义务，为
简洁从响应示例中省略。）

**HTTP 422**：事件流为空、时刻递减、阈值/时长非法（如复位阈值大于触发
阈值、`stale_after <= 0`）、未知事件类型、快门缺初值（全程无 `shutter`
事件）或跳态（未沿 `open → closing → closed → opening` 循环逐步转换）。

## 算法说明

角度使用**未展开（unwrapped）**表示，因此跨 0°/360° 的跟踪不需要特判：每个
时刻的可行域是若干闭区间（目标在各整数圈 `lift` 下的狭缝窗口
`[az + 360·lift − w/2, az + 360·lift + w/2]` 与线缆区间求交）。

求解器（`app/planner/`）是一个**精确的分段线性动态规划**，不做角度网格离散：

- 价值函数（从当前位置到序列末尾的最小剩余转角）对**单个连通前驱分量**
  始终连续且每段斜率属于 `{-1, 0, 1}`，用断点 + 斜率精确表示；当前一时刻
  存在多个互不连通的 lift 分量时，完整价值函数是各分量结果**逐点取 min**
  的函数（在分量恰好进入可达半径的位置可能向下跳变），因此每个时刻以
  若干 `(区间, PL)` 覆盖的形式表示，跳变点处保持分裂、绝不桥接；
- 时间转移 `g(y) = min_x |y−x| + f(x)`（受可达半径 `R = max_speed·Δt` 限制）
  通过结构固定的 4 类仿射候选片取**下包络**得到（段内截断片、左/右可达边界
  片、端点锥），见 `piecewise.py`；
- 自末时刻向前递推后，再从初始角正向重建，每一步在所有覆盖上取使
  `|y−start| + W(y)` 最小的**最左点**，从而唯一确定累计转角最小、且字典序最小
  的序列；
- 另有一次纯区间前向传播用于在无解时定位**最早失效时刻**与未满足约束。

5000 个时刻的序列在普通机器上约 2~3 秒内求解完成。

## 本地开发（不使用 Docker）

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

运行测试（含对网格暴力 DP 的随机对拍，验证最优性与字典序）：

```bash
pip install pytest httpx
pytest -q
python3 scripts/fuzz_check.py
```

## 项目结构

```
app/
  main.py              # FastAPI 入口：/health、/api/v1/track、/api/v1/calibrate-sync、/api/v1/audit-safety-interlock
  models.py            # Pydantic 请求/响应模型与参数校验
  planner/
    geometry.py        # 方位归一化、狭缝窗口（lift）、线缆裁剪、区间合并
    piecewise.py       # 分段线性函数、距离转移算子、下包络
    solver.py          # 前向诊断 + 反向值函数 DP + 字典序路径重建
    calibration.py     # 最短弧展开 + 时延枚举/离群剔除/RMSE 同步标定
    interlock.py       # 安全联锁审计：分段常值重放、天气状态机、关闭义务
examples/              # 示例请求（跟踪有解 / 无解、同步标定、联锁审计）
scripts/fuzz_check.py  # 与网格暴力 DP 的随机对拍
tests/                 # 几何、求解器、标定算法、联锁审计、HTTP API 测试
Dockerfile
docker-compose.yml
```
