# Final Runtime Trace — Airfoil ASO Research Platform

## Execution graph

```mermaid
flowchart TB
  UI[FastAPI ui/app.py :8000] -->|POST /api/job/start| Sub[scripts/run_optimization.py]
  Sub --> Pipe[pipeline.ASOPipeline.run]
  Pipe --> WD[SystemWatchdog]
  Pipe --> Tel[PipelineTelemetryBridge]
  Tel --> JSONL[data/logs/telemetry_events.jsonl]
  Pipe --> RT[data/logs/latest_runtime.json]
  Pipe --> Geo[geometry.validation HARD GATE]
  Pipe --> SU2[cfd.su2.SU2Evaluator]
  SU2 --> Gmsh[Gmsh mesh]
  SU2 --> Primal[SU2 primal]
  SU2 --> Adj[Adjoint extract]
  Pipe --> MMA[optimization.mma_engine.SvanbergMMA]
  Pipe --> Gov[governance.scientific_truth_policy]
  UI --> Hub[TelemetryHub file tail]
  Hub --> WS[/ws/telemetry WebSocket]
  React[frontend React :5173] --> WS
  React --> API[/api/* REST]
```

## Call hierarchy (optimization iteration)

1. `ASOPipeline.run()` — iteration loop
2. `watchdog.heartbeat("pipeline")` — stale detection
3. `telemetry.emit("optimization_iteration_start")`
4. For each AoA in `[2, 4, 6]`:
   - `run_with_timeout("cfd_eval_*", evaluator.run_evaluation)`
   - `SU2Evaluator`: geometry validate → mesh → primal → convergence → LSB → adjoint
   - On `status != OK`: archive diagnostics, `telemetry.failure`, **STOP**
5. `objective_factory.package()` — constrained Cd objective
6. Zero-gradient check → **STOP** if `||df|| < 1e-12`
7. `mma.run_optimization_step()` — Svanberg MMA + trust region
8. `governor.update(rho)` — gain-ratio trust-region
9. `telemetry.snapshot()` + `tracker.log_optimization_step()`
10. `database.insert_result()` — SQLite archive

## Validation checkpoints

| Stage | Gate | On failure |
|-------|------|------------|
| Geometry | `geometry/validation.py` | CONFIG_ERROR, stop |
| Mesh | Gmsh + file size | MESH_GENERATION, stop |
| Primal | SU2 return code + history | DIVERGED/CRASHED, stop |
| Convergence | convergence verifier | reject polar |
| Physics | LSB, transition, dissipation | credibility score |
| Adjoint | gradient audit | GRADIENT_ZERO, stop |
| MMA | trust-region collapse | stagnation stop |
| Watchdog | operation timeout | CRASHED + archived |

## Failure propagation

- All failures emit `telemetry_events.jsonl` with `event_type=failure`
- Runtime JSON `status=failed` — UI shows immediately
- Case directory copied to `data/cache/diagnostics/iter_*`
- **No synthetic Cl/Cd continuation**

## Telemetry lifecycle

1. Pipeline subprocess writes JSONL (`AIRFOIL_TELEMETRY_PATH`)
2. FastAPI `TelemetryHub` tails file every 250ms
3. WebSocket clients receive full event + replay buffer (5000 events)
4. `ResearchTelemetry` persists metrics to `data/telemetry/metrics.db`

## Watchdog timeouts (defaults)

| Operation | Timeout |
|-----------|---------|
| SU2 CFD | 1800 s |
| Gmsh mesh | 300 s |
| Adjoint | 3600 s |
| Full optimization | 7200 s |
| Heartbeat stale | 3× interval |

## Instrumentation fields

`latest_runtime.json`: objective, gradient, trust radius, Cl/Cd histories, gain ratio, watchdog status, heartbeats, running cases, ETA.

JSONL events: `optimization_start`, `cfd_start`, `cfd_complete`, `mma_step`, `telemetry_snapshot`, `heartbeat`, `watchdog`, `failure`.
