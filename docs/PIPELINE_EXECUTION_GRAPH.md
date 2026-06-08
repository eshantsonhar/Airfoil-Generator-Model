# Pipeline Execution Graph — Complete Integration Audit

## EXECUTION CHAIN (Current State)

### UI → Optimizer → Candidate Generation
```
ui/app.py: start_job()
  → subprocess: scripts/run_optimization.py
    → pipeline.py: ASOPipeline.run()
      → pipeline.py: _candidate_for()  [HARDCODED - NOT USING OPTIMIZER]
        → Uses fixed formula, not MMA
      → cfd/su2.py: SU2Evaluator.run_evaluation()
```

### Candidate → Geometry → Validation
```
_candidate_for() returns CSTParameters
  → cfd/su2.py: CSTAirfoil.full_coordinates()
  → geometry/cst.py: coordinates() → surface_y()
  → geometry/validation.py: AirfoilGeometryValidator  [CREATED BUT NEVER CALLED]
  → NO GEOMETRY VALIDATION BEFORE CFD
```

### Geometry → Mesh → CFD
```
su2.py: _write_airfoil_dat()
  → su2.py: _write_gmsh_geo()
    → cfd/mesh.py: build_geo_script()
  → su2.py: _run_gmsh()  [NO MESH QUALITY CHECK]
    → subprocess.run(gmsh, capture_output=True)
  → su2.py: _write_su2_config()
    → cfd/su2_config.py: build_stage1_config()
    → STRING REPLACEMENT for transition model
```

### CFD Execution → Results Extraction
```
su2.py: _run_su2_primal()  [NO CONVERGENCE CHECK]
  → subprocess.run(SU2_CFD, capture_output=True)
  → su2.py: _read_results()  [NO VALIDATION OF NaN/Inf]
    → Reads last line of history.csv
    → verification/convergence.py  [NEVER CALLED - 558 lines unused]
  → su2.py: _run_su2_adjoint()  [SAME BINARY - not actual adjoint]
  → su2.py: _extract_adjoint_gradients()  [ALWAYS RETURNS ZEROS]
  → physics/lsb_detection.py  [NEVER CALLED - 624 lines unused]
```

### Results → Objective → Optimizer Update
```
pipeline.py: AirfoilScorer.score()  [ARBITRARY CONSTANTS]
  → optimization/scoring.py: score = w1*eff/30 + w2*stall + w3*cd
  → database.insert_result()
  → MMA OPTIMIZER NEVER UPDATED
  → pipeline.py: _candidate_for() uses raw iteration counter
```

### Telemetry → Archival → UI
```
pipeline.py: RuntimeTracker.flush()
  → writes JSON to data/logs/latest_runtime.json
  → ui/app.py: job_runtime() reads JSON file
  → ui/app.py: job_status() polls process
  → NO WEBSOCKET - HTTP polling only
  → runtime/watchdog.py  [CREATED BUT NOT INTEGRATED INTO PIPELINE]
```

## DEAD EXECUTION PATHS

| Module | Lines | Status |
|--------|-------|--------|
| `verification/convergence.py` | 558 | **NEVER IMPORTED** |
| `physics/lsb_detection.py` | 624 | **NEVER IMPORTED** |
| `physics/plausibility.py` | ? | **NEVER IMPORTED** |
| `physics/transition_governance.py` | ? | **NEVER IMPORTED** |
| `verification/cfd_governance.py` | ? | **NEVER IMPORTED** |
| `verification/gci.py` | ? | **NEVER IMPORTED** |
| `verification/mesh_verification.py` | ? | **NEVER IMPORTED** |
| `verification/numerical_dissipation.py` | ? | **NEVER IMPORTED** |
| `verification/gradient_audit.py` | ? | **NEVER IMPORTED** |
| `runtime/watchdog.py` | ? | **NOT INTEGRATED INTO PIPELINE** |
| `core/mma_solver.py` (TrustRegionMMA) | 58 | **SHADOWED by optimization/mma_engine.py** |
| `optimization/integrity_monitor.py` | ? | **NOT INTEGRATED** |
| `optimization/governed_optimizer.py` | ? | **NOT INTEGRATED** |
| `optimization/comprehensive_governance.py` | ? | **NOT INTEGRATED** |
| `optimization/objective_governance.py` | ? | **NOT INTEGRATED** |
| `optimization/conditioner.py` | ? | **INSTANTIATED BUT NOT USED** |
| `optimization/trust_region.py` | ? | **INSTANTIATED BUT NOT USED** |
| `core/archival.py` | ? | **NOT INTEGRATED** |

## FAKE DATA PATHWAYS

1. **Fake CFD Fallback**: `su2.py:200-212` — When `settings is None`, returns `cd = x·x`, `cl = sum(x)/len(x)`, `grad = 2x` — completely synthetic
2. **Zero Adjoint Gradients**: `su2.py:176-180` — `_extract_adjoint_gradients()` returns `np.zeros(10), np.zeros(10)`
3. **Hardcoded Candidates**: `pipeline.py:236-244` — `_candidate_for()` uses fixed CST parameter formula, NOT MMA optimizer
4. **Fake Thickness**: `su2.py:267` — `thickness = 0.12 - 0.02 * design_vector[2]` is an arbitrary formula not derived from geometry
5. **Fake Objective Scoring**: `optimization/scoring.py` — `_EFF_REFERENCE = 30.0` is arbitrary, mixed units (degrees + dimensionless)
6. **Dummy Adjoint Run**: `su2.py:133-145` — `_run_su2_adjoint()` runs the SAME primal binary, not a true adjoint solve
7. **Fake Gradient Auditor**: `core/monitoring.py:159-170` — Only tests 3 random dimensions with one-sided FD

## VALIDATION CHECKPOINTS (MISSING)

- [ ] Pre-CFD geometry validation (module exists, not called)
- [ ] Mesh quality verification (not implemented)
- [ ] SU2 config validation (not implemented)
- [ ] CFD convergence check (module exists, not called)
- [ ] Force plausibility check (not implemented)
- [ ] NaN/Inf detection on CFD outputs (not implemented)
- [ ] Gradient sanity check (implemented but zero-gradients pass)
- [ ] Step acceptance validation (trust-region broken)
- [ ] LSB detection (module exists, not called)
- [ ] Transition model validation (not implemented)

## STATE TRANSITIONS (CURRENT)

```
IDLE → _candidate_for() → run_evaluation() → _run_gmsh() → _run_su2() → _read_results()
  → _run_su2_adjoint() → _extract_adjoint_gradients() → score() → insert_result()
  → LOOP back to _candidate_for()
  → PROBLEM: No state machine. No failure recovery. No convergence check.
```

## NUMERICAL DEPENDENCY CHAIN (BROKEN)

```
Optimizer (SvanbergMMA) → [NEVER CONNECTED]
  ↓
Candidate (hardcoded) → CST params
  ↓
Geometry → CSTAirfoil.coordinates() → [NO VALIDATION, NO SANITY CHECK]
  ↓
Mesh → GMSH → [NO QUALITY CHECK]
  ↓
CFD → SU2 primal → [NO CONVERGENCE CHECK]
  ↓
Forces → history.csv last line → [NO NaN CHECK, NO STABILITY CHECK]
  ↓
Adjoint → ZEROS → [NO ACTUAL GRADIENT COMPUTATION]
  ↓
Objective → AirfoilScorer → ARBITRARY CONSTANTS
  ↓
Telemetry → RuntimeTracker → FILE-BASED POLLING
  ↓
UI → FastAPI → HTTP POLLING, STALE DATA
```

## CRITICAL ISSUES REQUIRING IMMEDIATE FIX

1. **CRITICAL-001**: Remove fake CFD fallback — raise ConfigurationError instead
2. **CRITICAL-002**: Implement real SU2 adjoint gradient extraction
3. **CRITICAL-004**: Connect MMA optimizer to candidate generation
4. **CRITICAL-007**: Integrate convergence analysis into CFD pipeline
5. **CRITICAL-008**: Integrate LSB detection into evaluation pipeline
6. **CRITICAL-013**: Fix trust-region mathematics
7. **HIGH-005**: Enforce geometry validation before CFD
8. **HIGH-002**: Fix watchdog subprocess termination

## RECOMMENDED REBUILD ORDER

1. Fix pipeline.py — connect optimizer, remove hardcoded candidates
2. Fix su2.py — remove fake fallback, implement real gradient parsing
3. Fix optimizer/mma_engine.py — implement real MMA mathematics
4. Integrate convergence.py into CFD evaluation
5. Integrate lsb_detection.py into evaluation pipeline
6. Integrate geometry/validation.py as pre-CFD gate
7. Rebuild scoring.py with physically meaningful objective
8. Rebuild ui/app.py with websocket streaming
9. Add force_truth_audit.py for physical plausibility
10. Add optimizer_mathematics_validation.py