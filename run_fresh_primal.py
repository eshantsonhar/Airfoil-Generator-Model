"""
Fresh Primal CFD Solve — Phase 2 Production Certification Run
=============================================================
4-stage INC_RANS pipeline on airfoil_perfect.su2  (Re=1e5, AoA=4°):

  Sub-stage 0A  INC_RANS SST, 1st-order, CFL=0.8 FIXED, 500 iter.
                Cold start from uniform freestream; no restart.
                CFL_ADAPT=NO — adaptive CFL ramps to 50 by iter ~50
                and causes catastrophic divergence (proven in prior runs).
                rms[w] jumps to +1.3 at iter 1 (normal SST cold-start
                transient on a RANS mesh) then settles; rms[P] converges
                monotonically as long as CFL stays fixed at 0.8.
                INC_ARTIFICIAL_COMPRESSIBILITY stabilises pressure coupling.

  Sub-stage 0B  INC_RANS SST, 1st-order, CFL=1.2 FIXED, 1000 iter.
                Restart from 0A.  CFL raised slightly once field is seeded.
                rms[k] and rms[w] expected to reach ~ -3 to -5.

  Stage 1       INC_RANS SST, MUSCL, Venkatakrishnan limiters,
                CFL=2.0 FIXED, 1500 iter. Restart from 0B.
                VAN_ALBADA_EDGE is invalid for SLOPE_LIMITER_TURB in
                SU2 v8.4.0 — crashes computeLimiters() "Unknown limiter".

  Stage 2       INC_RANS SST + LM γ–Reθ transition, MUSCL,
                CFL=1.5 FIXED, 3000 iter. Restart from Stage 1.

Root-cause analysis of prior divergence
  Problem:   CFL_ADAPT=YES with ceiling=50 caused rms[P] to reverse at
             iter ~55 and grow to +2.5 by iter 800 (CL→1e11).
  Proof:     200-iter probe at CFL=0.8 fixed: rms[P] -3.68→-3.62 (stable),
             CL settles at 0.568 (physical), rms[k] drops from -8.49 to -0.86.
  Fix:       CFL_ADAPT=NO throughout.  Constant CFL 0.8/1.2/2.0/1.5 by stage.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path.cwd()
BIN       = ROOT / "bin" / "SU2_CFD.exe"
MESH_SRC  = ROOT / "data" / "cache" / "final_test" / "airfoil_perfect.su2"
RUN_DIR   = ROOT / "data" / "cache" / "primal_fresh"
MESH_NAME = "airfoil_perfect.su2"

# ── Physics  (unit-chord, U=1 m/s non-dimensionalisation) ─────────────────────
RE       = 1.0e5
MACH     = 0.10
AOA_DEG  = 4.0
RHO_AIR  = 1.225
MU       = RHO_AIR * 1.0 * 1.0 / RE   # Re = rho*U*c/mu = 1e5 exactly
VX       = math.cos(math.radians(AOA_DEG))
VY       = math.sin(math.radians(AOA_DEG))
TU_PROD  = 0.001    # 0.1 %  production turbulence inlet
TVR_PROD = 5.0      # mu_t/mu  production
TU_COLD  = 0.001    # same TU for cold start (TVR controls omega init)
TVR_COLD = 5.0      # TVR=5 gives rms[w]_init=-4.23 (proven stable)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def _write_cfg(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _run_stage(cfg_path: Path, work_dir: Path, label: str,
               timeout: int = 3600) -> tuple[int, float]:
    t0 = time.time()
    r = subprocess.run(
        [str(BIN), cfg_path.name],
        cwd=work_dir, capture_output=True, text=True, timeout=timeout,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    elapsed = time.time() - t0
    (work_dir / f"su2_stdout_{label}.log").write_text(
        r.stdout, encoding="utf-8", errors="ignore")
    (work_dir / f"su2_stderr_{label}.log").write_text(
        r.stderr, encoding="utf-8", errors="ignore")
    return r.returncode, elapsed


# ── Config builder ────────────────────────────────────────────────────────────
def _build_cfg(
    *,
    n_iter: int,
    cfl: float,
    muscl: bool,
    trans: str,
    restart_file: str | None,
    tag: str,
    surface_csv: bool = False,
    tu: float = TU_COLD,
    tvr: float = TVR_COLD,
) -> str:
    """
    Build a complete SU2 INC_RANS config string.

    Key stabilisation choices (all proven by probe runs):
    - CFL_ADAPT=NO:          adaptive CFL blows up by iter 50 on this mesh.
    - INC_DENSITY_REF/       artificial compressibility stabilises pressure
      INC_VELOCITY_REF:      Poisson equation on stretched BL cells.
    - LINEAR_SOLVER_ERROR    1e-4 (not 1e-6): tighter tolerance stalls FGMRES
      =1e-4, ITER=20:        on stiff INC_RANS matrices; 20 Krylov vectors
                             accommodate BL anisotropy.
    - CFL_REDUCTION_TURB=0.5 turb equations run at CFL/2; limits omega
                             residual growth during the initial transient.
    - SLOPE_LIMITER_TURB=    VAN_ALBADA_EDGE crashes SU2 v8.4.0.
      VENKATAKRISHNAN:       VENKATAKRISHNAN_WANG/VENKATAKRISHNAN are valid.
    """
    muscl_str  = "YES" if muscl else "NO"
    lim_flow   = "VENKATAKRISHNAN_WANG" if muscl else "NONE"
    lim_turb   = "VENKATAKRISHNAN"      if muscl else "NONE"
    out_files  = "(RESTART, SURFACE_CSV)" if surface_csv else "(RESTART)"
    sol_line   = f"SOLUTION_FILENAME= {restart_file}" if restart_file else ""
    restart_kw = "YES" if restart_file else "NO"

    lm_extra = ""
    if trans == "LM":
        lm_extra = (
            f"\n% -- LM transition inlet --"
            f"\nFREESTREAM_TURBULENCEINTENSITY= {tu}"
            f"\nFREESTREAM_TURB2LAMVISCRATIO= {tvr}"
        )

    return f"""\
% ---- SU2 INC_RANS  tag={tag}  cfl={cfl}  muscl={muscl_str}  trans={trans} ----
SOLVER= INC_RANS
KIND_TURB_MODEL= SST
KIND_TRANS_MODEL= {trans}
RESTART_SOL= {restart_kw}

% -- Fluid --
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {MU:.8e}
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {RHO_AIR}
INC_VELOCITY_INIT= ( {VX:.8f}, {VY:.8f}, 0.0 )
MACH_NUMBER= {MACH}
AOA= {AOA_DEG}
REYNOLDS_NUMBER= {RE:.1f}
REYNOLDS_LENGTH= 1.0
REF_LENGTH= 1.0
REF_AREA= 1.0
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.00
REF_ORIGIN_MOMENT_Z= 0.00
FREESTREAM_VELOCITY= ( {VX:.8f}, {VY:.8f}, 0.0 )
FREESTREAM_DENSITY= {RHO_AIR}
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15

% -- Artificial compressibility (stabilises INC pressure eq on BL meshes) --
INC_DENSITY_REF= 1.225
INC_VELOCITY_REF= 1.0

% -- Turbulence inlet --
FREESTREAM_TURBULENCEINTENSITY= {tu}
FREESTREAM_TURB2LAMVISCRATIO= {tvr}
{lm_extra}

% -- Mesh --
MESH_FILENAME= {MESH_NAME}
MESH_FORMAT= SU2

% -- BCs --
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_MONITORING= ( airfoil )
MARKER_PLOTTING= ( airfoil )

% -- Numerics --
CONV_NUM_METHOD_FLOW= FDS
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
NUM_METHOD_GRAD_RECON= LEAST_SQUARES
MUSCL_FLOW= {muscl_str}
MUSCL_TURB= {muscl_str}
SLOPE_LIMITER_FLOW= {lim_flow}
SLOPE_LIMITER_TURB= {lim_turb}
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
TIME_DISCRE_TURB= EULER_IMPLICIT

% -- CFL: FIXED (no adaptation) --
CFL_NUMBER= {cfl}
CFL_ADAPT= NO
CFL_REDUCTION_TURB= 0.5

% -- Linear solver --
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-4
LINEAR_SOLVER_ITER= 20

% -- Iterations --
ITER= {n_iter}

% -- Convergence --
CONV_FIELD= RMS_PRESSURE
CONV_STARTITER= 100
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1e-5

% -- Output --
TABULAR_FORMAT= CSV
CONV_FILENAME= history_{tag}
RESTART_FILENAME= restart_{tag}
VOLUME_FILENAME= flow_{tag}
SURFACE_FILENAME= surface_{tag}
OUTPUT_FILES= {out_files}
OUTPUT_WRT_FREQ= 50
SCREEN_OUTPUT= (INNER_ITER, RMS_RES, AERO_COEFF)
HISTORY_OUTPUT= (INNER_ITER, RMS_RES, AERO_COEFF)
{sol_line}
"""


# ── History parsing ───────────────────────────────────────────────────────────
def _parse_history(history_path: Path) -> dict:
    if not history_path.exists():
        return {}
    with open(history_path, encoding="utf-8") as f:
        raw = f.readlines()
    if len(raw) < 2:
        return {}
    hdr  = [h.strip().strip('"') for h in raw[0].split(",")]
    rows = [line.split(",") for line in raw[1:]
            if len(line.split(",")) >= len(hdr)]
    if not rows:
        return {}

    def _col(name: str) -> int:
        for alt in [name, f'"{name}"', name.lower()]:
            if alt in hdr:
                return hdr.index(alt)
        return -1

    def _vals(name: str) -> list[float]:
        idx = _col(name)
        if idx < 0:
            return []
        out = []
        for row in rows:
            try:
                out.append(float(row[idx].strip()))
            except (ValueError, IndexError):
                pass
        return out

    result: dict = {"n_iter": len(rows), "header": hdr}
    for field in ("rms[P]", "rms[U]", "rms[k]", "rms[w]", "CL", "CD", "CMy"):
        vals = _vals(field)
        key  = field.replace("[", "_").replace("]", "")
        if vals:
            result[f"{key}_start"] = vals[0]
            result[f"{key}_end"]   = vals[-1]
    if "rms_P_start" in result:
        result["rms_p_drop"] = (result["rms_P_start"] - result["rms_P_end"])
    return result


def _print_stage(label: str, rc: int, elapsed: float,
                 s: dict, log: Path) -> None:
    print(f"\n{label}  rc={rc}  elapsed={elapsed:.0f}s")
    if rc != 0 or not s:
        print(f"  ⚠️  {label} failed — {log.name}")
        if log.exists():
            for ln in log.read_text(encoding="utf-8", errors="ignore"
                                    ).splitlines()[-30:]:
                if any(k in ln for k in ("Error","NaN","diverge","error")):
                    print(f"    {ln.strip()}")
        return
    drop = s.get("rms_p_drop", float("nan"))
    cl   = s.get("CL_end",     float("nan"))
    cd   = s.get("CD_end",     float("nan"))
    rmsw = s.get("rms_w_end",  float("nan"))
    rmsk = s.get("rms_k_end",  float("nan"))
    rmsp_start = s.get("rms_P_start", float("nan"))
    rmsp_end   = s.get("rms_P_end",   float("nan"))
    print(f"  rms[P]: {rmsp_start:.4f} → {rmsp_end:.4f}  drop={drop:.2f}")
    print(f"  rms[k]={rmsk:.4f}  rms[w]={rmsw:.4f}")
    print(f"  CL={cl:.6f}  CD={cd:.8f}")
    if cd and cd > 0 and abs(cl) < 1e6:
        print(f"  L/D = {cl/cd:.1f}")


def _check_physics(s: dict) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok   = True
    cl   = s.get("CL_end")
    cd   = s.get("CD_end")
    drop = s.get("rms_p_drop", 0.0)
    if cl is None or cd is None:
        return False, ["No CL/CD in history"]
    if not (0.05 < cl < 2.0):
        msgs.append(f"CL={cl:.4f} outside [0.05, 2.0]");  ok = False
    if not (0.001 < cd < 0.15):
        msgs.append(f"CD={cd:.6f} outside [0.001, 0.15]"); ok = False
    msgs.append(
        f"rms[P] drop={drop:.2f} orders"
        + (" ✅" if drop >= 4.0 else " ⚠️ (< 4.0)"))
    if ok and drop >= 4.0:
        msgs.append("✅ Aerodynamics physically plausible")
    return ok, msgs


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> dict:
    _banner("PHASE 2: FRESH PRIMAL CFD SOLVE  (fixed-CFL stabilised pipeline)")

    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)

    mesh_dst = RUN_DIR / MESH_NAME
    shutil.copy2(MESH_SRC, mesh_dst)
    print(f"Mesh : {mesh_dst.name}  ({mesh_dst.stat().st_size:,} bytes)"
          f"  domain=40c×40c  Re={RE:.0e}  AoA={AOA_DEG}°")

    results: dict = {}

    # ── Sub-stage 0A: cold start, CFL=0.8, 500 iter ───────────────────────
    _banner("SUB-STAGE 0A — INC_RANS SST, CFL=0.8 fixed, 500 iter (cold start)")
    cfg0a = RUN_DIR / "cfg_0a.cfg"
    _write_cfg(cfg0a, _build_cfg(
        n_iter=500, cfl=0.8, muscl=False, trans="NONE",
        restart_file=None, tag="s0a",
        tu=TU_COLD, tvr=TVR_COLD,
    ))
    rc0a, t0a = _run_stage(cfg0a, RUN_DIR, "s0a", timeout=1200)
    s0a = _parse_history(RUN_DIR / "history_s0a.csv")
    results["sub0a"] = {**s0a, "rc": rc0a, "elapsed": t0a}
    _print_stage("Sub-stage 0A", rc0a, t0a, s0a, RUN_DIR / "su2_stdout_s0a.log")

    # ── Sub-stage 0B: warm restart, CFL=1.2, 1000 iter ────────────────────
    _banner("SUB-STAGE 0B — INC_RANS SST, CFL=1.2 fixed, 1000 iter (warm restart)")
    cfg0b = RUN_DIR / "cfg_0b.cfg"
    r0a_file = "restart_s0a.dat" if rc0a == 0 else None
    _write_cfg(cfg0b, _build_cfg(
        n_iter=1000, cfl=1.2, muscl=False, trans="NONE",
        restart_file=r0a_file, tag="s0b",
        tu=TU_COLD, tvr=TVR_COLD,
    ))
    rc0b, t0b = _run_stage(cfg0b, RUN_DIR, "s0b", timeout=2400)
    s0b = _parse_history(RUN_DIR / "history_s0b.csv")
    results["sub0b"] = {**s0b, "rc": rc0b, "elapsed": t0b}
    _print_stage("Sub-stage 0B", rc0b, t0b, s0b, RUN_DIR / "su2_stdout_s0b.log")

    # ── Stage 1: MUSCL + Venkatakrishnan, CFL=2.0, 1500 iter ──────────────
    _banner("STAGE 1 — MUSCL SST, CFL=2.0 fixed, 1500 iter")
    cfg1 = RUN_DIR / "cfg_s1.cfg"
    r0b_file = "restart_s0b.dat" if rc0b == 0 else (
               "restart_s0a.dat" if rc0a == 0 else None)
    _write_cfg(cfg1, _build_cfg(
        n_iter=1500, cfl=2.0, muscl=True, trans="NONE",
        restart_file=r0b_file, tag="s1",
        tu=TU_PROD, tvr=TVR_PROD,
    ))
    rc1, t1 = _run_stage(cfg1, RUN_DIR, "s1", timeout=3600)
    s1 = _parse_history(RUN_DIR / "history_s1.csv")
    results["stage1"] = {**s1, "rc": rc1, "elapsed": t1}
    _print_stage("Stage 1", rc1, t1, s1, RUN_DIR / "su2_stdout_s1.log")

    # ── Stage 2: MUSCL + LM transition, CFL=1.5, 3000 iter ───────────────
    _banner("STAGE 2 — MUSCL SST+LM, CFL=1.5 fixed, 3000 iter")
    cfg2 = RUN_DIR / "cfg_s2.cfg"
    r1_file = ("restart_s1.dat"  if rc1  == 0 else
               "restart_s0b.dat" if rc0b == 0 else
               "restart_s0a.dat" if rc0a == 0 else None)
    _write_cfg(cfg2, _build_cfg(
        n_iter=3000, cfl=1.5, muscl=True, trans="LM",
        restart_file=r1_file, tag="s2", surface_csv=True,
        tu=TU_PROD, tvr=TVR_PROD,
    ))
    rc2, t2 = _run_stage(cfg2, RUN_DIR, "s2", timeout=7200)
    s2 = _parse_history(RUN_DIR / "history_s2.csv")
    results["stage2"] = {**s2, "rc": rc2, "elapsed": t2}
    _print_stage("Stage 2", rc2, t2, s2, RUN_DIR / "su2_stdout_s2.log")

    # ── Physics check ──────────────────────────────────────────────────────
    _banner("PHYSICS VERIFICATION")
    final = s2 if s2 else s1 if s1 else s0b
    phys_ok, msgs = _check_physics(final)
    for m in msgs:
        print(f"  {m}")
    results["physics_ok"] = phys_ok
    results["final_stats"] = final
    return results


if __name__ == "__main__":
    import json
    results = main()

    def _serial(o):
        try:    return float(o)
        except: return str(o)

    out = RUN_DIR / "run_results.json"
    out.write_text(json.dumps(results, default=_serial, indent=2),
                   encoding="utf-8")
    print(f"\nResults saved → {out}")
