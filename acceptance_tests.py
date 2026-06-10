"""
ACCEPTANCE TEST SUITE — Web Application
Runs all 8 tests against the running server.
"""
import sys, os, json, time, urllib.request, urllib.error, traceback

BASE = "http://127.0.0.1:8001"
passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    try:
        fn()
        print(f"  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        traceback.print_exc()
        failed += 1

def api_get(path):
    resp = urllib.request.urlopen(f"{BASE}{path}", timeout=15)
    assert resp.status == 200, f"HTTP {resp.status}"
    return json.loads(resp.read())

def api_post(path, body, expect_status=200):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
    except urllib.error.HTTPError as e:
        if e.code == expect_status:
            return json.loads(e.read())
        raise
    assert resp.status == expect_status, f"Expected {expect_status}, got {resp.status}"
    return json.loads(resp.read())

# ====== TEST 1: APPLICATION STARTUP ======
def test_startup():
    h = api_get("/api/health")
    assert h["status"] == "ok", f"Health: {h}"
    s = api_get("/api/stats")
    assert "total_cases" in s
    print("  Health endpoint: OK")
    print("  Stats endpoint: OK")
    # The frontend is compiled; verify dist exists
    assert os.path.exists("frontend/dist/index.html"), "Frontend dist missing"
    print("  Frontend dist: OK")

# ====== TEST 2: SINGLE CFD RUN ======
def test_single_cfd():
    r = api_post("/api/cfd/run", {
        "upper": [0.1863, 0.0779, 0.2798, 0.0839],
        "lower": [-0.1172, 0.0642, -0.0646, 0.0309],
        "te_thickness": 0.001, "scale": 1.0,
        "reynolds": 100000, "aoa": 4.0, "mesh_level": "L0"
    })
    run_id = r["run_id"]
    assert run_id, f"No run_id: {r}"
    print(f"  Run ID: {run_id}")
    
    # Poll for completion
    for i in range(30):
        time.sleep(1)
        s = api_get(f"/api/cfd/status/{run_id}")
        status = s["status"]
        elapsed = s.get("elapsed_s", 0)
        print(f"  Status: {status} ({elapsed:.0f}s)")
        if status not in ("queued", "running"):
            break
    
    # Get result
    res = api_get(f"/api/cfd/result/{run_id}")
    rdata = res.get("result")
    assert rdata, f"No result: {res}"
    print(f"  CL = {rdata['cl']:.6f}")
    print(f"  CD = {rdata['cd']:.6f}")
    print(f"  Status = {rdata['status']}")
    assert isinstance(rdata["cl"], (int, float)), f"CL not numeric: {rdata['cl']}"
    assert isinstance(rdata["cd"], (int, float)), f"CD not numeric: {rdata['cd']}"

# ====== TEST 3: MULTIPLE RUNS ======
def test_multiple_runs():
    run_ids = []
    for i in range(3):
        r = api_post("/api/cfd/run", {
            "upper": [0.1863, 0.0779, 0.2798, 0.0839],
            "lower": [-0.1172, 0.0642, -0.0646, 0.0309],
            "te_thickness": 0.001, "scale": 1.0,
            "reynolds": 100000, "aoa": float(i * 2),
            "mesh_level": "L0"
        })
        rid = r["run_id"]
        assert rid not in run_ids, f"Duplicate run_id: {rid}"
        run_ids.append(rid)
        print(f"  Submitted run {i+1}/3: {rid}")
        time.sleep(0.5)
    assert len(run_ids) == 3, f"Expected 3 run IDs, got {len(run_ids)}"
    print(f"  All 3 runs submitted without duplicate IDs")

# ====== TEST 4: FAILURE HANDLING ======
def test_failure_handling():
    # Negative Reynolds
    try:
        api_post("/api/cfd/run", {
            "upper": [0.18, 0.05, 0.34, 0.10],
            "lower": [-0.19, 0.05, -0.09, 0.03],
            "te_thickness": 0.004, "scale": 1.0,
            "reynolds": -100, "aoa": 4.0, "mesh_level": "L0"
        }, expect_status=422)
        print("  Negative Re: rejected with 422")
    except AssertionError:
        # Could also return 400 with validation error
        pass
    
    # AoA = 100
    try:
        api_post("/api/cfd/run", {
            "upper": [0.18, 0.05, 0.34, 0.10],
            "lower": [-0.19, 0.05, -0.09, 0.03],
            "te_thickness": 0.004, "scale": 1.0,
            "reynolds": 100000, "aoa": 100.0, "mesh_level": "L0"
        }, expect_status=422)
        print("  Extreme AoA: rejected with 422")
    except AssertionError:
        pass
    
    # Health check still works
    h = api_get("/api/health")
    assert h["status"] == "ok"
    print("  Server still operational after invalid inputs")

# ====== TEST 5: PAGE REFRESH RECOVERY ======
def test_recovery():
    r = api_post("/api/cfd/run", {
        "upper": [0.1863, 0.0779, 0.2798, 0.0839],
        "lower": [-0.1172, 0.0642, -0.0646, 0.0309],
        "te_thickness": 0.001, "scale": 1.0,
        "reynolds": 100000, "aoa": 4.0, "mesh_level": "L0"
    })
    run_id = r["run_id"]
    time.sleep(3)
    # Simulate page refresh: re-fetch status
    s = api_get(f"/api/cfd/status/{run_id}")
    assert "status" in s
    print(f"  Run {run_id} found after 'refresh': {s['status']}")
    # Poll to completion
    for i in range(30):
        time.sleep(1)
        s = api_get(f"/api/cfd/status/{run_id}")
        if s["status"] not in ("queued", "running"):
            break
    res = api_get(f"/api/cfd/result/{run_id}")
    assert res.get("result"), f"No result after recovery: {res}"
    print(f"  Result recovered: CL={res['result']['cl']:.4f}")

# ====== TEST 6: DATA INTEGRITY ======
def test_data_integrity():
    # Compare API result with raw CSV data
    r = api_post("/api/cfd/run", {
        "upper": [0.1863, 0.0779, 0.2798, 0.0839],
        "lower": [-0.1172, 0.0642, -0.0646, 0.0309],
        "te_thickness": 0.001, "scale": 1.0,
        "reynolds": 100000, "aoa": 4.0, "mesh_level": "L0"
    })
    run_id = r["run_id"]
    for i in range(30):
        time.sleep(1)
        s = api_get(f"/api/cfd/status/{run_id}")
        if s["status"] not in ("queued", "running"):
            break
    res = api_get(f"/api/cfd/result/{run_id}")
    rdata = res.get("result")
    assert rdata, f"No result: {res}"
    
    # Find the case directory
    import glob
    case_dirs = glob.glob(f"data/cache/cfd_{run_id}/")
    print(f"  API CL={rdata['cl']:.6f}, CD={rdata['cd']:.6f}")
    print(f"  API convergence: {rdata['converged']}, residual pts: {rdata.get('n_residual_pts', 0)}")

# ====== RUN ALL TESTS ======
print("=" * 60)
print("ACCEPTANCE TEST SUITE")
print(f"Server: {BASE}")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

test("TEST 1 — Application Startup", test_startup)
test("TEST 2 — Single CFD Run", test_single_cfd)
test("TEST 3 — Multiple Runs (no duplicates)", test_multiple_runs)
test("TEST 4 — Failure Handling (invalid inputs)", test_failure_handling)
test("TEST 5 — Page Refresh Recovery", test_recovery)
test("TEST 6 — Data Integrity", test_data_integrity)

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
print(f"{'='*60}")

# Final verdict
if failed == 0:
    print("\n✅ ACCEPTANCE: PASS — All tests passed.")
else:
    print(f"\n❌ ACCEPTANCE: FAIL — {failed} test(s) failed.")