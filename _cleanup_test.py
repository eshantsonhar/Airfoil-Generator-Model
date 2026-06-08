import sys
sys.path.insert(0, '.')
from pathlib import Path

# Clean up old failure artifacts from previous test runs
fail_dir = Path('data/failures/iter_001_aoa_+02p0')
if fail_dir.exists():
    import shutil
    shutil.rmtree(fail_dir)
    print(f"Removed stale failure dir: {fail_dir}")

fail_json = Path('data/failures/iter_001_aoa_+02p0_failure.json')
if fail_json.exists():
    fail_json.unlink()
    print(f"Removed stale failure JSON: {fail_json}")

print("Cleanup done. Ready for fresh test run.")
