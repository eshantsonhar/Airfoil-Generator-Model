from pathlib import Path

history = Path('data/failures/iter_001_aoa_+02p0/history.csv')
text = history.read_text(encoding='utf-8')
lines = text.splitlines()

headers = [item.strip().strip('"') for item in lines[0].split(',')]
traces = {h: [] for h in headers}

for line in lines[1:]:
    if not line.strip() or line.strip() == ',':
        continue
    vals = [item.strip() for item in line.split(',')]
    for i, h in enumerate(headers):
        if i < len(vals):
            try:
                traces[h].append(float(vals[i]))
            except Exception as e:
                print(f"Parse error key={h!r} val={vals[i]!r}: {e}")

print("Headers:", headers)
print("Num headers:", len(headers))
print()
print("rms[P] len:", len(traces.get("rms[P]", [])))
print("rms[P] first 3:", traces.get("rms[P]", [])[:3])
print("rms[P] last 3:", traces.get("rms[P]", [])[-3:])
print()
print("CL len:", len(traces.get("CL", [])))
print("CL first 3:", traces.get("CL", [])[:3])
print("CL last 3:", traces.get("CL", [])[-3:])
print()
print("CD len:", len(traces.get("CD", [])))
print("CD first 3:", traces.get("CD", [])[:3])
print("CD last 3:", traces.get("CD", [])[-3:])
print()
residual_history = (
    traces.get("rms[P]")
    or traces.get("RMS_PRESSURE")
    or traces.get("rms[Rho]")
    or traces.get("RMS_DENSITY")
    or traces.get("RES_RHO")
    or []
)
print("residual_history len:", len(residual_history))
print("residual_history type:", type(residual_history))
