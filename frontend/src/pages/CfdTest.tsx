import { useState, useEffect, useRef } from "react";

const API_BASE = "";

type CfdResult = {
  cl: number;
  cd: number;
  status: string;
  elapsed_s: number;
  converged: boolean;
  failure_stage?: string;
  failure_reason?: string;
  files?: Record<string, number>;
};

export default function CfdTest() {
  const [upper, setUpper] = useState("0.1863,0.0779,0.2798,0.0839");
  const [lower, setLower] = useState("-0.1172,0.0642,-0.0646,0.0309");
  const [te, setTe] = useState("0.001");
  const [scale, setScale] = useState("1.0");
  const [re, setRe] = useState("100000");
  const [aoa, setAoa] = useState("4.0");
  const [mesh, setMesh] = useState("L0");
  const [loading, setLoading] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<CfdResult | null>(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const startRun = async () => {
    setLoading(true);
    setResult(null);
    setError("");
    setRunId(null);
    setStatus("submitting...");
    setElapsed(0);

    try {
      const upperArr = upper.split(",").map(Number);
      const lowerArr = lower.split(",").map(Number);
      const body = {
        upper: upperArr,
        lower: lowerArr,
        te_thickness: parseFloat(te),
        scale: parseFloat(scale),
        reynolds: parseInt(re),
        aoa: parseFloat(aoa),
        mesh_level: mesh,
      };

      const resp = await fetch(`${API_BASE}/api/cfd/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Failed to submit");
      setRunId(data.run_id);
      setStatus("queued");

      // Poll for completion
      const t0 = Date.now();
      intervalRef.current = window.setInterval(async () => {
        setElapsed(Math.round((Date.now() - t0) / 1000));
        try {
          const sresp = await fetch(
            `${API_BASE}/api/cfd/status/${data.run_id}`
          );
          const sdata = await sresp.json();
          setStatus(sdata.status);
          if (sdata.status !== "queued" && sdata.status !== "running") {
            if (intervalRef.current) clearInterval(intervalRef.current);
            // Fetch result
            const rresp = await fetch(
              `${API_BASE}/api/cfd/result/${data.run_id}`
            );
            const rdata = await rresp.json();
            if (rdata.result) setResult(rdata.result);
            if (rdata.error) setError(rdata.error);
            setLoading(false);
          }
        } catch (e: any) {
          setError(e.message);
          if (intervalRef.current) clearInterval(intervalRef.current);
          setLoading(false);
        }
      }, 1000);
    } catch (e: any) {
      setError(e.message);
      setLoading(false);
    }
  };

  return (
    <>
      <h1>CFD Run Test</h1>
      <div className="grid">
        <div className="panel">
          <h2>Parameters</h2>
          <label>
            Upper CST (4 comma-separated):
            <input value={upper} onChange={(e) => setUpper(e.target.value)} />
          </label>
          <label>
            Lower CST (4 comma-separated):
            <input value={lower} onChange={(e) => setLower(e.target.value)} />
          </label>
          <label>
            TE Thickness:
            <input
              type="number"
              value={te}
              onChange={(e) => setTe(e.target.value)}
            />
          </label>
          <label>
            Scale:
            <input
              type="number"
              value={scale}
              onChange={(e) => setScale(e.target.value)}
            />
          </label>
          <label>
            Reynolds:
            <input
              type="number"
              value={re}
              onChange={(e) => setRe(e.target.value)}
            />
          </label>
          <label>
            AoA:
            <input
              type="number"
              value={aoa}
              onChange={(e) => setAoa(e.target.value)}
            />
          </label>
          <label>
            Mesh:
            <select value={mesh} onChange={(e) => setMesh(e.target.value)}>
              <option value="L0">L0 (fastest)</option>
              <option value="L1">L1 (medium)</option>
              <option value="L2">L2 (finest)</option>
            </select>
          </label>
          <button onClick={startRun} disabled={loading}>
            {loading ? "Running..." : "Run CFD"}
          </button>
        </div>

        <div className="panel">
          <h2>Status</h2>
          {runId && <p>Run ID: <code>{runId}</code></p>}
          <p>Status: <strong>{status}</strong></p>
          {loading && <p>Elapsed: {elapsed}s</p>}
          {error && (
            <div className="error">{error}</div>
          )}
          {result && (
            <div>
              <h3>Result</h3>
              <table>
                <tbody>
                  <tr><td>CL</td><td>{result.cl?.toFixed(6)}</td></tr>
                  <tr><td>CD</td><td>{result.cd?.toFixed(6)}</td></tr>
                  <tr><td>Status</td><td>{result.status}</td></tr>
                  <tr><td>Converged</td><td>{result.converged ? "Yes" : "No"}</td></tr>
                  <tr><td>Time</td><td>{result.elapsed_s}s</td></tr>
                  <tr><td>Failure</td><td>{result.failure_reason || "None"}</td></tr>
                </tbody>
              </table>
              {result.files && (
                <div>
                  <h4>Generated Files</h4>
                  <ul>
                    {Object.entries(result.files).map(([n, s]) => (
                      <li key={n}>
                        {n}: {(s as number).toLocaleString()} bytes
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}