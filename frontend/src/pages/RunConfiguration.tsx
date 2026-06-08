import { useEffect, useState } from "react";
import { apiGet, apiPost } from "../api";

export default function RunConfiguration() {
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [hash, setHash] = useState("");
  const [reynolds, setReynolds] = useState(100000);
  const [iterations, setIterations] = useState(2);
  const [message, setMessage] = useState("");

  useEffect(() => {
    apiGet<{ config: Record<string, unknown>; hash: string }>("/api/config/current").then((d) => {
      setConfig(d.config);
      setHash(d.hash);
      const flow = (d.config.flow ?? {}) as Record<string, number>;
      const opt = (d.config.optimization ?? {}) as Record<string, number>;
      setReynolds(flow.reynolds_min ?? 100000);
      setIterations(opt.iterations ?? 2);
    });
  }, []);

  const save = async () => {
    const res = await apiPost<{ saved: string; hash: string }>("/api/config/save", {
      reynolds_min: reynolds,
      iterations,
    });
    setMessage(`Saved ${res.saved} (hash ${res.hash})`);
    setHash(res.hash);
  };

  return (
    <>
      <h1 style={{ marginTop: 0 }}>Run Configuration</h1>
      <div className="panel">
        <h2>Editable Parameters</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxWidth: 360 }}>
          <label>
            Reynolds
            <input type="number" value={reynolds} onChange={(e) => setReynolds(Number(e.target.value))} />
          </label>
          <label>
            Iterations
            <input type="number" value={iterations} onChange={(e) => setIterations(Number(e.target.value))} />
          </label>
          <button onClick={save}>Save Config Snapshot</button>
          {message && <p className="status-ok">{message}</p>}
          <p className="badge">Current hash: {hash}</p>
        </div>
        <h2 style={{ marginTop: "1rem" }}>Loaded YAML</h2>
        <pre className="log">{JSON.stringify(config, null, 2)}</pre>
      </div>
    </>
  );
}
