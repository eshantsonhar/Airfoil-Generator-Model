import { useEffect, useState } from "react";
import { apiGet } from "../api";
import { useTelemetry } from "../hooks/useTelemetry";

export default function PhysicsAnalysis() {
  const { byType } = useTelemetry();
  const [full, setFull] = useState<Record<string, unknown>>({});

  useEffect(() => {
    apiGet<Record<string, unknown>>("/api/best_airfoil_full").then(setFull).catch(() => {});
  }, []);

  const snaps = byType("telemetry_snapshot").slice(-20);
  const transitions = (full.transition_points as unknown[]) ?? [];

  return (
    <>
      <h1 style={{ marginTop: 0 }}>Physics Analysis</h1>
      <div className="grid">
        <div className="panel">
          <h2>LSB / Transition Telemetry</h2>
          <pre className="log">{JSON.stringify(snaps, null, 2)}</pre>
        </div>
        <div className="panel">
          <h2>Transition Points (DB)</h2>
          <pre className="log">{JSON.stringify(transitions, null, 2)}</pre>
        </div>
      </div>
    </>
  );
}
