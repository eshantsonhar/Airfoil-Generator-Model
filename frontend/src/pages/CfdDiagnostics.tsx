import { useTelemetry } from "../hooks/useTelemetry";

export default function CfdDiagnostics() {
  const { events, byType } = useTelemetry();
  const cfd = byType("cfd_complete").slice(-30);
  const watchdog = byType("watchdog").slice(-20);
  const failures = byType("failure").slice(-20);

  return (
    <>
      <h1 style={{ marginTop: 0 }}>CFD Diagnostics</h1>
      <div className="grid">
        <div className="panel">
          <h2>Recent CFD Completions</h2>
          <table>
            <thead>
              <tr>
                <th>Case</th>
                <th>Cl</th>
                <th>Cd</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {cfd.map((e, i) => (
                <tr key={i}>
                  <td>{String(e.case_id)}</td>
                  <td>{Number(e.cl).toFixed(4)}</td>
                  <td>{Number(e.cd).toFixed(5)}</td>
                  <td>{String(e.status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>Watchdog Events</h2>
          <pre className="log">{JSON.stringify(watchdog, null, 2)}</pre>
        </div>
        <div className="panel">
          <h2>Failures</h2>
          <pre className="log">{JSON.stringify(failures, null, 2)}</pre>
        </div>
        <div className="panel">
          <h2>Event Stream</h2>
          <pre className="log">{events.slice(-40).map((e) => JSON.stringify(e)).join("\n")}</pre>
        </div>
      </div>
    </>
  );
}
