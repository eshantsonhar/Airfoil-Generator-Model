import { useEffect, useState } from "react";
import { apiGet, apiPost } from "../api";
import { useTelemetry } from "../hooks/useTelemetry";

type JobStatus = {
  is_running: boolean;
  detailed_status: string;
  runtime_data: Record<string, unknown>;
  job_age_seconds: number;
};

type Stats = { total_cases: number; best_score: number; best_efficiency: number };

export default function Dashboard() {
  const { connected, events } = useTelemetry();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [watchdog, setWatchdog] = useState<Record<string, unknown>>({});

  useEffect(() => {
    const tick = async () => {
      // Fetch each independently so one failure doesn't block all
      try {
        const j = await apiGet<JobStatus>("/api/job/status");
        setJob(j);
      } catch {
        /* backend may be starting */
      }
      try {
        const s = await apiGet<Stats>("/api/stats");
        setStats(s);
      } catch {
        /* backend may be starting */
      }
      try {
        const w = await apiGet<Record<string, unknown>>("/api/watchdog/status");
        setWatchdog(w);
      } catch {
        /* backend may be starting */
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, []);

  const rt = (job?.runtime_data ?? {}) as Record<string, unknown>;
  const clHist = (rt.cl_history as number[]) ?? [];
  const cdHist = (rt.cd_history as number[]) ?? [];
  const liveCl = clHist.length ? clHist[clHist.length - 1] : null;
  const liveCd = cdHist.length ? cdHist[cdHist.length - 1] : null;

  const startJob = async () => {
    await apiPost("/api/job/start", {
      iterations: 2,
      batch_size: 1,
      n_cores: 0,
      use_mpi: false,
      mpi_ranks_per_case: 1,
      omp_threads_per_rank: 1,
      prefer_gpu: false,
    });
  };

  const stopJob = async () => {
    await apiPost("/api/job/stop");
  };

  return (
    <>
      <h1 style={{ marginTop: 0 }}>Research Dashboard</h1>
      <div className="grid">
        <div className="panel">
          <h2>Optimizer</h2>
          <div className="metric">
            {String(rt.convergence_status ?? "—")}
            <small>
              Iteration {String(rt.current_iteration ?? 0)} / {String(rt.total_iterations ?? "?")}
            </small>
          </div>
        </div>
        <div className="panel">
          <h2>CFD</h2>
          <div className={`metric ${job?.is_running ? "status-ok" : ""}`}>
            {job?.is_running ? "RUNNING" : String(job?.detailed_status ?? "idle").toUpperCase()}
            <small>{String(rt.running_cases_count ?? 0)} active cases</small>
          </div>
        </div>
        <div className="panel">
          <h2>Watchdog</h2>
          <div className={`metric ${watchdog.watchdog_status === "OK" ? "status-ok" : "status-warn"}`}>
            {String(watchdog.watchdog_status ?? "UNKNOWN")}
            <small>Telemetry WS {connected ? "connected" : "disconnected"}</small>
          </div>
        </div>
        <div className="panel">
          <h2>Trust Region</h2>
          <div className="metric">
            ρ = {Number(rt.rho ?? 0).toExponential(2)}
            <small>{String(rt.trust_status ?? "—")}</small>
          </div>
        </div>
        <div className="panel">
          <h2>Live Cl / Cd</h2>
          <div className="metric">
            {liveCl != null ? liveCl.toFixed(4) : "—"} / {liveCd != null ? liveCd.toFixed(5) : "—"}
            <small>
              L/D ={" "}
              {liveCl != null && liveCd
                ? (liveCl / Math.max(liveCd, 1e-10)).toFixed(2)
                : "—"}
            </small>
          </div>
        </div>
        <div className="panel">
          <h2>Best Score</h2>
          <div className="metric">
            {stats?.best_score?.toFixed(4) ?? "—"}
            <small>{stats?.total_cases ?? 0} archived cases</small>
          </div>
        </div>
      </div>

      <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
        <button onClick={startJob} disabled={job?.is_running}>
          Start Optimization
        </button>
        <button className="secondary" onClick={stopJob} disabled={!job?.is_running}>
          Stop
        </button>
        <span className="badge">Events: {events.length}</span>
        <span className="badge">Runtime: {Math.round(job?.job_age_seconds ?? 0)}s</span>
      </div>
    </>
  );
}
