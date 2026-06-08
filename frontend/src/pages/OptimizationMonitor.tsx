import { useEffect, useState } from "react";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis, Legend } from "recharts";
import { apiGet } from "../api";
import { useTelemetry } from "../hooks/useTelemetry";

export default function OptimizationMonitor() {
  const { events } = useTelemetry();
  const [runtime, setRuntime] = useState<Record<string, unknown>>({});

  useEffect(() => {
    const tick = () =>
      apiGet<Record<string, unknown>>("/api/job/runtime").then(setRuntime).catch(() => {});
    tick();
    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  }, []);

  const obj = ((runtime.objective_history as number[]) ?? []).map((v, i) => ({
    i: i + 1,
    objective: v,
  }));
  const cl = ((runtime.cl_history as number[]) ?? []).map((v, i) => ({ i: i + 1, cl: v }));
  const grad = ((runtime.gradient_norm_history as number[]) ?? []).map((v, i) => ({
    i: i + 1,
    grad: v,
  }));
  const trust = ((runtime.trust_radius_history as number[]) ?? []).map((v, i) => ({
    i: i + 1,
    radius: v,
  }));

  const mmaEvents = events.filter((e) => e.event_type === "mma_step").slice(-50);

  return (
    <>
      <h1 style={{ marginTop: 0 }}>Live Optimization Monitor</h1>
      <div className="grid">
        <ChartPanel title="Objective" data={obj} lines={[{ key: "objective", color: "#3b82f6" }]} />
        <ChartPanel title="Cl history" data={cl} lines={[{ key: "cl", color: "#22c55e" }]} />
        <ChartPanel title="Gradient norm" data={grad} lines={[{ key: "grad", color: "#f59e0b" }]} />
        <ChartPanel title="Trust radius" data={trust} lines={[{ key: "radius", color: "#a78bfa" }]} />
      </div>
      <div className="panel" style={{ marginTop: "0.75rem" }}>
        <h2>Live MMA Events (WebSocket)</h2>
        <div className="log">
          {mmaEvents.length === 0
            ? "Waiting for optimization telemetry…"
            : mmaEvents.map((e, idx) => (
                <div key={idx}>
                  iter={String(e.iteration)} obj={String(e.objective)} ρ={String(e.gain_ratio)}
                </div>
              ))}
        </div>
      </div>
    </>
  );
}

function ChartPanel({
  title,
  data,
  lines,
}: {
  title: string;
  data: Record<string, number>[];
  lines: { key: string; color: string }[];
}) {
  return (
    <div className="panel">
      <h2>{title}</h2>
      <div className="chart-wrap">
        <ResponsiveContainer>
          <LineChart data={data}>
            <XAxis dataKey="i" stroke="#64748b" />
            <YAxis stroke="#64748b" />
            <Legend />
            {lines.map((l) => (
              <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
