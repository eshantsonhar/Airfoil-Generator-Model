import { useEffect, useState } from "react";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { apiGet } from "../api";

type Coord = { x: number; y: number };

export default function GeometryViewer() {
  const [coords, setCoords] = useState<Coord[]>([]);
  const [full, setFull] = useState<Record<string, unknown>>({});

  useEffect(() => {
    apiGet<{ coordinates: Coord[] }>("/api/best_airfoil")
      .then((d) => setCoords(d.coordinates ?? []))
      .catch(() => setCoords([]));
    apiGet<Record<string, unknown>>("/api/best_airfoil_full")
      .then(setFull)
      .catch(() => setFull({}));
  }, []);

  const geom = (full.geometry ?? {}) as Record<string, number>;
  const cst = (full.cst_params ?? {}) as Record<string, number[]>;

  return (
    <>
      <h1 style={{ marginTop: 0 }}>Geometry Viewer</h1>
      <div className="grid">
        <div className="panel" style={{ gridColumn: "span 2" }}>
          <h2>Current Airfoil</h2>
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={coords}>
                <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} stroke="#64748b" />
                <YAxis dataKey="y" stroke="#64748b" />
                <Line type="monotone" dataKey="y" stroke="#60a5fa" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel">
          <h2>Geometry Metrics</h2>
          <table>
            <tbody>
              {Object.entries(geom).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{Number(v).toFixed(5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>CST Parameters</h2>
          <pre className="log">{JSON.stringify(cst, null, 2)}</pre>
        </div>
      </div>
    </>
  );
}
