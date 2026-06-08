import { useEffect, useState } from "react";
import { apiGet } from "../api";

type FailureEntry = { path: string; name: string; size: number; modified: number };

export default function FailureAnalysis() {
  const [failures, setFailures] = useState<FailureEntry[]>([]);
  const [content, setContent] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ failures: FailureEntry[] }>("/api/failures").then((d) => setFailures(d.failures));
  }, []);

  const load = async (path: string) => {
    setSelected(path);
    const data = await apiGet<{ lines: string[] }>(
      `/api/failures/content?path=${encodeURIComponent(path)}&tail=300`
    );
    setContent(data.lines);
  };

  return (
    <>
      <h1 style={{ marginTop: 0 }}>Failure Analysis</h1>
      <div className="grid" style={{ gridTemplateColumns: "1fr 2fr" }}>
        <div className="panel">
          <h2>Archived Diagnostics</h2>
          <table>
            <tbody>
              {failures.map((f) => (
                <tr key={f.path} style={{ cursor: "pointer" }} onClick={() => load(f.path)}>
                  <td>{f.name}</td>
                  <td>{(f.size / 1024).toFixed(1)} KB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>{selected ?? "Select a file"}</h2>
          <pre className="log">{content.join("\n")}</pre>
        </div>
      </div>
    </>
  );
}
