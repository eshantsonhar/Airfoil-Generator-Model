import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import GeometryViewer from "./pages/GeometryViewer";
import OptimizationMonitor from "./pages/OptimizationMonitor";
import CfdDiagnostics from "./pages/CfdDiagnostics";
import CfdTest from "./pages/CfdTest";
import PhysicsAnalysis from "./pages/PhysicsAnalysis";
import FailureAnalysis from "./pages/FailureAnalysis";
import RunConfiguration from "./pages/RunConfiguration";

const links = [
  ["/", "Dashboard"],
  ["/geometry", "Geometry"],
  ["/optimization", "Optimization"],
  ["/cfd", "CFD Diagnostics"],
  ["/physics", "Physics"],
  ["/cfd/run", "CFD Run Test"],
  ["/failures", "Failures"],
  ["/config", "Run Config"],
] as const;

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>ASO Platform</h1>
        <nav className="nav">
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/geometry" element={<GeometryViewer />} />
          <Route path="/optimization" element={<OptimizationMonitor />} />
          <Route path="/cfd" element={<CfdDiagnostics />} />
          <Route path="/cfd/run" element={<CfdTest />} />
          <Route path="/physics" element={<PhysicsAnalysis />} />
          <Route path="/failures" element={<FailureAnalysis />} />
          <Route path="/config" element={<RunConfiguration />} />
        </Routes>
      </main>
    </div>
  );
}
