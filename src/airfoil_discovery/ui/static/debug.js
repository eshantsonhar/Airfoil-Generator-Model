function fmt(value) {
    if (value === null || value === undefined) return "-";
    return typeof value === "number" ? value.toFixed(1) : String(value);
}

async function fetchRuntime() {
    const res = await fetch("/api/job/runtime");
    return await res.json();
}

async function fetchLog() {
    const res = await fetch("/api/job/log?tail=150");
    return await res.json();
}

function renderRuntime(runtime) {
    document.getElementById("cases-completed").textContent = runtime.completed_cases ?? 0;
    document.getElementById("cases-running").textContent = runtime.running_cases_count ?? 0;
    document.getElementById("eta-total").textContent = fmt(runtime.estimated_total_remaining_s);

    const tbody = document.querySelector("#running-cases-table tbody");
    tbody.innerHTML = "";
    const runningCases = runtime.running_cases || [];
    if (runningCases.length === 0) {
        const row = document.createElement("tr");
        row.innerHTML = `<td colspan="5" style="padding:8px; color:#94A3B8;">No running cases right now.</td>`;
        tbody.appendChild(row);
        return;
    }
    for (const item of runningCases) {
        const startTs = Number(item.start_ts);
        const liveElapsed = Number.isFinite(startTs) ? Math.max(0, (Date.now() / 1000.0) - startTs) : null;
        const elapsed = liveElapsed ?? item.elapsed_s ?? null;
        const avgCaseRuntime = runtime.avg_case_runtime_s;
        const eta = item.eta_s ?? ((avgCaseRuntime && avgCaseRuntime > 0 && elapsed !== null) ? Math.max(0, avgCaseRuntime - elapsed) : null);
        const caseId = item.case_id ?? "-";
        const monitorUrl = caseId !== "-" ? `/monitor/${encodeURIComponent(caseId)}` : null;
        const row = document.createElement("tr");
        row.innerHTML = `
            <td style="padding:8px; font-family:monospace; font-size:0.82rem;">${caseId}</td>
            <td style="padding:8px;">${fmt(item.reynolds)}</td>
            <td style="padding:8px;">${fmt(elapsed)}</td>
            <td style="padding:8px;">${fmt(eta)}</td>
            <td style="padding:8px;">
              ${monitorUrl
                ? `<a href="${monitorUrl}" target="_blank" class="btn-monitor">Run Monitor</a>`
                : '<span style="color:#475569">—</span>'}
            </td>
        `;
        tbody.appendChild(row);
    }
}

function renderPhysicsPlan(runtime) {
    const planEl = document.getElementById("physics-plan");
    const runningCases = runtime.running_cases || [];
    if (runningCases.length === 0) {
        planEl.textContent = "No active case yet.";
        return;
    }

    const items = runningCases.map((item) => {
        const plan = item.simulation_plan || {};
        const stages = plan.stages || [];
        const stageMarkup = stages.map((stage) => {
            const extras = [];
            if (stage.transition) extras.push(`transition ${stage.transition}`);
            if (stage.muscl_flow !== undefined) extras.push(`MUSCL ${stage.muscl_flow ? "on" : "off"}`);
            if (stage.restart !== undefined) extras.push(stage.restart ? "restart" : "cold start");
            if (stage.outputs && stage.outputs.length) extras.push(stage.outputs.join(", "));
            if (stage.turbulence_intensity !== undefined) extras.push(`Tu ${stage.turbulence_intensity}`);
            if (stage.turb_viscosity_ratio !== undefined) extras.push(`Tv ${stage.turb_viscosity_ratio}`);
            return `
                <div class="physics-plan-card">
                    <div class="physics-plan-title">Stage ${stage.stage ?? "?"}</div>
                    <div class="physics-plan-meta">
                        ${extras.map((text) => `<span class="stage-pill">${text}</span>`).join("<br>")}
                        <div>Mesh factor: ${fmt(stage.mesh_factor)}</div>
                        <div>Iterations: ${stage.iterations ?? "-"}</div>
                        <div>CFL: ${fmt(stage.cfl)}</div>
                    </div>
                </div>
            `;
        }).join("");

        return `
            <div class="physics-plan-card">
                <div class="physics-plan-title">${item.case_id ?? "Running case"}</div>
                <div class="physics-plan-meta">
                    <div><strong>Re</strong> ${fmt(item.reynolds)}</div>
                    <div><strong>AoA sweep</strong> ${(plan.aoa_values || []).map((a) => Number(a).toFixed(1)).join(", ") || "-"}</div>
                    <div><strong>Solver</strong> ${plan.solver || "INC_RANS"}</div>
                    <div><strong>Turbulence</strong> ${plan.turbulence_model || "SST k-omega"}</div>
                    <div><strong>Transition</strong> ${plan.transition_model || "LM"}</div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.75rem;margin-top:0.75rem;">
                    ${stageMarkup}
                </div>
            </div>
        `;
    });

    planEl.innerHTML = items.join("");
}

function renderLog(runtime, logPayload) {
    const logEl = document.getElementById("debug-log");
    const events = runtime.debug_events || [];
    const eventLines = events.slice(-40).map((e) => {
        const payload = { ...e };
        delete payload.time;
        delete payload.event;
        return `[${e.time}] ${e.event} ${Object.keys(payload).length ? JSON.stringify(payload) : ""}`.trimEnd();
    });
    const processLines = (logPayload.lines || []).slice(-60);
    logEl.textContent = [
        "=== Runtime Events ===",
        ...eventLines,
        "",
        "=== Process Log Tail ===",
        ...processLines,
    ].join("\n");
}

async function refreshDebugScreen() {
    try {
        const [runtime, logPayload] = await Promise.all([fetchRuntime(), fetchLog()]);
        renderRuntime(runtime);
        renderPhysicsPlan(runtime);
        renderLog(runtime, logPayload);
    } catch (err) {
        document.getElementById("debug-log").textContent = `Failed to load debug data: ${err.message}`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    refreshDebugScreen();
    setInterval(refreshDebugScreen, 2000);
});
