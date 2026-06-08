// Chart configurations
Chart.defaults.color = '#94A3B8';
Chart.defaults.font.family = "'Inter', sans-serif";

let progressChart = null;
let airfoilChart = null;
let isJobRunning = false;
let currentLimits = null;

// Track last known best score to detect new bests
let lastBestScore = null;
let currentRawData = null;

function initCharts() {
    const ctxProgress = document.getElementById('progressChart').getContext('2d');
    
    progressChart = new Chart(ctxProgress, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Best Score',
                data: [],
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                borderWidth: 2,
                pointBackgroundColor: '#3B82F6',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#3B82F6',
                fill: true,
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#1E1E1E',
                    titleColor: '#F1F5F9',
                    bodyColor: '#e2e8f0',
                    borderColor: '#333333',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Simulation Attempt' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Score' }
                }
            }
        }
    });

    const ctxAirfoil = document.getElementById('airfoilChart').getContext('2d');
    airfoilChart = new Chart(ctxAirfoil, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Airfoil Surface',
                data: [],
                borderColor: '#60A5FA',
                backgroundColor: '#60A5FA',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                showLine: true,
                fill: false,
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    min: 0,
                    max: 1
                },
                y: {
                    type: 'linear',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    min: -0.2,
                    max: 0.2
                }
            }
        }
    });
}

function updateStatusUI(running) {
    isJobRunning = running;
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const inputIterations = document.getElementById('input-iterations');
    const inputBatch = document.getElementById('input-batch');
    const inputNCores = document.getElementById('input-n-cores');
    const inputUseMpi = document.getElementById('input-use-mpi');
    const inputMpiRanks = document.getElementById('input-mpi-ranks');
    const inputOmpThreads = document.getElementById('input-omp-threads');
    const inputPreferGpu = document.getElementById('input-prefer-gpu');

    if (running) {
        dot.classList.add('active');
        if (text.dataset.detail) {
            text.textContent = `Running (${text.dataset.detail})`;
        } else {
            text.textContent = 'Running';
        }
        btnStart.disabled = true;
        btnStop.disabled = false;
        inputIterations.disabled = true;
        inputBatch.disabled = true;
        inputNCores.disabled = true;
        inputUseMpi.disabled = true;
        inputMpiRanks.disabled = true;
        inputOmpThreads.disabled = true;
        inputPreferGpu.disabled = true;
    } else {
        dot.classList.remove('active');
        text.textContent = 'Idle';
        btnStart.disabled = false;
        btnStop.disabled = true;
        inputIterations.disabled = false;
        inputBatch.disabled = false;
        inputNCores.disabled = false;
        inputUseMpi.disabled = false;
        inputMpiRanks.disabled = false;
        inputOmpThreads.disabled = false;
        inputPreferGpu.disabled = false;
    }
}

async function startJob() {
    try {
        const iter = parseInt(document.getElementById('input-iterations').value);
        const batch = parseInt(document.getElementById('input-batch').value);
        const nCores = parseInt(document.getElementById('input-n-cores').value);
        const useMpi = document.getElementById('input-use-mpi').value === 'true';
        const mpiRanks = parseInt(document.getElementById('input-mpi-ranks').value);
        const ompThreads = parseInt(document.getElementById('input-omp-threads').value);
        const preferGpu = document.getElementById('input-prefer-gpu').value === 'true';
        
        const response = await fetch('/api/job/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                iterations: iter,
                batch_size: batch,
                n_cores: nCores,
                use_mpi: useMpi,
                mpi_ranks_per_case: mpiRanks,
                omp_threads_per_rank: ompThreads,
                prefer_gpu: preferGpu
            })
        });
        const payload = await response.json();
        if (!response.ok || payload.status === 'error') {
            throw new Error(payload.message || "failed to start job");
        }
        updateStatusUI(true);
    } catch (err) {
        console.error("Error starting job: ", err);
        alert(`Could not start optimization: ${err.message}`);
    }
}

function applyRuntimeDefaults(defaults) {
    if (!defaults) {
        return;
    }
    document.getElementById('input-iterations').value = defaults.iterations ?? defaults.recommended_iterations ?? 2;
    document.getElementById('input-batch').value = defaults.batch_size ?? defaults.recommended_batch_size ?? 1;
    document.getElementById('input-n-cores').value = defaults.n_cores ?? defaults.recommended_n_cores ?? 1;
    document.getElementById('input-use-mpi').value = String(defaults.use_mpi ?? defaults.recommended_use_mpi ?? false);
    document.getElementById('input-mpi-ranks').value = defaults.mpi_ranks_per_case ?? defaults.recommended_mpi_ranks_per_case ?? 1;
    document.getElementById('input-omp-threads').value = defaults.omp_threads_per_rank ?? defaults.recommended_omp_threads_per_rank ?? 1;
    document.getElementById('input-prefer-gpu').value = String(defaults.prefer_gpu ?? defaults.recommended_prefer_gpu ?? false);
}

async function loadLimits() {
    try {
        const response = await fetch('/api/limits');
        if (!response.ok) {
            return;
        }
        currentLimits = await response.json();
        applyRuntimeDefaults(currentLimits);
        document.getElementById('runtime-hint').textContent =
            `Safe defaults: ${currentLimits.recommended_n_cores} CPU cores, batch ${currentLimits.recommended_batch_size}, MPI off.`;
    } catch (err) {
        console.error("Error loading limits: ", err);
    }
}

async function stopJob() {
    try {
        await fetch('/api/job/stop', { method: 'POST' });
        updateStatusUI(false);
    } catch (err) {
        console.error("Error stopping job: ", err);
    }
}

async function fetchAndUpdate() {
    try {
        // Fetch Job Status
        const statusRes = await fetch('/api/job/status');
        const statusData = await statusRes.json();
        const statusText = document.getElementById('status-text');
        if (statusData.is_running) {
            statusText.dataset.detail = `+${statusData.new_cases_since_start} cases, ${statusData.job_age_seconds}s`;
        } else {
            statusText.dataset.detail = '';
        }
        updateStatusUI(statusData.is_running);

        // Fetch ASO Runtime Metrics
        const runtimeRes = await fetch('/api/job/runtime');
        const runtime = await runtimeRes.json();
        if (runtime && runtime.status !== 'idle') {
            const statVal = runtime.stationarity || 0;
            const compVal = runtime.complementarity || 0;
            
            document.getElementById('mon-stationarity').textContent = statVal.toFixed(5);
            document.getElementById('mon-complementarity').textContent = compVal.toFixed(5);
            document.getElementById('mon-mesh').textContent = runtime.mesh_level || 'L0';
            document.getElementById('mon-trust').textContent = runtime.trust_status || 'ACCEPTED';
            document.getElementById('mon-rho').textContent = `rho: ${(runtime.rho || 0).toFixed(3)}`;
            
            // Update bars (assuming 0.1 is 'full'/bad for stationarity)
            const statPct = Math.min(100, (statVal / 0.1) * 100);
            const compPct = Math.min(100, (compVal / 0.1) * 100);
            document.getElementById('bar-stationarity').style.width = `${statPct}%`;
            document.getElementById('bar-complementarity').style.width = `${compPct}%`;
            
            // Color coding
            document.getElementById('mon-stationarity').style.color = statVal < 1e-4 ? '#34D399' : '#F1F5F9';
            document.getElementById('mon-complementarity').style.color = compVal < 1e-4 ? '#34D399' : '#F1F5F9';
        }

        // Fetch stats
        const statsRes = await fetch('/api/stats');
        const stats = await statsRes.json();
        
        document.getElementById('val-total-cases').textContent = stats.total_cases;
        document.getElementById('val-best-score').textContent = stats.best_score.toFixed(3);
        document.getElementById('val-best-eff').textContent = stats.best_efficiency.toFixed(2);

        // Detect new best — fetch full raw data when score improves
        const isNewBest = lastBestScore === null || stats.best_score > lastBestScore;
        if (isNewBest && stats.total_cases > 0) {
            const shouldFlash = lastBestScore !== null; // don't flash on first load
            lastBestScore = stats.best_score;
            fetchAndRenderRawData(shouldFlash);
        }

        // Fetch progress
        const progRes = await fetch('/api/progress');
        const progress = await progRes.json();
        
        if (progress.length > 0) {
            progressChart.data.labels = progress.map(p => p.iteration);
            progressChart.data.datasets[0].data = progress.map(p => p.best_score);
            progressChart.update();
        }

        // Fetch best airfoil shape
        const airfoilRes = await fetch('/api/best_airfoil');
        const bestAirfoil = await airfoilRes.json();
        
        if (bestAirfoil.coordinates && bestAirfoil.coordinates.length > 0) {
            airfoilChart.data.datasets[0].data = bestAirfoil.coordinates;
            airfoilChart.options.plugins.title = {
                display: true,
                text: `Score: ${bestAirfoil.score.toFixed(3)} | Re: ${bestAirfoil.reynolds}`,
                color: '#94A3B8'
            };
            airfoilChart.update();
        }

    } catch (err) {
        console.error("Error fetching dashboard data: ", err);
    }
}

// ── Raw Data Panel ────────────────────────────────────────────────

function makeRow(label, value) {
    return `<tr><td>${label}</td><td>${value}</td></tr>`;
}

async function fetchAndRenderRawData(flashBadge) {
    try {
        const res = await fetch('/api/best_airfoil_full');
        if (!res.ok) return;
        const d = await res.json();
        if (!d || !d.case_key) return;
        currentRawData = d;

        const panel = document.getElementById('raw-data-panel');
        panel.style.display = 'block';

        // Glow + badge on new best
        const badge = document.getElementById('new-best-badge');
        if (flashBadge) {
            panel.classList.add('new-best-glow');
            badge.classList.add('visible');
            setTimeout(() => {
                panel.classList.remove('new-best-glow');
                badge.classList.remove('visible');
            }, 5000);
        }

        // Identity
        document.getElementById('tbl-identity').innerHTML =
            makeRow('Case Key', `<code style="font-size:0.78rem;word-break:break-all">${d.case_key}</code>`) +
            makeRow('Reynolds', d.reynolds.toLocaleString()) +
            makeRow('Signature', `<code style="font-size:0.72rem;word-break:break-all">${d.signature}</code>`);

        // CST params
        const c = d.cst_params;
        document.getElementById('tbl-cst').innerHTML =
            makeRow('Upper[0]', c.upper[0].toFixed(6)) +
            makeRow('Upper[1]', c.upper[1].toFixed(6)) +
            makeRow('Upper[2]', c.upper[2].toFixed(6)) +
            makeRow('Upper[3]', c.upper[3].toFixed(6)) +
            makeRow('Lower[0]', c.lower[0].toFixed(6)) +
            makeRow('Lower[1]', c.lower[1].toFixed(6)) +
            makeRow('Lower[2]', c.lower[2].toFixed(6)) +
            makeRow('Lower[3]', c.lower[3].toFixed(6)) +
            makeRow('TE Thickness', c.te_thickness.toFixed(6));

        // Geometry
        const g = d.geometry;
        document.getElementById('tbl-geometry').innerHTML =
            makeRow('Max Thickness', g.max_thickness.toFixed(6)) +
            makeRow('Max Camber', g.max_camber.toFixed(6)) +
            makeRow('LE Radius', g.leading_edge_radius.toFixed(6)) +
            makeRow('Smoothness Score', g.smoothness_score.toFixed(6)) +
            makeRow('Curvature Spike', g.curvature_spike.toFixed(4)) +
            makeRow('Prior Score', g.prior_score.toFixed(6));

        // Score breakdown
        const s = d.score_breakdown;
        document.getElementById('tbl-score').innerHTML =
            makeRow('Overall Score', `<strong style="color:#3B82F6">${s.score.toFixed(6)}</strong>`) +
            makeRow('Stall Angle (°)', s.stall_angle_deg.toFixed(4)) +
            makeRow('Cd at Cruise', s.cd_at_cruise.toFixed(6)) +
            makeRow('Separation Penalty', s.separation_penalty.toFixed(6)) +
            makeRow('Instability Penalty', s.instability_penalty.toFixed(6));

        // Polar table
        const tbody = document.getElementById('polar-body');
        tbody.innerHTML = d.polar.map(p =>
            `<tr>
                <td>${p.aoa_deg.toFixed(1)}</td>
                <td>${p.cl.toFixed(5)}</td>
                <td>${p.cd.toFixed(5)}</td>
                <td>${p.efficiency.toFixed(2)}</td>
            </tr>`
        ).join('');

        // .dat preview
        document.getElementById('dat-preview').textContent = d.dat_text;

    } catch (err) {
        console.error("Error fetching raw airfoil data:", err);
    }
}

function downloadDat() {
    if (!currentRawData) return;
    const blob = new Blob([currentRawData.dat_text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `best_airfoil_score${currentRawData.score_breakdown.score.toFixed(4)}.dat`;
    a.click();
    URL.revokeObjectURL(a.href);
}

function copySignature() {
    if (!currentRawData) return;
    navigator.clipboard.writeText(currentRawData.signature).then(() => {
        const btn = document.getElementById('btn-copy-sig');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initCharts();

    document.getElementById('btn-start').addEventListener('click', startJob);
    document.getElementById('btn-stop').addEventListener('click', stopJob);
    document.getElementById('btn-download-dat').addEventListener('click', downloadDat);
    document.getElementById('btn-copy-sig').addEventListener('click', copySignature);

    loadLimits();
    fetchAndUpdate();
    // Poll every 3 seconds
    setInterval(fetchAndUpdate, 3000);
});
