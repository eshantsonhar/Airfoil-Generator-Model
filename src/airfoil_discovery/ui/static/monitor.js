/**
 * monitor.js — Real-time CFD Run Monitor
 *
 * Reads case_id from the URL path (/monitor/<case_id>),
 * polls /api/monitor/{case_id}/history and /api/monitor/{case_id}/surface,
 * and renders live line charts for all aerodynamic quantities.
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
Chart.defaults.color = '#94A3B8';
Chart.defaults.font.family = "'Inter', sans-serif";

const POLL_MS = 2000;       // polling interval while running
const DONE_POLL_MS = 5000;  // slower poll once converged / failed

const COLORS = {
  cl:           '#60A5FA',   // blue
  cd:           '#F87171',   // red
  eff:          '#34D399',   // green
  delta:        '#FBBF24',   // amber
  cfl:          '#A78BFA',   // purple
  // Residual channels — distinct colors
  continuity:   '#38BDF8',
  momentum_x:   '#FB923C',
  momentum_y:   '#F472B6',
  turbulence_k: '#A3E635',
  turbulence_omega: '#E879F9',
  transition_gamma: '#FDE68A',
  fallback:     ['#60A5FA','#F87171','#34D399','#FBBF24','#A78BFA','#38BDF8','#FB923C'],
  // Surface
  cp_upper:     '#60A5FA',
  cp_lower:     '#F87171',
  cf_upper:     '#34D399',
  cf_lower:     '#FBBF24',
};

// ── State ─────────────────────────────────────────────────────────────────────
const caseId = decodeURIComponent(location.pathname.replace(/^\/monitor\//, ''));
let charts = {};
let pollTimer = null;
let lastIter = 0;
let isStopped = false;

// ── Utility ───────────────────────────────────────────────────────────────────
function fmt(v, decimals = 5) {
  if (v === null || v === undefined) return '—';
  if (typeof v !== 'number') return String(v);
  return v.toFixed(decimals);
}

function setFlag(el, flag) {
  if (!el) return;
  el.textContent = flag || '—';
  el.className = el.className.replace(/\b(pass|marginal|fail)\b/g, '');
  if (flag === 'PASS') el.classList.add('pass');
  else if (flag === 'MARGINAL') el.classList.add('marginal');
  else if (flag === 'FAIL') el.classList.add('fail');
}

function makeLineDataset(label, data, color, extra = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color + '22',
    borderWidth: 1.5,
    pointRadius: 0,
    pointHoverRadius: 3,
    fill: false,
    tension: 0,
    ...extra,
  };
}

function makeChart(id, yLabel, logScale = false, yMin = undefined) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');

  const scaleOpts = logScale
    ? { type: 'logarithmic', grid: { color: 'rgba(255,255,255,0.06)' }, title: { display: true, text: yLabel } }
    : {
        type: 'linear',
        grid: { color: 'rgba(255,255,255,0.06)' },
        title: { display: true, text: yLabel },
        ...(yMin !== undefined ? { min: yMin } : {}),
      };

  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: { display: true, position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: '#1E1E1E',
          titleColor: '#F1F5F9',
          bodyColor: '#e2e8f0',
          borderColor: '#333',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          title: { display: true, text: 'Iteration' },
          ticks: { maxTicksLimit: 10, font: { size: 10 } },
        },
        y: scaleOpts,
      },
    },
  });
}

function updateChart(chart, labels, datasets) {
  if (!chart) return;
  chart.data.labels = labels;
  chart.data.datasets = datasets;
  chart.update('none');   // no animation for live perf
}

function makeScatterChart(id, xLabel) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  return new Chart(canvas.getContext('2d'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: { display: true, position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          backgroundColor: '#1E1E1E',
          titleColor: '#F1F5F9',
          bodyColor: '#e2e8f0',
          borderColor: '#333',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          type: 'linear',
          grid: { color: 'rgba(255,255,255,0.05)' },
          title: { display: true, text: xLabel },
          min: 0,
          max: 1,
        },
        y: {
          type: 'linear',
          grid: {
            color: (ctx) => ctx.tick.value === 0 ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.05)',
          },
        },
      },
    },
  });
}

function zipXY(xs, ys) {
  if (!xs || !ys) return [];
  return xs.map((x, i) => ({ x, y: ys[i] }));
}

// ── Initialise charts ─────────────────────────────────────────────────────────
function initCharts() {
  charts.cl  = makeChart('chart-cl',  'Cl');
  charts.cd  = makeChart('chart-cd',  'Cd', false, 0);
  charts.eff = makeChart('chart-eff', 'Cl/Cd');
  charts.res = makeChart('chart-res', 'RMS Residual', true);
  charts.dcl = makeChart('chart-dcl', 'ΔCl', false, 0);
  charts.dcd = makeChart('chart-dcd', 'ΔCd', false, 0);

  // Surface charts
  charts.cp = makeScatterChart('chart-cp', 'x/c');
  charts.cf = makeScatterChart('chart-cf', 'x/c');

  // Polar charts (summary tab)
  charts.polarCl  = makeChart('chart-polar-cl',  'Cl');
  charts.polarCd  = makeChart('chart-polar-cd',  'Cd');
  charts.polarEff = makeChart('chart-polar-eff', 'Cl/Cd');
}

// ── Render history data ───────────────────────────────────────────────────────
function renderHistory(data) {
  if (!data.ready) return;

  const t = data.traces;
  const iters = t.iterations || [];
  const n = iters.length;
  if (n === 0) return;

  // Clamp to avoid duplicating unchanged data
  lastIter = n;

  const sc = data.scalars || {};
  const conv = data.convergence_flag || 'FAIL';

  // Scalar strip
  document.getElementById('s-iter').textContent = n;
  document.getElementById('s-cl').textContent   = fmt(sc.cl, 5);
  document.getElementById('s-cd').textContent   = fmt(sc.cd, 6);
  document.getElementById('s-eff').textContent  = fmt(sc.efficiency, 2);
  document.getElementById('s-dcl').textContent  = fmt(sc.delta_cl, 6);
  document.getElementById('s-dcd').textContent  = fmt(sc.delta_cd, 6);
  document.getElementById('s-cfl').textContent  = fmt(sc.cfl, 2);
  const sConv = document.getElementById('s-conv');
  sConv.textContent = conv;
  sConv.style.color = conv === 'PASS' ? '#34D399' : conv === 'MARGINAL' ? '#FBBF24' : '#EF4444';

  // Header flag
  const flagBadge = document.getElementById('conv-flag-badge');
  setFlag(flagBadge, conv);

  // Cl chart
  updateChart(charts.cl, iters, [makeLineDataset('Cl', t.cl || [], COLORS.cl)]);

  // Cd chart
  updateChart(charts.cd, iters, [makeLineDataset('Cd', t.cd || [], COLORS.cd)]);

  // Efficiency chart
  updateChart(charts.eff, iters, [makeLineDataset('Cl/Cd', t.efficiency || [], COLORS.eff)]);

  // Residuals — log scale, all channels
  const resChannels = data.residual_channels || [];
  const resColorMap = {
    continuity:        COLORS.continuity,
    momentum_x:        COLORS.momentum_x,
    momentum_y:        COLORS.momentum_y,
    turbulence_k:      COLORS.turbulence_k,
    turbulence_omega:  COLORS.turbulence_omega,
    transition_gamma:  COLORS.transition_gamma,
  };
  const resDatasets = resChannels.map((ch, idx) => {
    const raw = t[`res_${ch}`] || [];
    // For log chart: take abs value (residuals may be signed)
    const absVals = raw.map(v => Math.abs(v));
    const color = resColorMap[ch] || COLORS.fallback[idx % COLORS.fallback.length];
    return makeLineDataset(ch.replace(/_/g, ' '), absVals, color);
  });
  if (resDatasets.length > 0) {
    updateChart(charts.res, iters, resDatasets);
  }

  // ΔCl chart
  updateChart(charts.dcl, iters, [makeLineDataset('ΔCl', t.delta_cl || [], COLORS.delta)]);

  // ΔCd chart
  updateChart(charts.dcd, iters, [makeLineDataset('ΔCd', t.delta_cd || [], COLORS.delta)]);
}

// ── Render surface distributions ──────────────────────────────────────────────
function renderSurface(data) {
  const notReady = document.getElementById('surface-not-ready');

  if (!data || !data.ready) {
    if (notReady) notReady.style.display = 'block';
    return;
  }
  if (notReady) notReady.style.display = 'none';

  const upper = data.upper || {};
  const lower = data.lower || {};

  // Scalar badges
  const xsep  = data.x_sep  !== null && data.x_sep  !== undefined ? data.x_sep.toFixed(4)  : '—';
  const xreat = data.x_reat !== null && data.x_reat !== undefined ? data.x_reat.toFixed(4) : '—';
  const xtr   = data.x_tr   !== null && data.x_tr   !== undefined ? data.x_tr.toFixed(4)   : '—';
  const bub   = (data.bubble_length || 0).toFixed(4);
  document.getElementById('sf-sep').textContent  = xsep;
  document.getElementById('sf-reat').textContent = xreat;
  document.getElementById('sf-bub').textContent  = bub;
  document.getElementById('sf-tr').textContent   = xtr;

  // Cp chart: upper (inverted convention) + lower
  const cpDatasets = [
    {
      label: 'Cp upper',
      data: zipXY(upper.x, upper.cp),
      borderColor: COLORS.cp_upper,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      showLine: true,
    },
    {
      label: 'Cp lower',
      data: zipXY(lower.x, lower.cp),
      borderColor: COLORS.cp_lower,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      showLine: true,
    },
  ];

  // Vertical markers for separation/reattachment/transition
  if (data.x_sep !== null && data.x_sep !== undefined) {
    cpDatasets.push({
      label: `x_sep ${xsep}`,
      data: [{ x: data.x_sep, y: -3 }, { x: data.x_sep, y: 1.5 }],
      borderColor: '#EF4444',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [4, 3],
      pointRadius: 0,
      showLine: true,
    });
  }
  if (data.x_reat !== null && data.x_reat !== undefined) {
    cpDatasets.push({
      label: `x_reat ${xreat}`,
      data: [{ x: data.x_reat, y: -3 }, { x: data.x_reat, y: 1.5 }],
      borderColor: '#34D399',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [4, 3],
      pointRadius: 0,
      showLine: true,
    });
  }
  if (data.x_tr !== null && data.x_tr !== undefined) {
    cpDatasets.push({
      label: `x_tr ${xtr}`,
      data: [{ x: data.x_tr, y: -3 }, { x: data.x_tr, y: 1.5 }],
      borderColor: '#FBBF24',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [2, 4],
      pointRadius: 0,
      showLine: true,
    });
  }

  if (charts.cp) {
    charts.cp.data.datasets = cpDatasets;
    // Invert y-axis for Cp (aerodynamic convention: negative up)
    charts.cp.options.scales.y.reverse = true;
    charts.cp.update('none');
  }

  // Cf chart: upper + lower, with zero reference
  const cfDatasets = [
    {
      label: 'Cf upper',
      data: zipXY(upper.x, upper.cf),
      borderColor: COLORS.cf_upper,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      showLine: true,
    },
    {
      label: 'Cf lower',
      data: zipXY(lower.x, lower.cf),
      borderColor: COLORS.cf_lower,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      showLine: true,
    },
    // Zero reference line
    {
      label: 'Cf = 0',
      data: [{ x: 0, y: 0 }, { x: 1, y: 0 }],
      borderColor: 'rgba(255,255,255,0.25)',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [4, 3],
      pointRadius: 0,
      showLine: true,
    },
  ];

  if (data.x_sep !== null && data.x_sep !== undefined) {
    cfDatasets.push({
      label: `x_sep ${xsep}`,
      data: [{ x: data.x_sep, y: -0.02 }, { x: data.x_sep, y: 0.02 }],
      borderColor: '#EF4444',
      borderWidth: 1,
      borderDash: [4, 3],
      pointRadius: 0,
      showLine: true,
    });
  }
  if (data.x_reat !== null && data.x_reat !== undefined) {
    cfDatasets.push({
      label: `x_reat ${xreat}`,
      data: [{ x: data.x_reat, y: -0.02 }, { x: data.x_reat, y: 0.02 }],
      borderColor: '#34D399',
      borderWidth: 1,
      borderDash: [4, 3],
      pointRadius: 0,
      showLine: true,
    });
  }

  if (charts.cf) {
    charts.cf.data.datasets = cfDatasets;
    charts.cf.update('none');
  }
}

// ── Render summary tab ────────────────────────────────────────────────────────
function renderSummary(data) {
  if (!data.ready) return;

  const flag = data.convergence_flag || 'FAIL';
  const flagBig = document.getElementById('sum-flag');
  flagBig.textContent = flag;
  flagBig.className = 'mon-flag-big ' + flag.toLowerCase();

  document.getElementById('sum-score').textContent = `Score: ${data.convergence_score ?? '—'} / 100`;

  const sc = data.scalars || {};

  function makeRows(rows) {
    return rows
      .map(([label, val]) => `<tr><td>${label}</td><td>${val}</td></tr>`)
      .join('');
  }

  document.getElementById('sum-aerotable').innerHTML = makeRows([
    ['Cl',     fmt(sc.cl, 5)],
    ['Cd',     fmt(sc.cd, 6)],
    ['Cl/Cd',  fmt(sc.efficiency, 2)],
    ['ΔCl (last 50)', fmt(sc.delta_cl, 6)],
    ['ΔCd (last 50)', fmt(sc.delta_cd, 6)],
    ['CFL',    fmt(sc.cfl, 2)],
  ]);

  document.getElementById('sum-convtable').innerHTML = makeRows([
    ['Iterations',           data.n_iterations ?? '—'],
    ['Residual drop (decades)', data.residual_drop_decades ?? '—'],
    ['Cl variation (last 50)',   fmt(data.cl_variation_last50, 6)],
    ['Cd variation (last 50)',   fmt(data.cd_variation_last50, 6)],
  ]);

  const lsb = data.lsb || {};
  document.getElementById('sum-lsbtable').innerHTML = makeRows([
    ['Separation x/c',    lsb.x_sep   !== null && lsb.x_sep   !== undefined ? lsb.x_sep.toFixed(4)   : '—'],
    ['Reattachment x/c',  lsb.x_reat  !== null && lsb.x_reat  !== undefined ? lsb.x_reat.toFixed(4)  : '—'],
    ['Transition x/c',    lsb.x_tr    !== null && lsb.x_tr    !== undefined ? lsb.x_tr.toFixed(4)    : '—'],
    ['Bubble length x/c', (lsb.bubble_length || 0).toFixed(4)],
  ]);

  // Sync header flag
  setFlag(document.getElementById('conv-flag-badge'), flag);
}

// ── Polling loop ──────────────────────────────────────────────────────────────
async function poll() {
  try {
    // Always fetch history
    const histRes = await fetch(`/api/monitor/${encodeURIComponent(caseId)}/history`);
    if (histRes.ok) {
      const histData = await histRes.json();
      renderHistory(histData);

      // Determine if we should slow down
      const flag = histData.convergence_flag;
      if (flag === 'PASS') {
        isStopped = true;  // Converged — switch to slow poll
      }
    }

    // Surface data (only fetch when on surface tab or summary tab to save bandwidth)
    const activeTab = document.querySelector('.mon-tab.active')?.dataset?.tab;
    if (activeTab === 'surface' || activeTab === 'summary') {
      const surfRes = await fetch(`/api/monitor/${encodeURIComponent(caseId)}/surface`);
      if (surfRes.ok) {
        const surfData = await surfRes.json();
        renderSurface(surfData);
        if (activeTab === 'summary') {
          // Also fetch summary
          const sumRes = await fetch(`/api/monitor/${encodeURIComponent(caseId)}/summary`);
          if (sumRes.ok) {
            renderSummary(await sumRes.json());
          }
        }
      }
    } else if (activeTab === 'live') {
      // Still want summary tab scalars for the flag badge
    }

  } catch (err) {
    console.warn('[monitor] poll error:', err.message);
  } finally {
    const delay = isStopped ? DONE_POLL_MS : POLL_MS;
    pollTimer = setTimeout(poll, delay);
  }
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.mon-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });
  document.querySelectorAll('.mon-tab-content').forEach(pane => {
    pane.classList.toggle('hidden', pane.id !== `tab-${name}`);
  });

  // Trigger an immediate poll when switching tabs so data refreshes
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, 0);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Show case ID in header
  const badge = document.getElementById('mon-case-id');
  if (badge) badge.textContent = caseId;

  // Wire tabs
  document.querySelectorAll('.mon-tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  initCharts();
  poll();
});
