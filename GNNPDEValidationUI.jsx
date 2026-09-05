import React, { useEffect, useState, useRef } from "react";

const COLORS = {
  pass: { bg: "oklch(0.94 0.05 160)", fg: "oklch(0.4 0.12 160)", dot: "oklch(0.55 0.14 160)" },
  warn: { bg: "oklch(0.95 0.06 80)", fg: "oklch(0.45 0.12 80)", dot: "oklch(0.65 0.15 80)" },
  fail: { bg: "oklch(0.95 0.06 25)", fg: "oklch(0.45 0.14 25)", dot: "oklch(0.55 0.18 25)" },
};

// Where the computed dashboard data lives, relative to wherever this
// component is served from. Regenerate it with:
//   python -m validator.build_dashboard_data
// after training or re-validating anything -- this file is a static
// snapshot of the last time that command ran, not a live query.
const DASHBOARD_DATA_URL = "validator/results/dashboard_data.json";

// Offline fallback, used only if DASHBOARD_DATA_URL can't be fetched (e.g.
// this component is dropped into a different app with no access to this
// repo's validator/results/ directory). Real usage should always be
// looking at computed data, not this hand-typed snapshot -- keep it
// updated by copying validator/results/dashboard_data.json here if you
// want an offline copy, but don't hand-edit metrics/status/checks.
export const FALLBACK_CHECKPOINTS = [
  { id: "elliptic", stage: "elliptic", name: "Poisson (elliptic)", tagline: "2D steady-state Poisson on a 32x32 grid, multiscale graph, hard Dirichlet BC.",
    status: "warn",
    metrics: [
      { label: "test rel. L2 (mean)", value: "0.74" },
      { label: "test rel. L2 (median)", value: "0.68" },
      { label: "test rel. L2 (range)", value: "0.57–1.42" },
      { label: "epochs", value: "24" },
      { label: "hidden / layers", value: "20 / 2" },
    ],
    checks: [
      { label: "Train/val loss converge without overfitting", pass: true },
      { label: "Bump location & sign recovered correctly", pass: true },
      { label: "Long-range Green's-function halo reproduced", pass: false },
    ],
    note: "The network gets each source bump's sign and rough location right but under-predicts how far the true solution spreads: Poisson's Green's function decays slowly, and 2 message-passing layers (even with multiscale shortcuts) only give a coarse approximation of that nonlocal coupling.",
    series: null, image: null, compareId: null },

  { id: "hyperbolic_v1", stage: "hyperbolic", name: "Burgers' — v1 naive one-step", tagline: "Plain one-step supervised MSE, no rollout signal during training.",
    status: "fail",
    metrics: [
      { label: "rel. L2 @ 25%", value: "0.674" },
      { label: "rel. L2 @ 50%", value: "0.778" },
      { label: "rel. L2 @ final", value: "0.796" },
      { label: "max |pred|", value: "2.97" },
      { label: "diverged", value: "0 / 15" },
    ],
    checks: [
      { label: 'Val loss beats "predict no change" baseline', pass: true },
      { label: "Shock advects across the domain", pass: false },
      { label: "No numerical blow-up over 100 steps", pass: true },
    ],
    note: "Looks fine by training loss — but the shock steepens then stalls instead of advecting; error climbs to 0.80 within the first fifth of the trajectory and plateaus.",
    series: [[0, 0.02], [0.125, 0.674], [0.25, 0.778], [0.5, 0.796]], image: null, compareId: null },

  { id: "hyperbolic_v2", stage: "hyperbolic", name: "Burgers' — v2 5-step unrolled", tagline: "Warm-started from v1, fine-tuned with a 5-step unrolled rollout loss.",
    status: "warn",
    metrics: [
      { label: "rel. L2 @ 25%", value: "0.582" },
      { label: "rel. L2 @ 50%", value: "0.648" },
      { label: "rel. L2 @ final", value: "0.648" },
      { label: "max |pred|", value: "2.97" },
      { label: "diverged", value: "0 / 15" },
    ],
    checks: [
      { label: "Improves over v1 across the full trajectory", pass: true },
      { label: "Shock stall fully eliminated", pass: false },
      { label: "No numerical blow-up over 100 steps", pass: true },
    ],
    note: "Real, verified improvement: final error drops 0.80 → 0.65, consistent across the whole trajectory. The shock still stalls visibly. Shipped as the stable baseline for this stage.",
    series: [[0, 0.02], [0.125, 0.582], [0.25, 0.648], [0.5, 0.648]], image: null, compareId: "hyperbolic_v1" },

  { id: "hyperbolic_v3_cautionary", stage: "hyperbolic", name: "Burgers' — v3 over-trained", tagline: "v2 continued 13 more epochs at the same unroll horizon (k=5). Cautionary tale — not shipped.",
    status: "fail",
    metrics: [
      { label: "rel. L2 @ 25%", value: "2.77" },
      { label: "rel. L2 @ 50%", value: "224" },
      { label: "rel. L2 @ final", value: "1.3×10⁶" },
      { label: "max |pred|", value: "8,283,525" },
      { label: "diverged", value: "yes" },
    ],
    checks: [
      { label: "5-step training/val loss improving", pass: true },
      { label: "Full 100-step rollout stable", pass: false },
      { label: "No numerical blow-up", pass: false },
    ],
    note: "Training loss kept improving smoothly while the model diverged. A fixed k=5 horizon can't see a per-step amplification of ~1.17x: invisible at 5 steps, catastrophic at 100.",
    series: [[0, 0.03], [0.125, 2.77], [0.25, 224], [0.5, 1300000]], image: null, compareId: "hyperbolic_v2" },

  { id: "hyperbolic_v4", stage: "hyperbolic", name: "Burgers' — v4 curriculum unrolling", tagline: "Unroll horizon k grows 1→15 over training, plus gradient clipping and per-chunk rollout spot checks.",
    status: "pass",
    metrics: [
      { label: "rel. L2 @ 25%", value: "0.488" },
      { label: "rel. L2 @ 50%", value: "0.594" },
      { label: "rel. L2 @ final", value: "0.626" },
      { label: "max |pred|", value: "2.97" },
      { label: "diverged", value: "0 / 15" },
    ],
    checks: [
      { label: "No divergence across all 15 test trajectories", pass: true },
      { label: "Improves over v2 baseline", pass: true },
      { label: "Per-chunk rollout spot checks stayed at true scale", pass: true },
    ],
    note: "Growing k removes the blind spot that broke v3. Best model trained directly in u-space, though a residual bias toward under-propagation remains.",
    series: [[0, 0.02], [0.125, 0.488], [0.25, 0.594], [0.5, 0.626]], image: null, compareId: "hyperbolic_v2" },

  { id: "hyperbolic_cole_hopf_v2", stage: "hyperbolic", name: "Burgers' — Cole-Hopf v2 (log-φ)", tagline: "Plain one-step training in a transformed coordinate (psi = log phi) instead of u directly. Best model overall.",
    status: "pass",
    metrics: [
      { label: "rel. L2 @ 25%", value: "0.105" },
      { label: "rel. L2 @ 50%", value: "0.122" },
      { label: "rel. L2 @ final", value: "0.237" },
      { label: "max |pred|", value: "2.97" },
      { label: "diverged", value: "0 / 15" },
    ],
    checks: [
      { label: "No divergence across all 15 test trajectories", pass: true },
      { label: "Beats best u-space model (v4) at every checkpoint", pass: true },
      { label: "No positivity-constraint failure (v1 Cole-Hopf issue)", pass: true },
    ],
    note: "4.6x better than v4 at the 25% mark using plain one-step training. psi is the antiderivative of u, so a shock becomes a kink rather than a discontinuity — a smoother, easier target.",
    series: [[0, 0.02], [0.125, 0.105], [0.25, 0.122], [0.5, 0.237]], image: null, compareId: "hyperbolic_v4" },

  { id: "parabolic", stage: "parabolic", name: "Heat equation (control case)", tagline: "Same architecture, graph, and one-step recipe as v1 — on a dissipative PDE instead of Burgers'.",
    status: "pass",
    metrics: [
      { label: "rel. L2 @ 25%", value: "0.053" },
      { label: "rel. L2 @ 50%", value: "0.057" },
      { label: "rel. L2 @ final", value: "0.070" },
      { label: "max |pred|", value: "2.97" },
      { label: "diverged", value: "0 / 15" },
    ],
    checks: [
      { label: "One-step training alone is rollout-stable", pass: true },
      { label: "Error stays flat over 100 steps (no growth trend)", pass: true },
      { label: "No divergence across all 15 test trajectories", pass: true },
    ],
    note: "Confirms the hypothesis: diffusion is dissipative, so small per-step errors get smoothed away rather than amplified. Lands 9x lower final error than hyperbolic v4.",
    series: [[0, 0.02], [0.125, 0.053], [0.25, 0.057], [0.5, 0.070]], image: null, compareId: "hyperbolic_v4" },
];

function fmtVal(v) {
  return v >= 1000 ? v.toExponential(1).replace("e+", "e") : String(v);
}

function buildChart(cp, all, showOverlay) {
  const W = 640, H = 260, left = 40, right = 620, top = 20, bottom = 230;
  const allVals = cp.series.map((p) => p[1]);
  let overlaySeries = null, base = null;
  if (showOverlay !== false && cp.compareId) {
    base = all.find((c) => c.id === cp.compareId);
    if (base) overlaySeries = base.series;
  }
  if (overlaySeries) allVals.push(...overlaySeries.map((p) => p[1]));
  const minV = Math.max(0.01, Math.min(...allVals) * 0.6);
  const maxV = Math.max(...allVals) * 1.8;
  const logMin = Math.log10(minV), logMax = Math.log10(maxV);
  const xOf = (t) => left + (t / 0.5) * (right - left);
  const yOf = (v) => bottom - ((Math.log10(Math.max(v, minV)) - logMin) / (logMax - logMin)) * (bottom - top);
  const toPath = (series) => series.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p[0]).toFixed(1)} ${yOf(p[1]).toFixed(1)}`).join(" ");
  const gridLines = [];
  const decadeStart = Math.floor(logMin), decadeEnd = Math.ceil(logMax);
  for (let d = decadeStart; d <= decadeEnd; d++) {
    const v = Math.pow(10, d);
    if (v < minV || v > maxV) continue;
    gridLines.push({ y: yOf(v), label: v >= 1 ? (v >= 1000 ? v.toExponential(0).replace("e+", "e") : String(v)) : v.toFixed(2) });
  }
  const xTicks = [0, 0.25, 0.5].map((t) => ({ x: xOf(t), label: "t=" + t }));
  return {
    W, H, mainPath: toPath(cp.series), overlayPath: overlaySeries ? toPath(overlaySeries) : "",
    hasOverlay: !!overlaySeries, overlayLabel: base ? base.name.replace(/^.*—\s*/, "") : "",
    mainColor: COLORS[cp.status].fg, points: cp.series.map((p) => ({ x: xOf(p[0]), y: yOf(p[1]) })),
    gridLines, xTicks,
  };
}

function ImageDrop({ label }) {
  const [src, setSrc] = useState(null);
  const inputRef = useRef(null);
  const onFiles = (files) => {
    const f = files && files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (e) => setSrc(e.target.result);
    reader.readAsDataURL(f);
  };
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 600, color: "oklch(0.4 0.01 250)", marginBottom: 6 }}>{label}</div>
      <div
        onClick={() => inputRef.current && inputRef.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); onFiles(e.dataTransfer.files); }}
        style={{
          width: "100%", height: 160, borderRadius: 8, cursor: "pointer", overflow: "hidden",
          border: "1px dashed oklch(0.82 0.01 250)", background: src ? `center/cover no-repeat url(${src})` : "oklch(0.97 0.004 250)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        {!src && <span style={{ fontSize: 11, fontFamily: "IBM Plex Mono, monospace", color: "oklch(0.6 0.01 250)" }}>drop image</span>}
      </div>
      <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => onFiles(e.target.files)} />
    </div>
  );
}

/**
 * Drop-in GNN-PDE rollout validation dashboard.
 * Props: checkpoints (array; if omitted, fetches DASHBOARD_DATA_URL and
 *        falls back to FALLBACK_CHECKPOINTS above if that fetch fails),
 *        defaultCheckpoint (id string), showOverlay (bool), strictMode (bool, warn counts as fail).
 */
export default function GNNPDEValidationUI({ checkpoints: checkpointsProp, defaultCheckpoint = "hyperbolic_cole_hopf_v2", showOverlay = true, strictMode = false }) {
  const [fetchedCheckpoints, setFetchedCheckpoints] = useState(null);
  const [dataSource, setDataSource] = useState("loading"); // 'loading' | 'computed' | 'fallback'
  const [selectedId, setSelectedId] = useState(defaultCheckpoint);
  const [verifying, setVerifying] = useState(false);
  const [lastVerified, setLastVerified] = useState(null);

  const loadDashboardData = () => {
    setVerifying(true);
    fetch(DASHBOARD_DATA_URL, { cache: "no-store" })
      .then((res) => { if (!res.ok) throw new Error(`${res.status} ${res.statusText}`); return res.json(); })
      .then((data) => { setFetchedCheckpoints(data); setDataSource("computed"); })
      .catch(() => { setDataSource("fallback"); })
      .finally(() => { setVerifying(false); setLastVerified(Date.now()); });
  };

  // Only auto-fetch if the caller didn't hand us data directly via props.
  useEffect(() => { if (!checkpointsProp) loadDashboardData(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const checkpoints = checkpointsProp || fetchedCheckpoints || FALLBACK_CHECKPOINTS;
  const effectiveSource = checkpointsProp ? "prop" : dataSource;

  const selectedCp = checkpoints.find((c) => c.id === selectedId) || checkpoints[0];
  const c = COLORS[selectedCp.status];
  const chart = selectedCp.series ? buildChart(selectedCp, checkpoints, showOverlay) : null;

  const passCount = checkpoints.filter((k) => k.status === "pass").length;
  const warnCount = checkpoints.filter((k) => k.status === "warn").length;
  const failCount = checkpoints.filter((k) => k.status === "fail").length + (strictMode ? warnCount : 0);

  const groups = [
    { label: "ELLIPTIC · POISSON", items: checkpoints.filter((k) => k.stage === "elliptic") },
    { label: "HYPERBOLIC · BURGERS'", items: checkpoints.filter((k) => k.stage === "hyperbolic") },
    { label: "PARABOLIC · HEAT (CONTROL)", items: checkpoints.filter((k) => k.stage === "parabolic") },
  ];

  const displayName = selectedCp.name.replace(/^Burgers'\s*—\s*/, "");

  return (
    <div style={{ display: "flex", flexWrap: "wrap", minHeight: "100vh", fontFamily: "'Helvetica Neue',Helvetica,Arial,sans-serif", color: "oklch(0.22 0.01 250)", background: "oklch(0.98 0.003 250)" }}>
      <div style={{ flex: "0 0 280px", minWidth: 260, borderRight: "1px solid oklch(0.9 0.005 250)", padding: "28px 22px", display: "flex", flexDirection: "column", gap: 22, background: "oklch(1 0 0)" }}>
        <div>
          <h1 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 19, fontWeight: 600, margin: "0 0 6px", letterSpacing: "-0.01em" }}>GNN·PDE Validation</h1>
          <p style={{ fontSize: 12, lineHeight: 1.5, color: "oklch(0.5 0.01 250)", margin: 0 }}>Autoregressive rollout checks against classical reference solvers, per training stage.</p>
          <p style={{ fontSize: 10, fontFamily: "IBM Plex Mono, monospace", color: effectiveSource === "computed" ? "oklch(0.5 0.12 160)" : "oklch(0.6 0.08 60)", margin: "8px 0 0 0" }}>
            {effectiveSource === "loading" && "loading validator/results/dashboard_data.json…"}
            {effectiveSource === "computed" && "● live: computed by the validator"}
            {effectiveSource === "fallback" && "○ offline sample data (dashboard_data.json not found — run the validator)"}
            {effectiveSource === "prop" && "● data supplied by caller"}
          </p>
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[["pass", passCount], ["warn", strictMode ? 0 : warnCount], ["fail", failCount]].map(([k, n]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, fontFamily: "IBM Plex Mono, monospace", padding: "4px 9px", borderRadius: 5, background: COLORS[k].bg, color: COLORS[k].fg }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS[k].dot }} />{n} {k}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {groups.map((g) => (
            <div key={g.label} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", color: "oklch(0.6 0.01 250)", padding: "0 4px" }}>{g.label}</div>
              {g.items.map((item) => (
                <div key={item.id} onClick={() => setSelectedId(item.id)} style={{ cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, padding: "9px 10px", borderRadius: 8, background: item.id === selectedId ? "oklch(0.955 0.01 220)" : "transparent" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                    <span style={{ width: 7, height: 7, borderRadius: "50%", flexShrink: 0, background: COLORS[item.status].dot }} />
                    <span style={{ fontSize: 13, fontWeight: 500, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{item.name.replace(/^Burgers'\s*—\s*/, "")}</span>
                  </div>
                  <span style={{ fontSize: 11, fontFamily: "IBM Plex Mono, monospace", color: "oklch(0.5 0.01 250)", flexShrink: 0 }}>{item.metrics.find((m) => /final/.test(m.label))?.value ?? ""}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div style={{ marginTop: "auto", fontSize: 11, color: "oklch(0.65 0.01 250)", lineHeight: 1.5, paddingTop: 8, borderTop: "1px solid oklch(0.93 0.005 250)" }}>
          Reference: FD / spectral solvers.<br />Validation metric: relative L2 error of GNN rollout vs. reference trajectory.
        </div>
      </div>

      <div style={{ flex: "1 1 380px", minWidth: 0, padding: "36px 44px", display: "flex", flexDirection: "column", gap: 26, maxWidth: 980 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div style={{ maxWidth: 560 }}>
            <h2 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 21, fontWeight: 600, margin: 0, letterSpacing: "-0.01em", lineHeight: 1.35 }}>{displayName}</h2>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 600, fontFamily: "IBM Plex Mono, monospace", padding: "4px 10px", borderRadius: 20, background: c.bg, color: c.fg, width: "fit-content", marginTop: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: c.fg }} />{selectedCp.status.toUpperCase()}
            </div>
            <p style={{ fontSize: 13, color: "oklch(0.48 0.01 250)", margin: "10px 0 0 0", lineHeight: 1.5 }}>{selectedCp.tagline}</p>
          </div>
          <button onClick={loadDashboardData} disabled={!!checkpointsProp} title={checkpointsProp ? "data was supplied via props, not fetched" : "re-fetch validator/results/dashboard_data.json (run validator.build_dashboard_data first to refresh it)"}
            style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px", borderRadius: 8, border: "1px solid oklch(0.85 0.01 250)", background: "oklch(1 0 0)", fontSize: 13, fontWeight: 500, color: checkpointsProp ? "oklch(0.7 0.01 250)" : "oklch(0.25 0.01 250)", cursor: checkpointsProp ? "default" : "pointer", whiteSpace: "nowrap" }}>
            {verifying && <span style={{ width: 13, height: 13, borderRadius: "50%", border: "2px solid oklch(0.85 0.01 250)", borderTopColor: "oklch(0.4 0.01 250)", display: "inline-block", animation: "spin 0.7s linear infinite" }} />}
            {verifying ? "Loading…" : lastVerified ? "Reloaded ✓" : "Reload results"}
          </button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", gap: 12 }}>
          {selectedCp.metrics.map((m) => (
            <div key={m.label} style={{ background: "oklch(1 0 0)", border: "1px solid oklch(0.91 0.005 250)", borderRadius: 10, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "oklch(0.55 0.01 250)", marginBottom: 6 }}>{m.label}</div>
              <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 19, fontWeight: 600, color: "oklch(0.25 0.01 250)" }}>{m.value}</div>
            </div>
          ))}
        </div>

        {chart && (
          <div style={{ background: "oklch(1 0 0)", border: "1px solid oklch(0.91 0.005 250)", borderRadius: 12, padding: "20px 22px" }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4, flexWrap: "wrap", gap: 6 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>Rollout error vs. time</div>
              <div style={{ fontSize: 11, color: "oklch(0.55 0.01 250)" }}>log scale · relative L2 vs. spectral/FD reference</div>
            </div>
            <svg viewBox={`0 0 ${chart.W} ${chart.H}`} style={{ width: "100%", height: "auto", display: "block", marginTop: 8 }}>
              {chart.gridLines.map((gl, i) => (
                <g key={i}>
                  <line x1={40} x2={620} y1={gl.y} y2={gl.y} stroke="oklch(0.93 0.005 250)" strokeWidth={1} />
                  <text x={34} y={gl.y} textAnchor="end" dominantBaseline="middle" fontSize={9} fill="oklch(0.55 0.01 250)" fontFamily="IBM Plex Mono, monospace">{gl.label}</text>
                </g>
              ))}
              <line x1={40} x2={620} y1={230} y2={230} stroke="oklch(0.8 0.005 250)" strokeWidth={1} />
              {chart.xTicks.map((xt, i) => (
                <text key={i} x={xt.x} y={245} textAnchor="middle" fontSize={9} fill="oklch(0.55 0.01 250)" fontFamily="IBM Plex Mono, monospace">{xt.label}</text>
              ))}
              {chart.hasOverlay && <path d={chart.overlayPath} fill="none" stroke="oklch(0.7 0.01 250)" strokeWidth={2} strokeDasharray="4,4" />}
              <path d={chart.mainPath} fill="none" stroke={chart.mainColor} strokeWidth={2.5} />
              {chart.points.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={3.5} fill={chart.mainColor} />)}
            </svg>
            <div style={{ display: "flex", gap: 16, marginTop: 6, fontSize: 11, color: "oklch(0.5 0.01 250)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 14, height: 2, background: chart.mainColor, display: "inline-block" }} />{displayName}</div>
              {chart.hasOverlay && <div style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 14, height: 0, borderTop: "2px dashed oklch(0.7 0.01 250)", display: "inline-block" }} />{chart.overlayLabel} (baseline)</div>}
            </div>
          </div>
        )}

        <div style={{ background: "oklch(1 0 0)", border: "1px solid oklch(0.91 0.005 250)", borderRadius: 12, padding: "20px 22px" }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Spatial field comparison</div>
          <div style={{ fontSize: 11, color: "oklch(0.55 0.01 250)", marginBottom: 14 }}>Drop your own frames from this checkpoint's evaluation — one 2D field or 1D snapshot per panel.</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 14 }}>
            <ImageDrop label="GNN prediction" />
            <ImageDrop label="Reference / ground truth" />
            <ImageDrop label="Error residual" />
          </div>
        </div>

        <div style={{ background: "oklch(1 0 0)", border: "1px solid oklch(0.91 0.005 250)", borderRadius: 12, padding: "18px 20px" }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Validation checks</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
            {selectedCp.checks.map((ck, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
                <span style={{ width: 16, height: 16, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "oklch(1 0 0)", background: ck.pass ? COLORS.pass.dot : COLORS.fail.dot }}>{ck.pass ? "✓" : "✕"}</span>
                <span style={{ color: "oklch(0.3 0.01 250)" }}>{ck.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ borderRadius: 12, padding: "18px 20px", background: "oklch(0.965 0.006 250)", border: "1px solid oklch(0.91 0.005 250)" }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "oklch(0.4 0.01 250)" }}>Read on this result</div>
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "oklch(0.35 0.01 250)", margin: 0, whiteSpace: "pre-wrap" }}>{selectedCp.note}</p>
        </div>
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}
