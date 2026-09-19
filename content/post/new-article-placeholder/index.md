---
title: "Placeholder title"
description: "Placeholder description"
slug: placeholder-slug
date: 2026-09-11
image: cover.png
math: true
categories:
    - Networks_Graphs
tags:
    - Network Analysis
    - PCA
    - Data Visualization
keywords:
    - French National Assembly
    - force-directed graph
    - PCA
    - interactive visualization
    - voting behavior
    - cosine similarity
weight: 2
toc: true
draft: true
---

<style>
.anviz { --anviz-bg: #1b1d23; --anviz-tip: #272a31; --anviz-fg: rgba(255,255,255,.92); --anviz-muted: rgba(255,255,255,.62); --anviz-border: rgba(255,255,255,.14); --anviz-hover: rgba(255,255,255,.08); color-scheme: dark; background: var(--anviz-bg); color: var(--anviz-fg); border: 1px solid var(--anviz-border); box-shadow: 0 2px 12px rgba(0,0,0,.15); border-radius: 10px; padding: 12px; margin: 1.5rem 0; font-size: 14px; line-height: 1.4; }
.anviz-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
.anviz-controls label { display: inline-flex; gap: 6px; align-items: center; color: var(--anviz-muted); font-size: 13px; }
.anviz-controls select, .anviz-controls button, .anviz-controls input { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--anviz-border); background: var(--anviz-bg); color: var(--anviz-fg); cursor: pointer; }
.anviz-controls input { cursor: text; min-width: 160px; }
.anviz-controls button:hover { background: var(--anviz-hover); }
.anviz [hidden] { display: none !important; }
.anviz-toggle { display: inline-flex; }
.anviz-group { display: inline-flex; gap: 6px; align-items: center; }
.anviz-toggle button { border-radius: 0; }
.anviz-toggle button:first-child { border-radius: 6px 0 0 6px; }
.anviz-toggle button:last-child { border-radius: 0 6px 6px 0; }
.anviz-toggle button + button { margin-left: -1px; }
.anviz-toggle button[aria-pressed="true"] { background: var(--anviz-fg); color: var(--anviz-bg); }
.anviz-status { margin-left: auto; color: var(--anviz-muted); font-size: 13px; font-variant-numeric: tabular-nums; }
.anviz-stage { position: relative; width: 100%; }
.anviz-stage canvas { display: block; width: 100%; aspect-ratio: 5 / 4; max-height: 640px; cursor: crosshair; }
.anviz-tip { position: absolute; pointer-events: none; z-index: 2; background: var(--anviz-tip); color: var(--anviz-fg); border: 1px solid var(--anviz-border); box-shadow: 0 4px 14px rgba(0,0,0,.45); border-radius: 8px; padding: 8px 10px; font-size: 12.5px; max-width: 250px; }
.anviz-tip b { font-size: 13.5px; }
.anviz-tip-sub { margin-top: 6px; color: var(--anviz-muted); font-size: 11.5px; }
.anviz-muted { color: var(--anviz-muted); font-variant-numeric: tabular-nums; }
.anviz .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.anviz-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 10px; font-size: 12.5px; color: var(--anviz-muted); }
.anviz-legend button { font: inherit; background: none; border: 0; padding: 0; color: inherit; cursor: pointer; }
.anviz-legend button.off { opacity: .35; text-decoration: line-through; }
.anviz-caption { margin-top: 6px; font-size: 12px; color: var(--anviz-muted); }
.anviz-plot { width: 100%; height: 560px; }
.anviz-headline { margin: 2px 0 8px; font-size: 13px; }
.anviz-headline b { font-size: 14px; }
.anviz-split { display: grid; grid-template-columns: 1.55fr 1fr; gap: 10px; align-items: start; }
.anviz-split .anviz-plot { height: 460px; }
.anviz-split .anviz-plot-bars { height: 460px; }
@media (max-width: 700px) { .anviz-split { grid-template-columns: 1fr; } .anviz-split .anviz-plot, .anviz-split .anviz-plot-bars { height: 340px; } }
@media (max-width: 600px) { .anviz-plot { height: 420px; } .anviz-status { margin-left: 0; width: 100%; } .anviz-controls input { min-width: 0; flex: 1 1 140px; } }
</style>

<script>
window.anviz = (function () {
  const bus = new EventTarget();
  const state = { focus: null, pinned: null, leg: null, source: null };
  const files = {};
  const PLOTLY_SRC = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";
  let plotly = null;
  const api = {
    state,
    on(name, fn) { bus.addEventListener(name, fn); },
    // The MP highlighted everywhere: the one under the pointer, or the one last searched for.
    focused() { return state.focus || state.pinned; },
    // `source` is the widget the pointer is in: it already draws its own hover, so it skips the redraw.
    hover(id, source) {
      if (state.focus === (id || null)) return;
      state.focus = id || null; state.source = source || null;
      bus.dispatchEvent(new Event("focus"));
    },
    pin(id) { state.pinned = id || null; state.focus = null; state.source = null; bus.dispatchEvent(new Event("focus")); },
    legislature(v) { if (!v || v === state.leg) return; state.leg = v; bus.dispatchEvent(new Event("legislature")); },
    // "no-cache" revalidates with the server: after a data update, a stale cached file would not match the page's code.
    json(path) { if (!files[path]) files[path] = fetch(path, { cache: "no-cache" }).then(r => r.json()); return files[path]; },
    plotly() {
      if (!plotly) {
        plotly = window.Plotly ? Promise.resolve() : new Promise((resolve, reject) => {
          const s = document.createElement("script");
          s.src = PLOTLY_SRC; s.onload = () => resolve(); s.onerror = reject;
          document.head.appendChild(s);
        });
      }
      return plotly;
    },
    esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); },
    cssVar(el, name, fallback) { return getComputedStyle(el).getPropertyValue(name).trim() || fallback; },
    pct(v) { return v === null || v === undefined ? "n/a" : Math.round(v * 100) + "%"; },
    // Type-ahead over the MPs of the widget; picking one pins it in every figure of the page.
    search(input, listId, items) {
      const list = document.getElementById(listId);
      list.innerHTML = items.map(m => '<option value="' + api.esc(m.name) + " (" + api.esc(m.group) + ')"></option>').join("");
      const run = () => {
        const value = input.value.trim().toLowerCase();
        if (!value) return api.pin(null);
        const hit = items.find(m => (m.name + " (" + m.group + ")").toLowerCase() === value) ||
          items.find(m => m.name.toLowerCase() === value) ||
          (value.length >= 3 ? items.find(m => m.name.toLowerCase().includes(value)) : null);
        if (hit) api.pin(hit.id);
      };
      input.addEventListener("change", run);
      input.addEventListener("keydown", e => {
        if (e.key === "Enter") { e.preventDefault(); run(); }
        if (e.key === "Escape") { input.value = ""; api.pin(null); }
      });
    },
  };
  return api;
})();
</script>

*Draft. Every figure below is built from the same data and metrics as the [previous article](/p/networks-analysis/) (NosDéputés.fr votes, cosine similarity, top-5 neighbors graph, standardized PCA). The model's framework is the same as in the previous article*

## 0. Why?

After the release of the first article [previous article](/p/networks-analysis/) I got interesting discussions with friends and colleagues. Some advised me to make the article more intuitive. Thus I decided to propose some interactive plots and visualisation, to let the reader manipulate and taking conclusions by actions.

## 1. From random thrown to a map

As in the previous article, MPs are thrown randomly into the plot and exerces attractive or repulsive forces on others depending on voting similarity. A force-directed layout has no natural orientation, so the converged map is finally rotated (and mirrored if needed) to line up with the first two principal components of section 2: every run ends in the same orientation.

Hover over a dot to see the MP's name, group, closest voting neighbors and how often they vote. Click a group in the legend to hide or show it.

The buttons next to the legislature change k, the number of most similar colleagues each MP is linked to.

<div class="anviz" id="an-network">
<div class="anviz-controls">
<select data-role="leg" aria-label="Legislature"><option value="14">14th legislature (2012-2017)</option><option value="15">15th legislature (2017-2022)</option><option value="16" selected>16th legislature (2022-2024)</option></select>
<span class="anviz-group"><span class="anviz-muted">Neighbors (k)</span><span class="anviz-toggle" role="group" aria-label="Neighbors per MP"><button type="button" data-k="3" aria-pressed="false">3</button><button type="button" data-k="5" aria-pressed="true">5</button><button type="button" data-k="10" aria-pressed="false">10</button><button type="button" data-k="15" aria-pressed="false">15</button></span></span>
<button type="button" data-action="play">Play</button>
<button type="button" data-action="step">Step +1</button>
<button type="button" data-action="restart">New random start</button>
<input type="search" data-role="search" list="an-network-mps" placeholder="Find an MP" aria-label="Find an MP">
<datalist id="an-network-mps"></datalist>
<span class="anviz-status">Loading…</span>
</div>
<div class="anviz-stage"><canvas aria-label="Force-directed network of MPs"></canvas><div class="anviz-tip" hidden></div></div>
<div class="anviz-legend"></div>
<div class="anviz-caption"></div>
</div>

<script>
(function () {
  const A = window.anviz;
  const root = document.getElementById("an-network");
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const tip = root.querySelector(".anviz-tip");
  const statusEl = root.querySelector(".anviz-status");
  const legendEl = root.querySelector(".anviz-legend");
  const captionEl = root.querySelector(".anviz-caption");
  const legSel = root.querySelector('[data-role="leg"]');
  const playBtn = root.querySelector('[data-action="play"]');
  const K = 0.15, ITERATIONS = 400, ALIGN_MS = 900;
  const stepMs = i => Math.max(16, 140 * Math.pow(0.9, i));
  let d = null, n = 0, adj = null, nbrs = null, colors = null, ref = null, links = [], kNN = 5, byId = new Map();
  let pos = null, prev = null, scr = null, temp = 0, dtemp = 0, iter = 0, done = false;
  const hidden = new Set();
  let playing = false, animating = false, stepStart = 0, stepDur = 0, raf = 0, hover = -1, focusIdx = -1, started = false;
  let align = null, alignA = 0, alignStart = 0;

  const esc = A.esc;
  const cssVar = (name, fallback) => A.cssVar(root, name, fallback);
  const isHidden = i => hidden.has(d.nodes[i].group);
  const active = () => {
    const i = hover >= 0 ? hover : focusIdx;
    return i >= 0 && isHidden(i) ? -1 : i;
  };

  function setData(data) {
    d = data; n = data.nodes.length; hidden.clear();
    const groupColor = Object.fromEntries(data.groups.map(g => [g.name, g.color]));
    colors = data.nodes.map(v => groupColor[v.group]);
    ref = data.nodes.map(v => [v.pc[0], -v.pc[1]]);
    byId = new Map(data.nodes.map((v, i) => [v.id, i]));
    buildLegend();
    A.search(root.querySelector('[data-role="search"]'), "an-network-mps", data.nodes);
    buildGraph();
    syncFocus();
  }

  function buildLegend() {
    legendEl.innerHTML = d.groups.map(g =>
      '<button type="button" data-group="' + esc(g.name) + '"' + (hidden.has(g.name) ? ' class="off"' : "") +
      '><span class="dot" style="background:' + g.color + '"></span>' + esc(g.name) + " (" + g.count + ")</button>").join("");
  }

  legendEl.addEventListener("click", e => {
    const btn = e.target.closest("button[data-group]");
    if (!btn || !d) return;
    const g = btn.dataset.group;
    if (hidden.has(g)) hidden.delete(g); else hidden.add(g);
    buildLegend(); refreshTip();
  });

  // Each link carries the smallest k at which it belongs to the k-NN graph.
  function buildGraph() {
    links = d.links.filter(l => l[3] <= kNN);
    adj = new Float64Array(n * n);
    nbrs = Array.from({ length: n }, () => []);
    for (const [i, j, w] of links) {
      adj[i * n + j] = w; adj[j * n + i] = w;
      nbrs[i].push([j, w]); nbrs[j].push([i, w]);
    }
    nbrs.forEach(a => a.sort((x, y) => y[1] - x[1]));
    captionEl.textContent = n + " MPs, " + links.length + " edges (each MP linked to their " + kNN + " most similar colleagues), " + d.n_scrutins + " ballots.";
    hover = -1; tip.hidden = true;
    restart();
  }

  function restart() {
    pos = new Float64Array(2 * n);
    for (let i = 0; i < 2 * n; i++) pos[i] = Math.random();
    prev = pos.slice();
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      minX = Math.min(minX, pos[2 * i]); maxX = Math.max(maxX, pos[2 * i]);
      minY = Math.min(minY, pos[2 * i + 1]); maxY = Math.max(maxY, pos[2 * i + 1]);
    }
    temp = 0.1 * Math.max(maxX - minX, maxY - minY);
    dtemp = temp / (ITERATIONS + 1);
    iter = 0; done = false; animating = false; align = null; alignA = 0;
    setPlaying(false); updateStatus(); schedule();
  }

  // Best rotation (optionally after a mirror) of the final layout onto (PC1, PC2), closed-form 2D Procrustes.
  function startAlign(now) {
    let cx = 0, cy = 0, rx = 0, ry = 0;
    for (let i = 0; i < n; i++) { cx += pos[2 * i]; cy += pos[2 * i + 1]; rx += ref[i][0]; ry += ref[i][1]; }
    cx /= n; cy /= n; rx /= n; ry /= n;
    let a = 0, b = 0, af = 0, bf = 0;
    for (let i = 0; i < n; i++) {
      const x1 = pos[2 * i] - cx, x2 = pos[2 * i + 1] - cy, y1 = ref[i][0] - rx, y2 = ref[i][1] - ry;
      a += x1 * y1 + x2 * y2; b += x1 * y2 - x2 * y1;
      af += -x1 * y1 + x2 * y2; bf += -x1 * y2 - x2 * y1;
    }
    const flip = Math.hypot(af, bf) > Math.hypot(a, b);
    align = { flip, theta: flip ? Math.atan2(bf, af) : Math.atan2(b, a) };
    alignA = 0; alignStart = now;
  }

  // One iteration of networkx's _fruchterman_reingold (weighted adjacency, clipped distances, linear cooling).
  function frStep() {
    const disp = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const xi = pos[2 * i], yi = pos[2 * i + 1], row = i * n;
      let sx = 0, sy = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = xi - pos[2 * j], dy = yi - pos[2 * j + 1];
        let dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 0.01) dist = 0.01;
        const f = (K * K) / (dist * dist) - (adj[row + j] * dist) / K;
        sx += dx * f; sy += dy * f;
      }
      disp[2 * i] = sx; disp[2 * i + 1] = sy;
    }
    prev = pos.slice();
    let moved = 0;
    for (let i = 0; i < n; i++) {
      let len = Math.hypot(disp[2 * i], disp[2 * i + 1]);
      if (len < 0.01) len = 0.1;
      const mx = (disp[2 * i] * temp) / len, my = (disp[2 * i + 1] * temp) / len;
      pos[2 * i] += mx; pos[2 * i + 1] += my;
      moved += mx * mx + my * my;
    }
    temp -= dtemp; iter++;
    if (iter >= ITERATIONS || Math.sqrt(moved) / n < 1e-4) done = true;
  }

  function advance() {
    frStep();
    stepStart = performance.now(); stepDur = stepMs(iter); animating = true;
    updateStatus(); schedule();
  }

  function setPlaying(p) {
    playing = p;
    playBtn.textContent = p ? "Pause" : (done ? "Replay" : "Play");
  }

  function updateStatus() {
    statusEl.textContent = "Iteration " + iter + " / " + ITERATIONS + (done ? " (converged)" : "");
  }

  function schedule() { if (!raf) raf = requestAnimationFrame(frame); }

  function frame(now) {
    raf = 0;
    let a = 1;
    if (animating) {
      a = Math.max(0, Math.min(1, (now - stepStart) / stepDur));
      if (a >= 1) {
        animating = false;
        if (playing && !done) { advance(); a = 0; }
        else if (done) { setPlaying(false); startAlign(now); }
      }
    }
    if (align && alignA < 1) alignA = Math.max(0, Math.min(1, (now - alignStart) / ALIGN_MS));
    draw(a);
    if (animating || (align && alignA < 1)) schedule();
  }

  function draw(a) {
    if (!d) return;
    const W = canvas.clientWidth, H = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!W || !H) return;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const cur = new Float64Array(2 * n);
    for (let i = 0; i < 2 * n; i++) cur[i] = prev[i] + (pos[i] - prev[i]) * a;
    if (align) {
      const th = align.theta * alignA, c = Math.cos(th), s = Math.sin(th), fx = align.flip ? 1 - 2 * alignA : 1;
      let mx = 0, my = 0;
      for (let i = 0; i < n; i++) { mx += cur[2 * i]; my += cur[2 * i + 1]; }
      mx /= n; my /= n;
      for (let i = 0; i < n; i++) {
        const x = (cur[2 * i] - mx) * fx, y = cur[2 * i + 1] - my;
        cur[2 * i] = c * x - s * y; cur[2 * i + 1] = s * x + c * y;
      }
    }
    for (let i = 0; i < n; i++) {
      minX = Math.min(minX, cur[2 * i]); maxX = Math.max(maxX, cur[2 * i]);
      minY = Math.min(minY, cur[2 * i + 1]); maxY = Math.max(maxY, cur[2 * i + 1]);
    }
    const pad = 16;
    const s = Math.min((W - 2 * pad) / (maxX - minX || 1), (H - 2 * pad) / (maxY - minY || 1));
    const ox = (W - s * (maxX + minX)) / 2, oy = (H - s * (maxY + minY)) / 2;
    scr = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) { scr[2 * i] = ox + s * cur[2 * i]; scr[2 * i + 1] = oy + s * cur[2 * i + 1]; }

    const act = active();
    ctx.lineWidth = 0.6;
    ctx.strokeStyle = cssVar("--anviz-muted", "#9a9a9a");
    ctx.globalAlpha = act >= 0 ? 0.06 : 0.16;
    ctx.beginPath();
    for (const [i, j] of links) {
      if (isHidden(i) || isHidden(j)) continue;
      ctx.moveTo(scr[2 * i], scr[2 * i + 1]); ctx.lineTo(scr[2 * j], scr[2 * j + 1]);
    }
    ctx.stroke();

    let near = null;
    if (act >= 0) {
      near = new Set(nbrs[act].map(x => x[0]));
      ctx.globalAlpha = 0.9; ctx.lineWidth = 1.4; ctx.strokeStyle = colors[act];
      ctx.beginPath();
      for (const [j] of nbrs[act]) {
        if (isHidden(j)) continue;
        ctx.moveTo(scr[2 * act], scr[2 * act + 1]); ctx.lineTo(scr[2 * j], scr[2 * j + 1]);
      }
      ctx.stroke();
    }

    const r = Math.max(2.4, Math.min(4, W / 200));
    ctx.lineWidth = 0.6; ctx.strokeStyle = cssVar("--anviz-bg", "#1b1d23");
    for (let i = 0; i < n; i++) {
      if (isHidden(i)) continue;
      ctx.globalAlpha = act < 0 || i === act || near.has(i) ? 0.95 : 0.2;
      ctx.fillStyle = colors[i];
      ctx.beginPath(); ctx.arc(scr[2 * i], scr[2 * i + 1], r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    if (act >= 0) {
      ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = cssVar("--anviz-fg", "#fff");
      ctx.beginPath(); ctx.arc(scr[2 * act], scr[2 * act + 1], r + 3, 0, 2 * Math.PI); ctx.stroke();
      placeTip();
    }
    ctx.globalAlpha = 1;
  }

  function showTip(i) {
    const node = d.nodes[i];
    const top = nbrs[i].slice(0, 3).map(([j, w]) => esc(d.nodes[j].name) + ' <span class="anviz-muted">' + w.toFixed(2) + "</span>").join("<br>");
    tip.innerHTML = "<b>" + esc(node.name) + '</b><br><span class="dot" style="background:' + colors[i] + '"></span>' + esc(node.group) +
      '<div class="anviz-tip-sub">Voted in ' + A.pct(node.part) + " of the ballots held during their mandate</div>" +
      (top ? '<div class="anviz-tip-sub">Closest voting neighbors (cosine)</div>' + top : "");
    tip.hidden = false;
  }

  function placeTip() {
    const i = active();
    const x = scr[2 * i], y = scr[2 * i + 1];
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 12, top = y + 12;
    if (left + tw > canvas.clientWidth) left = x - tw - 12;
    if (top + th > canvas.clientHeight) top = y - th - 12;
    tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px";
  }

  function pick(e) {
    if (!scr) return;
    const rect = canvas.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = -1, bestD = 100;
    for (let i = 0; i < n; i++) {
      if (isHidden(i)) continue;
      const dx = scr[2 * i] - mx, dy = scr[2 * i + 1] - my, dd = dx * dx + dy * dy;
      if (dd < bestD) { bestD = dd; best = i; }
    }
    if (best !== hover) {
      hover = best;
      A.hover(best >= 0 ? d.nodes[best].id : null, root.id);
      refreshTip();
    }
  }
  canvas.addEventListener("pointermove", pick);
  canvas.addEventListener("pointerdown", pick);
  canvas.addEventListener("pointerleave", e => {
    if (e.pointerType === "touch") return;
    hover = -1; A.hover(null, root.id); refreshTip();
  });

  function refreshTip() {
    const act = active();
    if (act >= 0) showTip(act); else tip.hidden = true;
    schedule();
  }

  function syncFocus() {
    const id = A.focused();
    focusIdx = id && byId.has(id) ? byId.get(id) : -1;
    refreshTip();
  }
  A.on("focus", () => { if (A.state.source !== root.id) syncFocus(); });

  function play() { started = true; setPlaying(true); if (!animating) advance(); }

  playBtn.addEventListener("click", () => {
    if (playing) { setPlaying(false); return; }
    if (done) restart();
    play();
  });
  root.querySelector('[data-action="step"]').addEventListener("click", () => {
    started = true; setPlaying(false);
    if (!done) advance();
  });
  root.querySelector('[data-action="restart"]').addEventListener("click", () => { restart(); play(); });

  function showLegislature(leg) {
    started = true; setPlaying(false);
    return A.json("network_L" + leg + ".json").then(data => { setData(data); play(); });
  }
  legSel.addEventListener("change", () => { A.legislature(legSel.value); showLegislature(legSel.value); });
  A.on("legislature", () => {
    if (A.state.leg === legSel.value) return;
    legSel.value = A.state.leg;
    showLegislature(A.state.leg);
  });

  root.querySelectorAll("[data-k]").forEach(b => b.addEventListener("click", () => {
    kNN = +b.dataset.k;
    root.querySelectorAll("[data-k]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (!d) return;
    started = true; buildGraph(); play();
  }));

  new ResizeObserver(schedule).observe(canvas);
  const io = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { io.disconnect(); if (!started) play(); }
  }, { threshold: 0.4 });

  A.json("network_L" + legSel.value + ".json").then(data => { A.legislature(legSel.value); setData(data); io.observe(root); })
    .catch(() => { statusEl.textContent = "Could not load the data."; });
})();
</script>

## 2. The voting space in 3D

The original article projected the MPs onto the first two principal components only. Here you can choose the legislature, switch between a 3D and a 2D view, and pick any of the first six components for each axis (for instance PC2 against PC3 in 2D). In 3D, drag to rotate and scroll to zoom. The percentage next to each axis is its share of explained variance. 

With the colors set to "Participation", the MPs who vote most often (pale) are the furthest out, and those who vote least (dark) are grouped in one area of the plot. In the vote matrix, an abstention and a missing vote are both 0. Each ballot is then centred and scaled before the PCA, so an MP with no recorded vote at all would not land at the origin of the plot but at a fixed point: PC1 = −9.6 in the 16th legislature and −13.2 in the 15th (dashed line below). The fewer ballots an MP votes on, the closer their coordinates are to that point, whatever they voted.

Two caveats. Participation is the share of public ballots (scrutins publics) where the MP’s vote was recorded; it is not attendance, since many votes in the chamber are taken by show of hands and not recorded individually. And some of the lowest values have institutional causes: in the 16th legislature, the three lowest are Yaël Braun-Pivet, President of the Assembly, and Carole Grandjean and Clément Beaune, members of the government from July 2022 to January 2024 (their participation only counts the periods when they sat).

<div class="anviz" id="an-pca">
<div class="anviz-controls">
<select data-role="leg" aria-label="Legislature"><option value="15">15th legislature (2017-2022)</option><option value="16" selected>16th legislature (2022-2024)</option></select>
<span class="anviz-toggle" role="group" aria-label="View"><button type="button" data-mode="3d" aria-pressed="true">3D</button><button type="button" data-mode="2d" aria-pressed="false">2D</button></span>
<label>X <select data-role="x"></select></label>
<label>Y <select data-role="y"></select></label>
<label data-role="z-label">Z <select data-role="z"></select></label>
<input type="search" data-role="search" list="an-pca-mps" placeholder="Find an MP" aria-label="Find an MP">
<datalist id="an-pca-mps"></datalist>
</div>
<div class="anviz-controls">
<span class="anviz-group"><span class="anviz-muted">Color</span><span class="anviz-toggle" role="group" aria-label="Color by"><button type="button" data-color="group" aria-pressed="true">Group</button><button type="button" data-color="part" aria-pressed="false">Participation</button></span></span>
<span class="anviz-muted">Presets:</span>
<button type="button" data-preset="1,2,3">3D: PC1 · PC2 · PC3</button>
<button type="button" data-preset="1,2">2D: PC1 · PC2</button>
</div>
<div class="anviz-plot"><span class="anviz-muted">Loading…</span></div>
<div class="anviz-legend"></div>
<div class="anviz-caption"></div>
</div>

<script>
(function () {
  const A = window.anviz;
  const root = document.getElementById("an-pca");
  const plot = root.querySelector(".anviz-plot");
  const legendEl = root.querySelector(".anviz-legend");
  const captionEl = root.querySelector(".anviz-caption");
  const sel = {};
  ["leg", "x", "y", "z"].forEach(k => { sel[k] = root.querySelector('[data-role="' + k + '"]'); });
  const zLabel = root.querySelector('[data-role="z-label"]');
  const axes = { x: "1", y: "2", z: "3" };
  let mode = "3d", colorBy = "group";
  const hidden = new Set();
  let data = null, ready = false, bound = false;

  const esc = A.esc;
  const cssVar = (name, fallback) => A.cssVar(root, name, fallback);
  const pcLabel = k => "PC" + k + " (" + (100 * data.explained_variance[k - 1]).toFixed(1) + "%)";

  function fillAxes() {
    const opts = [];
    for (let k = 1; k <= data.explained_variance.length; k++) opts.push('<option value="' + k + '">' + pcLabel(k) + "</option>");
    for (const key of ["x", "y", "z"]) { sel[key].innerHTML = opts.join(""); sel[key].value = axes[key]; }
  }

  function buildLegend() {
    legendEl.innerHTML = data.groups.map(g =>
      '<button type="button" data-group="' + esc(g.name) + '"' + (hidden.has(g.name) ? ' class="off"' : "") + '><span class="dot" style="background:' + g.color + '"></span>' + esc(g.name) + " (" + g.count + ")</button>").join("");
    captionEl.textContent = data.points.length + " MPs, " + data.n_scrutins + " ballots. Votes standardized per ballot before PCA, as in the original analysis.";
  }

  function render() {
    const x = +axes.x, y = +axes.y, z = +axes.z, is3d = mode === "3d";
    const text = cssVar("--anviz-fg", "#fff");
    const muted = cssVar("--anviz-muted", "#9a9a9a");
    const grid = cssVar("--anviz-border", "rgba(255,255,255,.14)");
    const bg = cssVar("--anviz-bg", "#1b1d23");
    const type = is3d ? "scatter3d" : "scatter";
    const shown = data.points.filter(p => !hidden.has(p.group));
    const coords = pts => {
      const t = { x: pts.map(p => p.pc[x - 1]), y: pts.map(p => p.pc[y - 1]) };
      if (is3d) t.z = pts.map(p => p.pc[z - 1]);
      return t;
    };
    const meta = pts => ({
      text: pts.map(p => p.name),
      customdata: pts.map(p => [p.group, A.pct(p.part), p.id]),
      hovertemplate: "<b>%{text}</b><br>%{customdata[0]}<br>voted in %{customdata[1]} of ballots<extra></extra>",
    });

    let traces;
    if (colorBy === "part") {
      const pts = shown.filter(p => p.part !== null);
      traces = [Object.assign({ type: type, mode: "markers", name: "Participation" }, coords(pts), meta(pts), {
        marker: {
          color: pts.map(p => p.part), colorscale: "Viridis", size: is3d ? 3.5 : 7, opacity: 0.9,
          colorbar: {
            title: { text: "Share of ballots", side: "right" }, tickformat: ".0%", thickness: 12, len: 0.7,
            outlinewidth: 0, tickfont: { color: muted }, titlefont: { color: muted },
          },
        },
      })];
    } else {
      traces = data.groups.map(g => {
        const pts = data.points.filter(p => p.group === g.name);
        return Object.assign({ type: type, mode: "markers", name: g.name, visible: !hidden.has(g.name) }, coords(pts), meta(pts), {
          marker: { color: g.color, size: is3d ? 3.5 : 7, opacity: 0.85, line: { width: is3d ? 0 : 0.5, color: bg } },
        });
      });
    }

    const focus = data.points.find(p => p.id === A.focused());
    if (focus) {
      traces.push(Object.assign({ type: type, mode: "markers+text", name: "selected", hoverinfo: "skip", showlegend: false },
        coords([focus]), {
          text: [focus.name], textposition: "top center", textfont: { color: text, size: 12 },
          marker: { size: is3d ? 9 : 16, symbol: "circle-open", color: text, line: { color: text, width: 2 } },
        }));
    }

    const axis = k => ({ title: { text: pcLabel(k) }, gridcolor: grid, zerolinecolor: grid, color: text });
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: text, size: 12 }, showlegend: false,
      modebar: { bgcolor: "rgba(0,0,0,0)", color: muted, activecolor: text },
      hoverlabel: { font: { size: 13 } },
      uirevision: [mode, x, y, is3d ? z : 0].join("-"),
    };
    if (is3d) {
      layout.margin = { l: 0, r: 0, t: 0, b: 0 };
      layout.scene = {
        xaxis: Object.assign(axis(x), { showbackground: false }),
        yaxis: Object.assign(axis(y), { showbackground: false }),
        zaxis: Object.assign(axis(z), { showbackground: false }),
        aspectmode: "cube",
      };
    } else {
      layout.margin = { l: 10, r: 10, t: 10, b: 10 };
      layout.xaxis = Object.assign(axis(x), { automargin: true });
      layout.yaxis = Object.assign(axis(y), { automargin: true });
    }
    if (!ready) { plot.innerHTML = ""; ready = true; }
    Plotly.react(plot, traces, layout, { responsive: true, displaylogo: false });
    if (!bound) {
      bound = true;
      plot.on("plotly_hover", e => { const c = e.points[0].customdata; if (c) A.hover(c[2], root.id); });
      plot.on("plotly_unhover", () => A.hover(null, root.id));
    }
  }

  function showLegislature(leg) {
    return A.json("pca_L" + leg + ".json").then(d => {
      data = d; hidden.clear(); fillAxes(); buildLegend();
      A.search(root.querySelector('[data-role="search"]'), "an-pca-mps", data.points);
      render();
    });
  }

  function setMode(m) {
    mode = m;
    root.querySelectorAll("[data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    zLabel.hidden = m === "2d";
    if (m === "3d" && (axes.z === axes.x || axes.z === axes.y)) {
      axes.z = ["1", "2", "3"].find(v => v !== axes.x && v !== axes.y);
      sel.z.value = axes.z;
    }
  }

  // Picking an axis already used elsewhere swaps the two, so no view ever repeats a component.
  ["x", "y", "z"].forEach(k => sel[k].addEventListener("change", () => {
    const v = sel[k].value;
    for (const o of ["x", "y", "z"]) if (o !== k && axes[o] === v) { axes[o] = axes[k]; sel[o].value = axes[o]; }
    axes[k] = v;
    if (data) render();
  }));
  root.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => { setMode(b.dataset.mode); if (data) render(); }));
  root.querySelectorAll("[data-color]").forEach(b => b.addEventListener("click", () => {
    colorBy = b.dataset.color;
    root.querySelectorAll("[data-color]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (data) render();
  }));
  sel.leg.addEventListener("change", () => { A.legislature(sel.leg.value); showLegislature(sel.leg.value); });
  A.on("legislature", () => {
    const leg = A.state.leg;
    if (leg === sel.leg.value || !Array.from(sel.leg.options).some(o => o.value === leg)) return;
    sel.leg.value = leg;
    showLegislature(leg);
  });
  A.on("focus", () => { if (A.state.source === root.id) return; if (data && ready) render(); });
  root.querySelectorAll("[data-preset]").forEach(btn => btn.addEventListener("click", () => {
    const [x, y, z] = btn.dataset.preset.split(",");
    axes.x = x; axes.y = y;
    if (z) axes.z = z;
    for (const k of ["x", "y", "z"]) sel[k].value = axes[k];
    setMode(z ? "3d" : "2d");
    if (data) render();
  }));
  legendEl.addEventListener("click", e => {
    const btn = e.target.closest("button[data-group]");
    if (!btn) return;
    const g = btn.dataset.group;
    if (hidden.has(g)) hidden.delete(g); else hidden.add(g);
    buildLegend(); render();
  });

  const io = new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) return;
    io.disconnect();
    A.plotly().then(() => showLegislature(sel.leg.value))
      .catch(() => { plot.textContent = "Could not load the interactive plot."; });
  }, { rootMargin: "300px" });
  io.observe(root);
})();
</script>

## 3. Theme by theme

The original article already ran a PCA per theme, with a simple rule: a ballot went to the first theme whose keywords appeared anywhere in its title. Here every text was read par a LLM instead.

- **Several ballots can be based on one text** The 172 ballots on the 2023 pension reform are 172 votes on one text, so they get one decision. The two legislatures hold 1,034 distinct texts, and each was read and placed, or left out.
- **In doubt, we don't classify** A text is placed only when its subject clearly belongs to one theme. Budget missions (*"loi de finances pour 2024 - Mission Justice"*: economy or justice?), the green industry bill (economy or environment?) are left out rather than forced into a box.
- **Votes on the government as a whole are left out**: motions of censure and general policy statements.
- **A theme needs at least 60 ballots** to get its own PCA. The 16th legislature has 54 ballots on institutions and 47 on education, so those two are shown for the 15th only.

The nine themes:

| Theme | What it covers | 15th leg. | 16th leg. |
| :--- | :--- | ---: | ---: |
| Economy & public finances | budgets, taxation, business law, purchasing power | 566 | 443 |
| Work, pensions & welfare | pensions, social security financing, employment, disability | 795 | 433 |
| Health & bioethics | the health system, the covid state of emergency, bioethics, end of life | 509 | 243 |
| Environment, energy & agriculture | climate, energy, farming, land use | 559 | 737 |
| Security, justice & immigration | courts and criminal law, police, immigration and asylum | 430 | 350 |
| Defence & foreign affairs | military programming, treaty ratifications, foreign policy resolutions | 63 | 208 |
| Housing & urban planning | housing supply, rents, urban planning | 193 | 148 |
| Institutions & democracy | the constitution, elections, local government, the civil service | 264 | *54* |
| Education, culture & research | schools, research, culture, sport | 208 | *47* |
| *Left out* | | *807* | *1,366* |
| **Total** | | **4,394** | **4,029** |


### A baseline for the explained variance

**The quantity.** For any set of $n$ ballots, the PCA is run on the matrix $X$ of MPs × ballots, each column centred and scaled. It diagonalizes $C = \frac{1}{m-1}X^\top X$, with eigenvalues $\lambda_1 \ge \dots \ge \lambda_n$. A standardized column contributes exactly $1$ to the total variance, so that total is $n$ itself, and the share carried by the first axis is

$$
\rho_1 = \frac{\lambda_1}{\sum_k \lambda_k} = \frac{\lambda_1}{n}.
$$

**How the two baseline columns are computed.** Take housing in the 16th legislature, $n = 148$ ballots out of the $N = 4{,}029$ of the legislature.

1. **The theme itself.** Run that PCA on the theme's 148 ballots: $\rho_1^{\text{theme}} = 21.9\%$. This is the column *16th: PC1*.
2. **One random draw.** Pick 148 of the 4,029 ballots uniformly at random, never twice the same ballot, and run exactly the same PCA on them. It returns one number, $\rho_1(S_1)$. Nothing of the theme is used here except its size.
3. **Repeat $B = 200$ times.** The draws give 200 values; sorted, $v_{(1)} \le \dots \le v_{(200)}$. The column *16th: random sets* is the 5th–95th percentile, $[v_{(10)}, v_{(190)}] = [15.9\%, 20.2\%]$: the range in which a set of 148 ballots picked at random usually lands.
4. **The $p$ column** counts the draws that match the theme:
$$
p = \frac{1}{B}\,\#\bigl\{\, b : \rho_1(S_b) \ge \rho_1^{\text{theme}} \,\bigr\}.
$$
For housing, 1 draw out of 200 reaches 21.9%, so $p = 0.005$. A small $p$ means the theme is more one-dimensional than ballots of the same number picked at random; a large one means it is unremarkable.

**Why the draws must have the theme's size.** Because $\rho_1 = \lambda_1/n$ has $n$ in its denominator. On a single ballot $\rho_1 = 1$, and each ballot added brings a dimension the first axis captures only in part, so $\rho_1$ drifts down as $n$ grows and swings more widely when $n$ is small. Comparing a 148-ballot theme to the legislature as a whole would compare two different scales; comparing it to 148-ballot draws does not. Drawing without replacement is part of the same care: repeated columns would be identical, load on the same component, and inflate the very $\lambda_1$ being tested.

| Theme | 15th: PC1 | 15th: random sets | *p* | 16th: PC1 | 16th: random sets | *p* |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Economy & public finances | 15.5% | 11.6–12.9% | <0.005 | 33.2% | 16.5–18.6% | <0.005 |
| Work, pensions & welfare | 19.6% | 11.6–12.8% | <0.005 | 40.5% | 16.4–18.9% | <0.005 |
| Health & bioethics | 20.9% | 11.6–13.1% | <0.005 | 17.4% | 16.0–19.4% | 0.58 |
| Environment, energy & agriculture | 14.5% | 11.6–12.9% | <0.005 | 18.0% | 16.4–18.3% | 0.20 |
| Security, justice & immigration | 16.7% | 11.5–13.0% | <0.005 | 16.2% | 16.4–19.1% | 0.96 |
| Defence & foreign affairs | 14.6% | 11.7–15.5% | 0.17 | 24.8% | 16.3–19.9% | <0.005 |
| Housing & urban planning | 33.8% | 11.4–13.6% | <0.005 | 21.9% | 15.9–20.2% | 0.005 |
| Institutions & democracy | 19.3% | 11.7–13.4% | <0.005 | | | |
| Education, culture & research | 19.6% | 11.5–13.8% | <0.005 | | | |

For reference, the PCA on all the ballots of a legislature gives $\rho_1 = 12.2\%$ in the 15th and $17.4\%$ in the 16th.

### Explore the themes

Same controls as the figure of section 2, plus a theme selector. The line above the plot gives the number of ballots, the variance of PC1 and PC2 with the range of the random sets, and the correlation, across MPs, between the theme's PC1 and the global PC1. The sign of each component is set to agree with the global component of the same rank, so that switching themes does not flip the plot. The "Votes in theme" coloring shows how many of the theme's ballots each MP voted on.

<div class="anviz" id="an-themes">
<div class="anviz-controls">
<select data-role="leg" aria-label="Legislature"><option value="15">15th legislature (2017-2022)</option><option value="16" selected>16th legislature (2022-2024)</option></select>
<select data-role="theme" aria-label="Theme"></select>
<span class="anviz-toggle" role="group" aria-label="View"><button type="button" data-mode="3d" aria-pressed="false">3D</button><button type="button" data-mode="2d" aria-pressed="true">2D</button></span>
<label>X <select data-role="x"></select></label>
<label>Y <select data-role="y"></select></label>
<label data-role="z-label" hidden>Z <select data-role="z"></select></label>
</div>
<div class="anviz-controls">
<span class="anviz-group"><span class="anviz-muted">Color</span><span class="anviz-toggle" role="group" aria-label="Color by"><button type="button" data-color="group" aria-pressed="true">Group</button><button type="button" data-color="cast" aria-pressed="false">Votes in theme</button></span></span>
<span class="anviz-muted">Presets:</span>
<button type="button" data-preset="1,2,3">3D: PC1 · PC2 · PC3</button>
<button type="button" data-preset="1,2">2D: PC1 · PC2</button>
<input type="search" data-role="search" list="an-themes-mps" placeholder="Find an MP" aria-label="Find an MP">
<datalist id="an-themes-mps"></datalist>
</div>
<div class="anviz-headline"></div>
<div class="anviz-plot"><span class="anviz-muted">Loading…</span></div>
<div class="anviz-legend"></div>
<div class="anviz-caption"></div>
</div>

<script>
(function () {
  const A = window.anviz;
  const root = document.getElementById("an-themes");
  const plot = root.querySelector(".anviz-plot");
  const legendEl = root.querySelector(".anviz-legend");
  const captionEl = root.querySelector(".anviz-caption");
  const headlineEl = root.querySelector(".anviz-headline");
  const sel = {};
  ["leg", "theme", "x", "y", "z"].forEach(k => { sel[k] = root.querySelector('[data-role="' + k + '"]'); });
  const zLabel = root.querySelector('[data-role="z-label"]');
  const axes = { x: "1", y: "2", z: "3" };
  let mode = "2d", colorBy = "group", themeKey = "social";
  const hidden = new Set();
  let data = null, theme = null, ready = false, bound = false;

  const esc = A.esc;
  const cssVar = (name, fallback) => A.cssVar(root, name, fallback);
  const pctOf = v => (100 * v).toFixed(1) + "%";
  const pcLabel = k => "PC" + k + " (" + pctOf(theme.explained_variance[k - 1]) + ")";

  function fillThemes() {
    sel.theme.innerHTML = data.themes.map(t => '<option value="' + t.key + '">' + esc(t.label) + " (" + t.n_scrutins + ")</option>").join("");
    if (!data.themes.some(t => t.key === themeKey)) themeKey = data.themes[0].key;
    sel.theme.value = themeKey;
  }

  function selectTheme() {
    theme = data.themes.find(t => t.key === themeKey);
    const opts = theme.explained_variance.map((v, i) => '<option value="' + (i + 1) + '">' + pcLabel(i + 1) + "</option>");
    for (const key of ["x", "y", "z"]) { sel[key].innerHTML = opts.join(""); sel[key].value = axes[key]; }
    const range = k => (100 * theme.baseline_range[k][0]).toFixed(1) + "–" + pctOf(theme.baseline_range[k][1]);
    // p is the share of the 200 random draws reaching the theme's own variance; 0 prints as the resolution bound.
    const pval = k => theme.baseline_p ? ", p " + (theme.baseline_p[k] ? "= " + theme.baseline_p[k].toFixed(2) : "< 0.005") : "";
    headlineEl.innerHTML = "<b>" + esc(theme.label) + "</b><br><span class='anviz-muted'>" + theme.n_scrutins + " ballots · PC1 " +
      pctOf(theme.explained_variance[0]) + " of variance (200 random sets of " + theme.n_scrutins + " ballots, 5–95%: " + range(0) + pval(0) +
      ") · PC2 " + pctOf(theme.explained_variance[1]) + " (random sets: " + range(1) + pval(1) +
      ") · correlation with the global PC1: " + theme.global_correlation[0].toFixed(2) + "</span>";
    buildLegend();
  }

  function points() {
    return data.mps.map((m, i) => theme.pc[i] ? Object.assign({ pc: theme.pc[i], cast: theme.cast[i] }, m) : null).filter(Boolean);
  }

  function buildLegend() {
    legendEl.innerHTML = data.groups.map(g =>
      '<button type="button" data-group="' + esc(g.name) + '"' + (hidden.has(g.name) ? ' class="off"' : "") + '><span class="dot" style="background:' + g.color + '"></span>' + esc(g.name) + " (" + g.count + ")</button>").join("");
    captionEl.textContent = points().length + " MPs who voted at least once on this theme. " +
      "Only ballots whose text matches exactly one theme are used; themes with fewer than " + data.min_ballots + " ballots are not shown.";
  }

  function render() {
    const x = +axes.x, y = +axes.y, z = +axes.z, is3d = mode === "3d";
    const text = cssVar("--anviz-fg", "#fff");
    const muted = cssVar("--anviz-muted", "#9a9a9a");
    const grid = cssVar("--anviz-border", "rgba(255,255,255,.14)");
    const bg = cssVar("--anviz-bg", "#1b1d23");
    const type = is3d ? "scatter3d" : "scatter";
    const all = points();
    const coords = pts => {
      const t = { x: pts.map(p => p.pc[x - 1]), y: pts.map(p => p.pc[y - 1]) };
      if (is3d) t.z = pts.map(p => p.pc[z - 1]);
      return t;
    };
    const meta = pts => ({
      text: pts.map(p => p.name),
      customdata: pts.map(p => [p.group, p.cast, p.id]),
      hovertemplate: "<b>%{text}</b><br>%{customdata[0]}<br>voted on %{customdata[1]} of the " + theme.n_scrutins + " ballots of the theme<extra></extra>",
    });

    let traces;
    if (colorBy === "cast") {
      const pts = all.filter(p => !hidden.has(p.group));
      traces = [Object.assign({ type: type, mode: "markers", name: "Votes in theme" }, coords(pts), meta(pts), {
        marker: {
          color: pts.map(p => p.cast / theme.n_scrutins), colorscale: "Viridis", size: is3d ? 3.5 : 7, opacity: 0.9,
          colorbar: {
            title: { text: "Share of the theme's ballots", side: "right" }, tickformat: ".0%", thickness: 12, len: 0.7,
            outlinewidth: 0, tickfont: { color: muted }, titlefont: { color: muted },
          },
        },
      })];
    } else {
      traces = data.groups.map(g => {
        const pts = all.filter(p => p.group === g.name);
        return Object.assign({ type: type, mode: "markers", name: g.name, visible: !hidden.has(g.name) }, coords(pts), meta(pts), {
          marker: { color: g.color, size: is3d ? 3.5 : 7, opacity: 0.85, line: { width: is3d ? 0 : 0.5, color: bg } },
        });
      });
    }

    const focus = all.find(p => p.id === A.focused());
    if (focus) {
      traces.push(Object.assign({ type: type, mode: "markers+text", name: "selected", hoverinfo: "skip", showlegend: false },
        coords([focus]), {
          text: [focus.name], textposition: "top center", textfont: { color: text, size: 12 },
          marker: { size: is3d ? 9 : 16, symbol: "circle-open", color: text, line: { color: text, width: 2 } },
        }));
    }

    const axis = k => ({ title: { text: pcLabel(k) }, gridcolor: grid, zerolinecolor: grid, color: text });
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: text, size: 12 }, showlegend: false,
      modebar: { bgcolor: "rgba(0,0,0,0)", color: muted, activecolor: text },
      hoverlabel: { font: { size: 13 } },
      uirevision: [data.legislature, themeKey, mode, x, y, is3d ? z : 0].join("-"),
    };
    if (is3d) {
      layout.margin = { l: 0, r: 0, t: 0, b: 0 };
      layout.scene = {
        xaxis: Object.assign(axis(x), { showbackground: false }),
        yaxis: Object.assign(axis(y), { showbackground: false }),
        zaxis: Object.assign(axis(z), { showbackground: false }),
        aspectmode: "cube",
      };
    } else {
      layout.margin = { l: 10, r: 10, t: 10, b: 10 };
      layout.xaxis = Object.assign(axis(x), { automargin: true });
      layout.yaxis = Object.assign(axis(y), { automargin: true });
    }
    if (!ready) { plot.innerHTML = ""; ready = true; }
    Plotly.react(plot, traces, layout, { responsive: true, displaylogo: false });
    if (!bound) {
      bound = true;
      plot.on("plotly_hover", e => { const c = e.points[0].customdata; if (c) A.hover(c[2], root.id); });
      plot.on("plotly_unhover", () => A.hover(null, root.id));
    }
  }

  function showLegislature(leg) {
    return A.json("pca_themes_L" + leg + ".json").then(d => {
      data = d; hidden.clear(); fillThemes(); selectTheme();
      A.search(root.querySelector('[data-role="search"]'), "an-themes-mps", data.mps);
      render();
    });
  }

  function setMode(m) {
    mode = m;
    root.querySelectorAll("[data-mode]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    zLabel.hidden = m === "2d";
    if (m === "3d" && (axes.z === axes.x || axes.z === axes.y)) {
      axes.z = ["1", "2", "3"].find(v => v !== axes.x && v !== axes.y);
      sel.z.value = axes.z;
    }
  }

  // Picking an axis already used elsewhere swaps the two, so no view ever repeats a component.
  ["x", "y", "z"].forEach(k => sel[k].addEventListener("change", () => {
    const v = sel[k].value;
    for (const o of ["x", "y", "z"]) if (o !== k && axes[o] === v) { axes[o] = axes[k]; sel[o].value = axes[o]; }
    axes[k] = v;
    if (data) render();
  }));
  sel.theme.addEventListener("change", () => { themeKey = sel.theme.value; selectTheme(); render(); });
  root.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => { setMode(b.dataset.mode); if (data) render(); }));
  root.querySelectorAll("[data-color]").forEach(b => b.addEventListener("click", () => {
    colorBy = b.dataset.color;
    root.querySelectorAll("[data-color]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (data) render();
  }));
  sel.leg.addEventListener("change", () => { A.legislature(sel.leg.value); showLegislature(sel.leg.value); });
  A.on("legislature", () => {
    const leg = A.state.leg;
    if (leg === sel.leg.value || !Array.from(sel.leg.options).some(o => o.value === leg)) return;
    sel.leg.value = leg;
    if (data) showLegislature(leg);
  });
  A.on("focus", () => { if (A.state.source === root.id) return; if (data && ready) render(); });
  root.querySelectorAll("[data-preset]").forEach(btn => btn.addEventListener("click", () => {
    const [x, y, z] = btn.dataset.preset.split(",");
    axes.x = x; axes.y = y;
    if (z) axes.z = z;
    for (const k of ["x", "y", "z"]) sel[k].value = axes[k];
    setMode(z ? "3d" : "2d");
    if (data) render();
  }));
  legendEl.addEventListener("click", e => {
    const btn = e.target.closest("button[data-group]");
    if (!btn) return;
    const g = btn.dataset.group;
    if (hidden.has(g)) hidden.delete(g); else hidden.add(g);
    buildLegend(); render();
  });

  const io = new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) return;
    io.disconnect();
    A.plotly().then(() => showLegislature(sel.leg.value))
      .catch(() => { plot.textContent = "Could not load the interactive plot."; });
  }, { rootMargin: "300px" });
  io.observe(root);
})();
</script>

### What the themes show

Group positions below are medians; "one end" and "the other end" refer to the component named.

**Correlation with the global PC1.** In the 15th legislature, it is between 0.76 and 0.90 for every theme except defence (0.59) and housing (0.49). In the 16th: economy 0.96, work & welfare 0.94, environment 0.80, security 0.77, defence 0.52, health 0.47, housing 0.26.

**16th legislature.**

- *Economy & public finances.* PC1 goes from LFI, the RN and the ecologists at one end to Renaissance, MoDem and Horizons at the other. On PC2, 29 of the 30 highest MPs are RN and 28 of the 30 lowest are LFI.
- *Work, pensions & welfare.* PC1 goes from LFI and the ecologists to Renaissance, MoDem and Horizons. LR is on the side of these three groups (median +4.4, against −2.3 on the global PC1), the RN on the other side (−8.8). On PC3, the RN median is +8.3; the other groups are between −2.9 and +0.1.
- *Security, justice & immigration.* PC1 goes from LFI, the ecologists, GDR and the RN to Renaissance, MoDem and Horizons. PC2 explains 11.7% of the variance (random sets: 6.9–8.1%, *p* < 0.005); 29 of the 30 highest MPs are RN, and 20 of the 30 lowest are LFI.
- *Environment, energy & agriculture.* PC1 goes from the ecologists and LFI to MoDem and Renaissance, with LR and the RN in between. On PC2, 29 of the 30 highest MPs are RN.
- *Health & bioethics.* 122 of the 243 ballots are about the end-of-life bill. On PC1, the 30 lowest MPs include 21 RN and 8 LR MPs; the 30 highest include 24 Renaissance and 2 LFI MPs.
- *Defence & foreign affairs.* 159 of the 208 ballots are about the military programming bill. On PC1, Renaissance is the only group with a positive median; on PC2, the 30 highest MPs are all RN.
- *Housing & urban planning.* On PC1, the 30 lowest MPs are all RN; the 30 highest include 13 Renaissance, 7 socialist and 7 LFI MPs. On PC2, 27 of the 30 lowest are LFI and 21 of the 30 highest are Renaissance.

**15th legislature.** In every theme, PC1 has LREM at one end, and LFI or LR among the MPs at the other. Two themes are dominated by one text: 244 of the 795 work & welfare ballots are about the universal pension bill, and 190 of the 193 housing ballots are about the ELAN bill (2018). For housing, the median MP voted on 6 of the 193 ballots, and 27 of the 30 MPs highest on PC1 are LREM.

**Comparison with the original article.** For its "Solidarity & Social" theme, the original article reported a PC1 of about 20% in the 15th legislature and over 40% in the 16th. With the classification used here, the closest theme, work & welfare, gives 19.6% and 40.5%. Random sets of the same size give 11.8–13.4% and 16.4–18.8%: PC1 is also higher in the 16th legislature for ballots taken at random.

**Participation.** Within each theme, the Spearman correlation between the number of the theme's ballots an MP voted on and their distance, on (PC1, PC2), to the theme's no-vote point (see section 3) is between 0.78 and 0.97.
