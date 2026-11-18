/* NEURAL / LAB · Defect experiment room.
   Draw on a real weld; every redraw, error map and score is a real forward pass. Transitions only animate
   between real values (the needle, crossfades between two real images); numbers are always shown exactly. */
"use strict";
const Defect = (() => {
  const N = 64, NN = N * N;
  let root, stage, ctx, off, mode = "dark", drawing = false, undo = [], history = [], lastLevel = null, busy = false, again = false;
  const $ = (s, el = root) => el.querySelector(s);
  const LOG0 = -5, LOG1 = -1, pos = s => 100 * (Math.log10(Math.max(s, 1e-5)) - LOG0) / (LOG1 - LOG0);

  function build(el) {
    root = el; stage = $("#dStage"); ctx = stage.getContext("2d"); off = document.createElement("canvas"); off.width = off.height = N;
    const sc = $("#dScale");
    sc.style.setProperty("--a", (100 - pos(NL.cfg.t_low)) + "%"); sc.style.setProperty("--b", (100 - pos(NL.cfg.t_high)) + "%");
    sc.insertAdjacentHTML("beforeend", `<span class="th" style="bottom:${pos(NL.cfg.t_low)}%">check ${NL.cfg.t_low.toFixed(5)}</span>
      <span class="th" style="bottom:${pos(NL.cfg.t_high)}%">high ${NL.cfg.t_high.toFixed(5)}</span>`);
    stage.addEventListener("pointerdown", e => { if (!NL.state.now) return; drawing = true; undo.push(NL.state.cur.slice()); stage.setPointerCapture(e.pointerId); brush(e); });
    let last = 0;
    stage.addEventListener("pointermove", e => { if (!drawing) return; const t = performance.now(); if (t - last < 14) return; last = t; brush(e); });
    stage.addEventListener("pointerup", () => drawing = false);
    root.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => { root.querySelectorAll("[data-mode]").forEach(o => o.classList.remove("on")); b.classList.add("on"); mode = b.dataset.mode; });
    $("#dUndo").onclick = () => { if (undo.length) { NL.state.cur = undo.pop(); refresh(); } };
    $("#dReset").onclick = () => { NL.state.cur = NL.state.orig.slice(); undo = []; history = []; refresh(); };
    const lab = () => { $("#dSizeV").textContent = $("#dSize").value + " px"; $("#dStrV").textContent = $("#dStr").value + "%";
      $("#dOvV").textContent = $("#dOv").value + "%"; $("#dWipeV").textContent = $("#dWipe").value + "%"; };
    ["#dSize", "#dStr"].forEach(s => $(s).oninput = lab); ["#dOv", "#dWipe"].forEach(s => $(s).oninput = () => { lab(); paintStage(); }); lab();
    $("#dPlay").onclick = playSequence;
    const wrap = $("#dStageWrap");
    wrap.addEventListener("dragover", e => { e.preventDefault(); wrap.classList.add("drop"); });
    wrap.addEventListener("dragleave", () => wrap.classList.remove("drop"));
    wrap.addEventListener("drop", e => { e.preventDefault(); wrap.classList.remove("drop"); if (e.dataTransfer.files[0]) NL.loadFile(e.dataTransfer.files[0]); });
    Sens.build($("#sens"));
    NL.on(what => { if (what === "tile") { undo = []; history = []; lastLevel = null; render(); if (visible()) playSequence(); } });
  }
  const visible = () => !root.hidden;

  /* ---- drawing on the shared weld ---- */
  function brush(e) {
    const r = stage.getBoundingClientRect(), cx = (e.clientX - r.left) / r.width * N, cy = (e.clientY - r.top) / r.height * N;
    const R = +$("#dSize").value, k = +$("#dStr").value / 100, cur = NL.state.cur, orig = NL.state.orig;
    for (let y = Math.floor(cy - R); y <= Math.ceil(cy + R); y++) for (let x = Math.floor(cx - R); x <= Math.ceil(cx + R); x++) {
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      const d = Math.hypot(x + .5 - cx, y + .5 - cy); if (d > R) continue;
      const w = k * (1 - d / R) ** .7, i = y * N + x, tgt = mode === "dark" ? 0 : mode === "light" ? 1 : orig[i];
      cur[i] += (tgt - cur[i]) * w;
    }
    refresh();
  }
  /* one inference in flight; the latest drawing always wins */
  async function refresh() {
    if (busy) { again = true; return; }
    busy = true; again = false;
    NL.state.now = await NL.run(NL.state.cur.slice());
    history.push(NL.state.now.score); if (history.length > 300) history.shift();
    render(); NL.emit("draw");
    busy = false; if (again) refresh();
  }
  /* used by the sensitivity study and the demo: replace the weld with an edited copy, keeping undo */
  function apply(x) { undo.push(NL.state.cur.slice()); NL.state.cur = x; return refresh(); }

  /* ---- display ---- */
  function paintStage() {
    const s = NL.state, W = stage.width, ov = +$("#dOv").value / 100, wipe = +$("#dWipe").value / 100, o = off.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    NL.paint(off, s.cur, N); ctx.drawImage(off, 0, 0, W, W);
    if (ov > 0) { NL.paint(off, s.now.err, N, "heat", NL.cfg.error_map_max); ctx.globalAlpha = ov; ctx.drawImage(off, 0, 0, W, W); ctx.globalAlpha = 1; }
    if (wipe > 0) { NL.paint(off, s.orig, N); ctx.drawImage(off, 0, 0, N * wipe, N, 0, 0, W * wipe, W);
      ctx.fillStyle = "#efe8da"; ctx.fillRect(W * wipe - 1, 0, 2, W); }
    $("#dWipeTag").hidden = wipe === 0; $("#dWipeTag").style.left = `${wipe * 100}%`;
  }
  function render() {
    const s = NL.state; if (!s.now) return;
    paintStage();
    const sc = s.now.score, L = NL.level(sc), base = s.base.score, edited = s.cur.some((v, i) => v !== s.orig[i]);
    $("#dScore").textContent = sc.toFixed(6);
    const lv = $("#dLevel"); lv.textContent = NL.LEVEL_TEXT[L]; lv.dataset.level = L;
    $("#dSentence").textContent = NL.LEVEL_SENTENCE[L];
    document.body.dataset.level = L;
    if (lastLevel && lastLevel !== L) { lv.classList.remove("shift"); void lv.offsetWidth; lv.classList.add("shift"); }
    lastLevel = L;
    const ratio = sc / base;
    $("#dDelta").innerHTML = edited
      ? `Original <b class="num">${base.toFixed(6)}</b>, now <b class="num">${sc.toFixed(6)}</b>: <b>${ratio >= 1 ? "×" + ratio.toFixed(1) : "÷" + (1 / ratio).toFixed(1)}</b>`
      : `Unedited tile · ${(sc / NL.cfg.threshold).toFixed(2)}× the alarm threshold`;
    $("#dNeedle").style.bottom = pos(sc) + "%"; $("#dOrig").style.bottom = pos(base) + "%";
    spark();
    if (!seqPlaying) { NL.paint($("#dSeqA"), s.cur, N); NL.paint($("#dSeqB"), s.now.rec, N); NL.paint($("#dSeqC"), s.now.err, N, "heat", NL.cfg.error_map_max); }
    const w = s.weld;
    $("#dInfo").textContent = w.label === "uploaded" ? "Your uploaded image, preprocessed in Python exactly like training."
      : `Test weld ${w.weld}, tile ${w.source}. True class: ${NL.CLASS_NAME[w.label]}.`;
  }
  function spark() {
    const svg = $("#dSpark"), h = history.length ? history : [NL.state.now.score], n = Math.max(h.length, 2);
    const y = s => 60 - 60 * pos(s) / 100, x = i => 400 * i / (n - 1);
    svg.innerHTML = [[NL.cfg.t_low, "var(--check)"], [NL.cfg.t_high, "var(--high)"]].map(([t, c]) =>
      `<line x1="0" x2="400" y1="${y(t)}" y2="${y(t)}" stroke="${c}" stroke-dasharray="3 4" opacity=".6"/>`).join("") +
      `<polyline fill="none" stroke="#efe8da" stroke-width="1.5" points="${(h.length === 1 ? [h[0], h[0]] : h).map((s, i) => `${x(i)},${y(s)}`).join(" ")}"/>`;
  }

  /* ---- original -> reconstruction -> error: crossfades between real images, then the real error map ---- */
  let seqPlaying = false;
  async function playSequence() {
    if (seqPlaying || matchMedia("(prefers-reduced-motion: reduce)").matches) return render();
    seqPlaying = true;
    const s = NL.state, frames = root.querySelectorAll(".seq figure");
    frames.forEach(f => f.classList.remove("lit"));
    NL.paint($("#dSeqA"), s.cur, N); NL.paint($("#dSeqB"), s.cur, N); NL.paint($("#dSeqC"), new Float32Array(NN), N, "heat", 1);
    frames[0].classList.add("lit"); await wait(500);
    frames[1].classList.add("lit");
    for (let t = 0; t <= 1.0001; t += 1 / 24) { const mix = new Float32Array(NN);       // crossfade: weld -> real reconstruction
      for (let i = 0; i < NN; i++) mix[i] = s.cur[i] * (1 - t) + s.now.rec[i] * t; NL.paint($("#dSeqB"), mix, N); await wait(28); }
    frames[2].classList.add("lit");
    for (let t = 0; t <= 1.0001; t += 1 / 18) { const e = new Float32Array(NN);          // error map fades in at its true fixed scale
      for (let i = 0; i < NN; i++) e[i] = s.now.err[i] * t; NL.paint($("#dSeqC"), e, N, "heat", NL.cfg.error_map_max); await wait(28); }
    seqPlaying = false; render();
  }
  const wait = ms => new Promise(r => setTimeout(r, ms));

  return { build, render, apply, playSequence, paintStage };
})();

/* ---- sensitivity study: one synthetic defect on 12 real sound test welds, real scores ---- */
const Sens = (() => {
  const N = 64, NN = N * N;
  const SHAPES = { pore: { size: [1, 12], unit: "radius, px" }, crack: { size: [0.5, 6], unit: "width, px" }, band: { size: [1, 16], unit: "height, px" } };
  let root, sound = null, rows = [], p = null;
  const $ = s => root.querySelector(s);
  const LOG0 = -5, LOG1 = -1, pos = s => 100 * (Math.log10(Math.max(s, 1e-5)) - LOG0) / (LOG1 - LOG0);
  function weight(shape, size, x, y) { const c = v => Math.min(Math.max(v, 0), 1);
    if (shape === "pore") return c(size + .5 - Math.hypot(x + .5 - 32, y + .5 - 32));
    if (shape === "band") return c(size / 2 + .5 - Math.abs(y + .5 - 32));
    const ax = 12, ay = 52, dx = 40, dy = -40, px = x + .5, py = y + .5, t = Math.min(Math.max(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0), 1);
    return c(size / 2 + .5 - Math.hypot(px - (ax + t * dx), py - (ay + t * dy))); }
  function withDefect(x, shape, size, dark) { const o = new Float32Array(NN);
    for (let y = 0; y < N; y++) for (let xx = 0; xx < N; xx++) { const i = y * N + xx; o[i] = x[i] * (1 - dark * weight(shape, size, xx, y)); } return o; }
  function params() { const shape = $("#sShape").value, v = $("#sVar").value, f = +$("#sFix").value / 100, [a, b] = SHAPES[shape].size, st = [...Array(10)].map((_, i) => i / 9);
    return v === "size" ? { shape, v, dark: .15 + .8 * f, xs: st.map(t => a + (b - a) * t), label: `${shape} ${SHAPES[shape].unit}` }
                        : { shape, v, size: a + (b - a) * f, xs: st.map(t => .05 + .9 * t), label: "darkness (0 = invisible, 1 = black)" }; }
  const at = (p, x) => p.v === "size" ? [x, p.dark] : [p.size, x];
  function fixLabel() { const q = params(); $("#sFixL").textContent = q.v === "size" ? `Fixed darkness ${q.dark.toFixed(2)}` : `Fixed size ${q.size.toFixed(1)} (${SHAPES[q.shape].unit})`; }

  function build(el) {
    root = el;
    ["#sShape", "#sVar", "#sFix"].forEach(s => $(s).addEventListener("input", fixLabel)); fixLabel();
    $("#sRun").onclick = run; $("#sCsv").onclick = csv;
    $("#sApply").onclick = () => { const q = params(), [size, dark] = at(q, q.xs[6]);
      Defect.apply(withDefect(NL.state.cur, q.shape, size, dark)); document.getElementById("dStage").scrollIntoView({ behavior: "smooth", block: "center" }); };
    draw();
  }
  async function run() {
    if (!sound) sound = await NL.json("/api/sound_set");
    p = params(); rows = []; $("#sCsv").disabled = true;
    const strip = $("#sStrip"); strip.innerHTML = "";
    for (let k = 0; k < p.xs.length; k++) {
      const [size, dark] = at(p, p.xs[k]), scores = [];
      for (const t of sound) scores.push((await NL.run(withDefect(Float32Array.from(t.x), p.shape, size, dark))).score);
      rows.push({ x: p.xs[k], size, dark, scores });
      const c = document.createElement("canvas"); NL.paint(c, withDefect(Float32Array.from(sound[0].x), p.shape, size, dark), N); strip.appendChild(c);
      $("#sNote").textContent = `Running · ${(k + 1) * sound.length} of ${p.xs.length * sound.length} real inferences`; draw();
    }
    const flagged = rows.map(r => r.scores.filter(s => s > NL.cfg.threshold).length);
    $("#sNote").textContent = `Done: ${p.xs.length * sound.length} real inferences. Welds above the alarm threshold at each step: ${flagged.join(" · ")} (of ${sound.length}).`;
    $("#sCsv").disabled = false;
  }
  function draw() {
    const q = p || params(), svg = $("#sChart"), W = 900, H = 380, L = 64, R = 190, T = 18, B = 50, xs = q.xs, x0 = xs[0], x1 = xs[xs.length - 1];
    const X = v => L + (W - L - R) * (v - x0) / (x1 - x0), Y = s => T + (H - T - B) * (1 - pos(s) / 100);
    let g = "";
    for (const e of [-5, -4, -3, -2, -1]) g += `<line x1="${L}" x2="${W - R}" y1="${Y(10 ** e)}" y2="${Y(10 ** e)}" stroke="#26241f"/><text x="${L - 10}" y="${Y(10 ** e) + 4}" class="ax" text-anchor="end">10${["⁻⁵", "⁻⁴", "⁻³", "⁻²", "⁻¹"][e + 5]}</text>`;
    for (const [t, c, n] of [[NL.cfg.t_low, "var(--check)", "check"], [NL.cfg.t_high, "var(--high)", "high"]])
      g += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="${c}" stroke-dasharray="4 5"/><text x="${W - R - 6}" y="${Y(t) - 6}" fill="${c}" class="ax" text-anchor="end">${n}</text>`;
    if (rows.length) {
      const lo = rows.map(r => Math.min(...r.scores)), hi = rows.map(r => Math.max(...r.scores)), mid = rows.map(r => r.scores.reduce((a, b) => a + b) / r.scores.length), rx = rows.map(r => r.x);
      g += `<polygon fill="var(--ember)" opacity=".16" points="${rx.map((v, i) => `${X(v)},${Y(hi[i])}`).join(" ")} ${rx.map((v, i) => `${X(v)},${Y(lo[i])}`).reverse().join(" ")}"/>`;
      g += `<polyline fill="none" stroke="var(--ember)" stroke-width="2.4" points="${rx.map((v, i) => `${X(v)},${Y(mid[i])}`).join(" ")}"/>`;
      g += rx.map((v, i) => `<circle cx="${X(v)}" cy="${Y(mid[i])}" r="4" fill="var(--ember)"/>`).join("");
    }
    [["porosity", "pores"], ["crack", "cracks"], ["lack_of_penetration", "lack of pen."], ["no_defect", "sound"]].forEach(([c, n], j) => {
      const [a, m, b] = NL.cfg.class_scores[c], x = W - R + 34 + j * 40;
      g += `<line x1="${x}" x2="${x}" y1="${Y(a)}" y2="${Y(b)}" stroke="${NL.CLASS_COL[c]}" stroke-width="8" stroke-linecap="round"/>
            <circle cx="${x}" cy="${Y(m)}" r="4.5" fill="#09090a" stroke="${NL.CLASS_COL[c]}" stroke-width="2"/>
            <text x="${x}" y="${H - B + 18}" class="ax" text-anchor="end" transform="rotate(-35 ${x} ${H - B + 18})">${n}</text>`; });
    g += `<text x="${W - R + 94}" y="${T + 2}" class="ax" text-anchor="middle">real test tiles, middle 50%</text>`;
    g += `<text x="${(L + W - R) / 2}" y="${H - 8}" class="ax" text-anchor="middle">${q.label}</text>`;
    xs.forEach(v => g += `<text x="${X(v)}" y="${H - B + 18}" class="ax dim" text-anchor="middle">${v.toFixed(v < 10 ? 1 : 0)}</text>`);
    svg.innerHTML = g;
  }
  function csv() {
    const out = [["shape", "size", "darkness", "weld_tile_id", "score"]];
    rows.forEach(r => r.scores.forEach((s, i) => out.push([p.shape, r.size.toFixed(3), r.dark.toFixed(3), sound[i].id, s.toExponential(6)])));
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([out.map(r => r.join(",")).join("\n")], { type: "text/csv" }));
    a.download = `sensitivity_${p.shape}_${p.v}.csv`; a.click();
  }
  return { build, withDefect };
})();
