/* NEURAL / LAB · Architecture room.
   The real ConvAE(64, 32) laid out as nine stations. Every plane is a real feature map from the current forward
   pass; every bar is a real per-channel mean. The lines between stations show the ORDER of operations only:
   they are not weights or neuron-to-neuron connections. */
"use strict";
const Arch = (() => {
  const STATIONS = [
    { k: "in",  name: "Input",          C: 1,   H: 64, src: "x",    op: "Your weld tile: greyscale, resized to 64×64, pixel values 0–1." },
    { k: "e1",  name: "Encoder 1",      C: 32,  H: 32, src: "enc1", op: "Convolution 3×3, stride 2 (1 → 32 channels), BatchNorm, ReLU." },
    { k: "e2",  name: "Encoder 2",      C: 64,  H: 16, src: "enc2", op: "Convolution 3×3, stride 2 (32 → 64 channels), BatchNorm, ReLU." },
    { k: "e3",  name: "Encoder 3",      C: 128, H: 8,  src: "enc3", op: "Convolution 3×3, stride 2 (64 → 128 channels), BatchNorm, ReLU." },
    { k: "z",   name: "Latent code",    C: 32,  H: 0,  src: "z",    op: "Flatten 128×8×8 = 8,192 values, then a linear layer to 32 numbers. The decoder sees only these." },
    { k: "d1",  name: "Decoder 1",      C: 128, H: 8,  src: "dec1", op: "Linear 32 → 8,192, ReLU, reshaped to 128 channels of 8×8." },
    { k: "d2",  name: "Decoder 2",      C: 64,  H: 16, src: "dec2", op: "Transposed convolution 3×3, stride 2 (128 → 64 channels), BatchNorm, ReLU." },
    { k: "d3",  name: "Decoder 3",      C: 32,  H: 32, src: "dec3", op: "Transposed convolution 3×3, stride 2 (64 → 32 channels), BatchNorm, ReLU." },
    { k: "out", name: "Reconstruction", C: 1,   H: 64, src: "rec",  op: "Transposed convolution 3×3, stride 2 (32 → 1), sigmoid: the redrawn 64×64 tile, values 0–1." },
  ];
  const SIZE = { 64: 164, 32: 132, 16: 106, 8: 86 };
  const PLANES = 5;
  let dirty = false;
  let root, zMod = null, steered = null, ranks = {}, busy = false, again = false;
  const $ = (s, el = document) => el.querySelector(s);

  /* which tensors are on screen: encoder side = your weld; decoder side = your edited code if you changed it */
  const view = () => { const b = NL.state.now; if (!steered) return b;
    return { ...b, dec1: steered.dec1, dec2: steered.dec2, dec3: steered.dec3, rec: steered.rec, err: steered.err, score: steered.score }; };
  const changed = d => Math.abs(zMod[d] - NL.state.now.z[d]) > 1e-6;
  const anyChanged = () => zMod && zMod.some((_, d) => changed(d));

  function build(el) {
    root = el;
    const rail = $("#rail", root);
    rail.innerHTML = STATIONS.map((s, i) => `
      <button class="station st-${s.k}" data-k="${s.k}" style="--i:${i}" aria-label="Inspect ${s.name}">
        ${s.k === "z" ? `<div class="latent" id="lat"></div>` :
          `<div class="stack" style="--w:${SIZE[s.H]}px">${[...Array(s.k === "out" ? 2 : Math.min(PLANES, s.C))].map((_, j) =>
            `<canvas class="plane" style="--j:${j}"></canvas>`).join("")}</div>`}
        <span class="nm">${s.name}</span>
        <span class="dims num">${s.k === "z" ? "32 numbers" : `${s.C} × ${s.H}×${s.H}`}</span>
        ${s.C > 1 && s.k !== "z" ? `<canvas class="strip" width="${s.C * 3}" height="22"></canvas>` : `<span class="strip-gap"></span>`}
        <span class="act num" id="act-${s.k}"></span>
      </button>`).join("");
    buildLatent();
    rail.querySelectorAll(".station").forEach(b => b.addEventListener("click", e => { if (!e.target.closest(".latent")) inspect(b.dataset.k); }));
    const scene = $("#scene", root);
    scene.addEventListener("pointermove", e => { if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      const r = scene.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
      scene.style.setProperty("--ry", `${-30 + x * 18}deg`); scene.style.setProperty("--rx", `${12 - y * 10}deg`); });
    $("#zReset", root).onclick = () => { zMod = NL.state.now.z.slice(); steered = null; render(); };
    $("#drawerClose").onclick = closeDrawer;
    addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
    addEventListener("resize", flows);
    NL.on(what => { if (what === "tile" || what === "draw") { zMod = NL.state.now.z.slice(); steered = null; rerank();
      if (root.hidden) dirty = true; else render(); } });
  }

  /* channel order (strongest first) is fixed per weld, so planes do not jump while you steer */
  function rerank() {
    const b = NL.state.now;
    for (const s of STATIONS) if (s.C > 1 && s.k !== "z") ranks[s.k] = NL.channelStats(b[s.src], s.C, s.H).order;
  }

  function render() {
    if (!NL.state.now) return;
    const v = view();
    for (const s of STATIONS) {
      if (s.k === "z") continue;
      const el = $(`.st-${s.k}`, root), planes = el.querySelectorAll(".plane");
      if (s.k === "in") NL.paint(planes[0], v.x, 64);
      else if (s.k === "out") { NL.paint(planes[0], v.rec, 64); NL.paint(planes[1], v.err, 64, "heat", NL.cfg.error_map_max); }
      else {
        const st = NL.channelStats(v[s.src], s.C, s.H), scale = st.max || 1, HH = s.H * s.H;
        planes.forEach((p, j) => NL.paint(p, v[s.src], s.H, "act", scale, ranks[s.k][j] * HH));
        strip(el.querySelector(".strip"), st.mean, ranks[s.k]);
        $(`#act-${s.k}`, root).textContent = `mean activation ${st.overall.toFixed(2)}`;
      }
      el.classList.toggle("steered", !!steered && s.k[0] !== "e" && s.k !== "in");
    }
    $("#act-in", root).textContent = "your weld";
    $("#act-out", root).textContent = `error ${v.score.toFixed(6)}`;
    renderLatent(); readline(v); flows(true);
    if (openK) inspect(openK, true);
  }

  /* per-channel mean activations, in the fixed strongest-first order of this weld */
  function strip(c, mean, order) {
    const ctx = c.getContext("2d"), W = c.width, H = c.height, m = Math.max(...mean) || 1;
    ctx.clearRect(0, 0, W, H);
    order.forEach((ch, i) => { const h = Math.max(1, (mean[ch] / m) * (H - 2));
      ctx.fillStyle = i < PLANES ? "#e8742c" : "#6e675d"; ctx.fillRect(i * 3, H - h, 2, h); });
  }

  /* ---- latent bars: drag to change, decoder answers immediately ---- */
  function buildLatent() {
    const lat = $("#lat", root);
    lat.innerHTML = [...Array(32)].map((_, d) => `<div class="zb" data-d="${d}" title="z${d}"><i class="fill"></i><i class="orig"></i></div>`).join("");
    lat.querySelectorAll(".zb").forEach(bar => {
      const set = e => { const d = +bar.dataset.d, r = bar.getBoundingClientRect(), t = 1 - Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1);
        const lo = NL.cfg.latent_lo[d], hi = NL.cfg.latent_hi[d]; zMod[d] = lo + t * (hi - lo); steer(); };
      bar.addEventListener("pointerdown", e => { e.stopPropagation(); bar.setPointerCapture(e.pointerId); bar.dataset.drag = 1; set(e); });
      bar.addEventListener("pointermove", e => { if (bar.dataset.drag) set(e); });
      bar.addEventListener("pointerup", () => delete bar.dataset.drag);
      bar.addEventListener("click", e => e.stopPropagation());
    });
  }
  function renderLatent() {
    const z0 = NL.state.now.z;
    $("#lat", root).querySelectorAll(".zb").forEach(bar => {
      const d = +bar.dataset.d, lo = NL.cfg.latent_lo[d], hi = NL.cfg.latent_hi[d], y = v => 100 * (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo);
      const zero = y(0), val = y(zMod[d]);
      const f = bar.querySelector(".fill"); f.style.bottom = Math.min(zero, val) + "%"; f.style.height = Math.max(Math.abs(val - zero), 1.2) + "%";
      bar.querySelector(".orig").style.bottom = y(z0[d]) + "%";
      bar.classList.toggle("moved", changed(d));
    });
    $("#act-z", root).textContent = anyChanged() ? `${zMod.filter((_, d) => changed(d)).length} changed by you` : "drag a bar";
  }
  async function steer() {
    renderLatent();
    if (busy) { again = true; return; }
    busy = true; again = false;
    if (anyChanged()) { const d = await NL.decode(zMod.slice()); steered = { ...d, ...NL.errorOf(NL.state.now.x, d.rec) }; }
    else steered = null;
    render(); busy = false; if (again) steer();
  }

  function readline(v) {
    const b = NL.state.now, L = NL.level(b.score);
    $("#rlScore", root).textContent = b.score.toFixed(6);
    $("#rlLevel", root).textContent = NL.LEVEL_TEXT[L]; $("#rlLevel", root).dataset.level = L;
    $("#rlSteer", root).innerHTML = steered
      ? `With your edited code, the redraw differs from your weld by <b class="num">${v.score.toFixed(6)}</b> (mean squared error). Changed: ${[...Array(32).keys()].filter(changed).map(d => "z" + d).join(", ")}.`
      : `Drag any of the 32 bars in the middle to change the code the decoder receives.`;
  }

  /* lines in the order of the computation; they pulse once when a real inference has just finished */
  function flows(pulse) {
    const svg = $("#flows", root), scene = $("#scene", root).getBoundingClientRect();
    const pts = [...root.querySelectorAll(".station")].map(el => { const r = (el.querySelector(".stack,.latent")).getBoundingClientRect();
      return [r.left + r.width / 2 - scene.left, r.top + r.height / 2 - scene.top, r.width / 2]; });
    svg.setAttribute("viewBox", `0 0 ${scene.width} ${scene.height}`);
    svg.innerHTML = pts.slice(1).map((p, i) => { const a = pts[i];
      return `<path class="flow${pulse ? " pulse" : ""}" d="M${a[0] + a[2] * .7},${a[1]} C${(a[0] + p[0]) / 2},${a[1]} ${(a[0] + p[0]) / 2},${p[1]} ${p[0] - p[2] * .7},${p[1]}" style="--d:${i * 70}ms"/>`; }).join("");
  }

  /* ---- inspector: every channel of the clicked layer, real values ---- */
  let openK = null;
  function inspect(k, refresh) {
    const s = STATIONS.find(t => t.k === k), v = view(), dr = $("#drawer");
    openK = k; dr.classList.add("open"); if (!refresh) dr.scrollTop = 0;
    $("#drName").textContent = s.name; $("#drOp").textContent = s.op;
    const body = $("#drBody");
    if (k === "z") {
      const z0 = NL.state.now.z, inf = NL.cfg.influence, mi = Math.max(...inf);
      body.innerHTML = `<p class="note">Measured influence = ${NL.cfg.influence_rule}. Ranges = ${NL.cfg.latent_range_rule}. Numbers have no assigned meaning.</p>
        <table class="ztab num"><tr><th>number</th><th>your weld</th><th>now</th><th>influence on the redraw</th></tr>${[...Array(32).keys()].sort((a, b) => inf[b] - inf[a]).map(d =>
          `<tr class="${changed(d) ? "moved" : ""}"><td>z${d}</td><td>${z0[d].toFixed(2)}</td><td>${zMod[d].toFixed(2)}</td><td><i style="width:${100 * inf[d] / mi}%"></i>${inf[d].toFixed(4)}</td></tr>`).join("")}</table>`;
      return;
    }
    if (s.C === 1) {
      body.innerHTML = `<div class="big3"><figure><canvas id="i1"></canvas><figcaption>${s.k === "in" ? "your weld" : "reconstruction"}</figcaption></figure>
        ${s.k === "out" ? `<figure><canvas id="i2"></canvas><figcaption>error map (weld − reconstruction)², scale 0–${NL.cfg.error_map_max}</figcaption></figure>` : ""}</div>
        <p class="note num">${s.k === "out" ? `Mean of the error map: ${v.score.toFixed(6)} (${NL.LEVEL_TEXT[NL.level(v.score)]}).` : "This is exactly what the model receives."}</p>`;
      NL.paint($("#i1"), s.k === "in" ? v.x : v.rec, 64); if (s.k === "out") NL.paint($("#i2"), v.err, 64, "heat", NL.cfg.error_map_max);
      return;
    }
    const st = NL.channelStats(v[s.src], s.C, s.H), HH = s.H * s.H;
    body.innerHTML = `<p class="note num">${s.C} channels of ${s.H}×${s.H}. Mean activation ${st.overall.toFixed(3)} · max ${st.max.toFixed(2)} ·
      ${(100 * st.zeroShare).toFixed(0)}% of values are exactly zero (ReLU). Sorted by mean activation for this input; one colour scale (0 – ${st.max.toFixed(2)}).
      Ranking is a measurement, not a meaning.${steered && k[0] === "d" ? " Showing the decoder's response to your edited code." : ""}</p>
      <div class="chgrid">${st.order.map(c => `<figure><canvas data-c="${c}"></canvas><figcaption class="num">ch ${c} · ${st.mean[c].toFixed(2)}</figcaption></figure>`).join("")}</div>`;
    body.querySelectorAll("canvas[data-c]").forEach(cv => NL.paint(cv, v[s.src], s.H, "act", st.max || 1, +cv.dataset.c * HH));
  }
  function closeDrawer() { openK = null; $("#drawer").classList.remove("open"); }

  function show() { if (dirty || !zMod) { dirty = false; render(); } else flows(); }
  /* used by the demo: move one latent number to a value inside its real range, decoder answers */
  function setLatent(d, v) { zMod[d] = v; return steer(); }
  return { build, render, show, setLatent, STATIONS, get zMod() { return zMod; } };
})();
