/* NEURAL / LAB · "Is it a neural network?" room: evidence only.
   - the network read from the published weight file (proof.json, made by playground/proof.py)
   - live control experiment: the trained network vs its untrained twin (same architecture, random weights)
   - the test-set result of that control, the learned first-layer filters, and the training record */
"use strict";
const Proof = (() => {
  let root, P = null, control = null, busy = false, again = false;
  const $ = s => root.querySelector(s);
  const fmt = n => n.toLocaleString("en-US");

  async function build(el) {
    root = el;
    try { P = await NL.json("proof.json"); }
    catch { $("#pIntro").textContent = "proof.json is missing: run  python -m neural_lab.playground.proof  once, then reload."; return; }
    try { control = await ort.InferenceSession.create("control.onnx"); } catch { control = null; }
    anatomy(); results(); filters(); training();
    NL.on(() => { if (!root.hidden) live(); });
  }
  function show() { live(); }

  /* the network, read from the weight file */
  function anatomy() {
    const groups = P.layers.map(r => `<tr><td>${r.layer}</td><td>${r.type}</td>
      <td class="num">${Object.entries(r.weights).map(([k, s]) => `${k} ${s.join("×")}`).join("<br>") || "–"}</td>
      <td class="num">${r.output.join("×")}</td><td class="num r">${r.learned ? fmt(r.learned) : "–"}</td></tr>`).join("");
    $("#pTable").innerHTML = `<tr><th>layer</th><th>type</th><th>learned weights</th><th>output</th><th class="r">learned values</th></tr>${groups}
      <tr class="tot"><td colspan="4">Total, counted from the published weight file</td><td class="num r">${fmt(P.total_learned)}</td></tr>`;
    $("#pFacts").innerHTML = `<b class="num">${fmt(P.total_learned)}</b> learned values in ${P.layers.filter(r => r.learned).length} layers ·
      <b class="num">${fmt(P.multiply_adds_per_weld)}</b> multiply-adds to process one weld · file <span class="num">${P.weights_file}</span>,
      SHA-256 <span class="num">${P.weights_sha256.slice(0, 16)}…</span>`;
  }

  /* live: the same weld through the trained network and through its untrained twin */
  async function live() {
    if (!P || !NL.state.now) return;
    if (busy) { again = true; return; }
    busy = true; again = false;
    const s = NL.state, x = s.cur.slice();
    NL.paint($("#pIn"), x, 64); NL.paint($("#pT"), s.now.rec, 64);
    $("#pTs").textContent = `error ${s.now.score.toFixed(6)}`;
    if (control) {
      const out = await control.run({ image: new ort.Tensor("float32", x, [1, 1, 64, 64]) });
      const rec = Float32Array.from(out.reconstruction.data), e = NL.errorOf(x, rec);
      NL.paint($("#pU"), rec, 64); $("#pUs").textContent = `error ${e.score.toFixed(6)}`;
    } else $("#pUs").textContent = "control.onnx missing";
    busy = false; if (again) live();
  }

  /* the control experiment on all test tiles */
  function results() {
    const c = P.control, rows = [["Trained network", c.auc_trained, "var(--ember)"], ["Same network, untrained", c.auc_untrained, "var(--sand)"],
      ["No network: pixel contrast", c.auc_contrast_baseline, "var(--ash)"]];
    const W = 640, L = 210, R = 60, X = v => L + (W - L - R) * v, rowH = 46;
    let g = `<line x1="${X(.5)}" x2="${X(.5)}" y1="6" y2="${rows.length * rowH + 6}" stroke="var(--ash)" stroke-dasharray="3 4"/>
      <text x="${X(.5)}" y="${rows.length * rowH + 22}" class="ax" text-anchor="middle">0.5 = coin flip</text>
      <text x="${X(1)}" y="${rows.length * rowH + 22}" class="ax" text-anchor="middle">1 = perfect</text>`;
    rows.forEach(([n, v, col], i) => { const y = 10 + i * rowH;
      g += `<text x="${L - 14}" y="${y + 17}" class="lbl" text-anchor="end">${n}</text>
            <rect x="${L}" y="${y}" width="${X(v) - L}" height="24" fill="${col}"/>
            <text x="${X(v) + 8}" y="${y + 17}" class="lbl num">${v.toFixed(3)}</text>`; });
    $("#pAuc").innerHTML = g;
    $("#pAucNote").textContent = `ROC AUC on the ${fmt(c.n_test)} test tiles of 4 welds the network never trained on (${fmt(c.n_sound)} sound, ${fmt(c.n_defect)} defect). ` +
      `Average error on sound tiles: trained ${c.mse_sound_trained.toFixed(6)}, untrained ${c.mse_sound_untrained.toFixed(6)}. ${c.rule}.`;
  }

  /* the 32 first-layer filters: 3x3 numbers each, learned vs random */
  function filters() {
    for (const [id, K] of [["#pKt", P.filters_trained], ["#pKu", P.filters_untrained]]) {
      const box = $(id), m = Math.max(...K.flat(2).map(Math.abs)) || 1;
      box.innerHTML = "";
      K.forEach(k => { const c = document.createElement("canvas"); NL.paint(c, k.flat(), 3, "div", m); box.appendChild(c); });
    }
  }

  function training() {
    const t = P.training;
    $("#pTrain").innerHTML = `<li>Trained on ${t.data}, batch ${t.batch}.</li><li>Loss: ${t.loss}. Optimiser: ${t.optimizer}.</li>
      <li>${t.early_stopping}: best epoch <b class="num">${t.best_epoch}</b> of ${t.epochs_run}, validation error <b class="num">${t.val_mse.toFixed(5)}</b>, ${t.seconds} s on an ${t.hardware}.</li>`;
    $("#pSrc").textContent = `Source: ${t.source}.`;
  }
  return { build, show, live };
})();
