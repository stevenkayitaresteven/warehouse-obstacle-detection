/*
 * Weekly-progress deck (6 slides) built from measured results.
 *
 *   python -m presentation.collect_data          # -> presentation/deck_data.json
 *   npm install pptxgenjs react-icons react react-dom sharp jszip
 *   node presentation/build_deck.js [deck_data.json] [out.pptx]
 */
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const DATA = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, "deck_data.json")));
const OUT = process.argv[3] || path.join(__dirname, "Group3_Weekly_Progress.pptx");
const SHOT = path.join(__dirname, "demo_screenshot.png");
const APPLY_THEME = process.env.APPLY_THEME_JS;

// ---------------------------------------------------------------------------
// Theme: warehouse charcoal + safety amber
const THEME = {
  name: "Warehouse Safety",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2A30", lt1: "FFFFFF", dk2: "3C4A52", lt2: "EEF1F3",
    accent1: "F2A900", // safety amber
    accent2: "1B998B", // success teal
    accent3: "D1495B", // collision red
    accent4: "5B7083", // slate
    accent5: "7FB3D5", // existing components (light blue)
    accent6: "E07A1F", // new logic (orange)
    hlink: "1B998B", folHlink: "5B7083",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Group 3 - Weekly progress";
pres.author = "Group 3";
const C = pres.SchemeColor;

const pct = (x) => (x === null || x === undefined ? "-" : `${Math.round(100 * x)}%`);
const pct1 = (x) => `${(100 * x).toFixed(1)}%`;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------------------------------------------------------------------------
// Layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.5, w: 5.3, h: 1.85,
      fontSize: 32, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 },
      text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.5, y: 2.5, w: 5.3, h: 0.7,
      fontSize: 14, color: C.accent1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.4, 0.5, 0.4, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9.0, h: 0.6,
      fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.5, y: 0.82, w: 9.0, h: 0.38,
      fontSize: 14, color: C.accent4, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Group 3 · Image and Scene Recognition · Weekly progress",
      options: { x: 0.5, y: 5.28, w: 6, h: 0.25, fontSize: 9, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 9.1, y: 5.28, w: 0.4, h: 0.25, fontSize: 9, color: C.accent4, align: "right" },
});

// Shared chart styling: same label colors / sizes everywhere
const AXIS = {
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "D9DEE2", size: 0.5 }, catGridLine: { style: "none" },
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: HEX.dk1,
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.dk2,
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.dk1,
};

function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill },
    objectName: name,
  });
}

(async () => {
  const ctl = DATA.controllers;
  const full = ctl.ppo_shield;
  const ICON = {
    camera: await icon(fa.FaCamera, HEX.dk1),
    eye: await icon(fa.FaEye, HEX.dk1),
    layer: await icon(fa.FaLayerGroup, HEX.dk1),
    brain: await icon(fa.FaBrain, HEX.dk1),
    shield: await icon(fa.FaShieldAlt, HEX.dk1),
    check: await icon(fa.FaCheckCircle, HEX.accent2),
    bug: await icon(fa.FaBug, HEX.accent3),
    route: await icon(fa.FaRoute, HEX.accent2),
    arrow: await icon(fa.FaArrowRight, HEX.accent4),
    flag: await icon(fa.FaFlagCheckered, HEX.accent1),
  };

  // =========================================================================
  // 1. Title
  pres.addSection({ title: "Overview" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
  s.addText("Weekly progress: our robot now drives itself through a warehouse",
    { placeholder: "title" });
  s.addText("Group 3 · Camera-Based Obstacle Detection for Autonomous Warehouse Robots",
    { placeholder: "body" });
  if (fs.existsSync(SHOT)) {
    s.addImage({ path: SHOT, x: 6.05, y: 0.55, w: 3.5, h: 3.5 * 0.664,
      sizing: { type: "cover", w: 3.5, h: 3.5 * 0.664 }, objectName: "Demo screenshot",
      altText: "Simulated warehouse: robot, racks, workers and camera rays" });
    s.addText("Live demo: python app.py", { x: 6.05, y: 2.95, w: 3.5, h: 0.3,
      fontSize: 11, color: C.background2, italic: true, margin: 0, isTextBox: true });
  }
  const stats = [
    [pct(full.success_rate), "of test runs reach the goal (PPO + shield)", C.accent1],
    [pct(full.collision_rate), "collisions with the safety shield on", C.accent2],
    [pct1(DATA.recall.full), "of obstacles found by detector + depth", C.accent5],
  ];
  stats.forEach(([v, l, col], i) => {
    const x = 0.5 + i * 3.05;
    s.addText(v, { x, y: 3.55, w: 2.85, h: 0.85, fontSize: 44, bold: true, color: col,
      margin: 0, isTextBox: true, objectName: `Stat ${i + 1}` });
    s.addText(l, { x, y: 4.4, w: 2.75, h: 0.55, fontSize: 12, color: C.background2,
      margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes(
    "This week we moved from perception-only work to a robot that drives itself. " +
    "We built a simulated warehouse, trained a PPO reinforcement-learning policy for 2 million steps, " +
    "and added the safety shield. The three numbers come from " + DATA.episodes +
    " test episodes on warehouse layouts the robot never saw during training. All navigation " +
    "results are from simulation, not a physical robot.");

  // =========================================================================
  // 2. What we built
  pres.addSection({ title: "What we built" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What we built" });
  s.addText("What we built this week", { placeholder: "title" });
  s.addText("All four layers of our design now run end to end in a simulated warehouse",
    { placeholder: "body" });
  const stages = [
    ["Camera", "120° view, 6 m range", ICON.camera, C.accent5, "Existing"],
    ["Detector + depth", "person, pallet, forklift + any surface", ICON.eye, C.accent5, "Existing"],
    ["Fusion", "unknown obstacles, uncertainty, tracking", ICON.layer, C.accent6, "New logic"],
    ["PPO policy", "GO · SLOW · STOP · LEFT · RIGHT", ICON.brain, C.accent6, "New logic"],
    ["Safety shield", "hard stop · person · uncertainty", ICON.shield, C.accent6, "New logic"],
  ];
  const bw = 1.6, gap = 0.25, y0 = 1.4;
  stages.forEach(([t, d, ic, col], i) => {
    const x = 0.5 + i * (bw + gap);
    card(s, x, y0, bw, 1.55, `Stage ${t}`, col);
    s.addImage({ data: ic, x: x + bw / 2 - 0.2, y: y0 + 0.15, w: 0.4, h: 0.4 });
    s.addText(t, { x: x + 0.08, y: y0 + 0.6, w: bw - 0.16, h: 0.35, fontSize: 14, bold: true,
      color: C.text1, align: "center", margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.08, y: y0 + 0.95, w: bw - 0.16, h: 0.55, fontSize: 10,
      color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true });
    if (i < stages.length - 1) {
      s.addImage({ data: ICON.arrow, x: x + bw + 0.04, y: y0 + 0.68, w: 0.17, h: 0.17 });
    }
  });
  // legend
  [["Existing components we reuse", C.accent5], ["New logic we added", C.accent6]].forEach(([t, col], i) => {
    const x = 0.5 + i * 3.0;
    s.addShape(pres.shapes.OVAL, { x, y: 3.1, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col } });
    s.addText(t, { x: x + 0.24, y: 3.04, w: 2.6, h: 0.28, fontSize: 11, color: C.text2, margin: 0, isTextBox: true });
  });
  // done list
  const done = [
    "Warehouse simulator (Gymnasium): racks, workers, forklifts, pallets, fallen boxes",
    "PPO trained 2 M steps with domain randomisation (layouts, noise, lighting cues)",
    "Safety shield with decision log + event-triggered path replanning",
    "Interactive Gradio demo: click to drop obstacles, switch perception modes",
  ];
  done.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6, y = 3.55 + row * 0.75;
    s.addImage({ data: ICON.check, x, y: y + 0.05, w: 0.26, h: 0.26 });
    s.addText(t, { x: x + 0.38, y, w: 4.0, h: 0.62, fontSize: 12, color: C.text1,
      valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    "Blue boxes are components we reuse, orange boxes are our new logic, the same colour code as our design document. " +
    "In the simulator, the detector branch labels known classes and can miss them, the depth branch sees any surface " +
    "but has noise, shadow false alarms and fails on shiny stretch wrap, exactly the weaknesses we listed. " +
    "The policy never sees pixels, only the obstacle map, which is our sim-to-real argument.");

  // =========================================================================
  // 3. Training curve
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("The PPO policy learned to reach the goal", { placeholder: "title" });
  s.addText("Mean episode reward during training (higher = faster, safer trips)", { placeholder: "body" });
  const cv = DATA.curve;
  s.addChart(pres.charts.LINE, [{ name: "Mean episode reward", labels: cv.steps_m.map(String), values: cv.reward }], {
    x: 0.4, y: 1.3, w: 5.9, h: 3.85, ...AXIS,
    chartColors: [HEX.accent6], lineSize: 2.5, lineDataSymbol: "none", showLegend: false,
    showTitle: false, catAxisTitle: "Training steps (millions)", showCatAxisTitle: true,
    catAxisTitleFontSize: 11, catAxisTitleColor: HEX.dk2, catAxisTitleFontFace: "+mn-lt",
    valAxisTitle: "Reward", showValAxisTitle: true, valAxisTitleFontSize: 11,
    valAxisTitleColor: HEX.dk2, valAxisTitleFontFace: "+mn-lt",
    catAxisLabelFrequency: Math.max(1, Math.round(cv.steps_m.length / 8)), catAxisLabelRotate: 0,
    objectName: "Training curve",
  });
  const iP2 = Math.max(0, cv.steps_m.findIndex((x) => x > 0.6) - 1);
  const r0 = cv.reward[iP2], r1 = cv.reward[cv.reward.length - 1];
  const notes3 = [
    [ICON.route, "Phase 1 (0–0.6 M)", "Learned to stop safely, but got stuck behind pallets on its path"],
    [ICON.flag, "Phase 2 (0.6–2 M)", "Path replanning around seen obstacles: reward " + r0.toFixed(0) + " → " + r1.toFixed(0)],
    [ICON.bug, "Bug found and fixed", "Robot 'farmed' reward by driving in circles; we fixed the reward"],
  ];
  notes3.forEach(([ic, h, t], i) => {
    const y = 1.35 + i * 1.27;
    card(s, 6.55, y, 2.95, 1.12, `Training note ${i + 1}`);
    s.addImage({ data: ic, x: 6.7, y: y + 0.15, w: 0.3, h: 0.3 });
    s.addText(h, { x: 7.1, y: y + 0.1, w: 2.3, h: 0.38, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(t, { x: 6.7, y: y + 0.5, w: 2.7, h: 0.58, fontSize: 11, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    "The reward goes up as the robot learns. In phase 1 it learned that stopping is safe, but it stayed stuck when a pallet " +
    "was on its planned path. We added event-triggered replanning, which is what real warehouse robots do with their costmap. " +
    "We also found reward hacking: the robot drove in circles to collect progress reward when the path changed. We now measure " +
    "progress on the fixed map distance and retrained from the phase-1 checkpoint. Training takes about 45 minutes on 4 CPU cores.");

  // =========================================================================
  // 4. Navigation comparison
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("RL + safety shield beats the rule-based baseline", { placeholder: "title" });
  s.addText(`${DATA.episodes} test runs per controller on unseen warehouse layouts (simulation)`, { placeholder: "body" });
  const order = [["ppo_shield", "PPO + shield"], ["ppo", "PPO only"], ["rule", "Rule-based"], ["rule_shield", "Rule + shield"]];
  const labels = order.map((o) => o[1]);
  s.addChart(pres.charts.BAR, [
    { name: "Reached goal", labels, values: order.map((o) => Math.round(100 * ctl[o[0]].success_rate)) },
    { name: "Collision", labels, values: order.map((o) => Math.round(100 * ctl[o[0]].collision_rate)) },
    { name: "Time limit", labels, values: order.map((o) => Math.round(100 * ctl[o[0]].timeout_rate)) },
  ], {
    x: 0.4, y: 1.3, w: 6.1, h: 3.85, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    ...AXIS, chartColors: [HEX.accent2, HEX.accent3, "B8C2CA"], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 100, valAxisMinVal: 0,
    valAxisLabelFormatCode: '0"%"', showLegend: true, legendPos: "b", objectName: "Controller comparison",
  });
  const t = (k) => ctl[k].mean_time_to_goal_s;
  const side = [
    [pct(ctl.ppo_shield.collision_rate), "collisions with PPO + shield", C.accent2],
    [pct(ctl.ppo.collision_rate), "collisions with PPO alone: the shield is needed", C.accent3],
    [t("ppo_shield") ? t("ppo_shield").toFixed(0) + " s" : "-", "average time to goal (PPO + shield)", C.accent6],
  ];
  side.forEach(([v, l, col], i) => {
    const y = 1.35 + i * 1.27;
    card(s, 6.75, y, 2.75, 1.12, `Result ${i + 1}`);
    s.addText(v, { x: 6.9, y: y + 0.08, w: 2.5, h: 0.6, fontSize: 30, bold: true, color: col, margin: 0, isTextBox: true });
    s.addText(l, { x: 6.9, y: y + 0.66, w: 2.5, h: 0.42, fontSize: 11, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    "Green is reaching the goal, red is a collision, grey is running out of time. The rule-based baseline is our " +
    "'stop if closer than 1 m' controller. PPO alone is faster but still collides sometimes; with the safety shield on top " +
    "it never collided in the test runs. Remaining failures are time-outs, where the robot waits or hesitates, which is the safe failure mode.");

  // =========================================================================
  // 5. Confusion matrix + ablation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Depth fusion finds what the detector misses", { placeholder: "title" });
  s.addText("Confusion matrix: true obstacle (rows) vs what our perception reported (columns), % of frames", { placeholder: "body" });
  const cmTable = (cm, x, y, w, heading, objName) => {
    const m = cm.matrix;
    const head = ["True \\ Predicted", "person", "pallet", "forklift", "unknown", "missed"];
    const rows = [head.map((h) => ({ text: h, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, fontSize: 9, align: "center" } }))];
    m.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      const cells = [{ text: cm.rows_true[i], options: { bold: true, fontSize: 10, color: HEX.dk1, fill: { color: HEX.lt2 } } }];
      row.forEach((v, j) => {
        const p = v / tot;
        const good = j === i || (i === 3 && j === 3);
        const fill = p < 0.005 ? "FFFFFF" : (j === 4 ? (p > 0.5 ? "E6A1AB" : "F6D5DA")
          : (good ? (p > 0.75 ? "8FD3C9" : "CBECE7") : "FBE3C4"));
        cells.push({ text: p < 0.005 ? "·" : `${Math.round(100 * p)}%`,
          options: { fontSize: 11, align: "center", color: HEX.dk1, fill: { color: fill }, bold: p > 0.5 } });
      });
      rows.push(cells);
    });
    s.addText(heading, { x, y, w, h: 0.3, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addTable(rows, { x, y: y + 0.32, w, colW: [0.96, 0.58, 0.58, 0.64, 0.78, 0.62],
      rowH: 0.3, border: { type: "solid", color: "D9DEE2", pt: 0.75 }, fontFace: THEME.bodyFontFace,
      objectName: objName });
  };
  cmTable(DATA.confusion_detector_only, 0.5, 1.3, 4.16, "Detector only (existing method)", "Confusion detector only");
  cmTable(DATA.confusion_full, 5.0, 1.3, 4.16, "Our fusion (detector + depth)", "Confusion full");
  const ab = DATA.ablation;
  const abOrder = [["detector_only", "Detector only"], ["fusion", "+ depth fusion"], ["fusion_uncertainty", "+ uncertainty"], ["full", "+ tracking (full)"]];
  s.addChart(pres.charts.BAR, [{ name: "Collision rate", labels: abOrder.map((a) => a[1]),
    values: abOrder.map((a) => Math.round(100 * ab[a[0]].collision_rate)) }], {
    x: 0.4, y: 3.25, w: 5.3, h: 1.95, barDir: "bar", ...AXIS, chartColors: [HEX.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisHidden: true,
    valGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Collision rate, PPO + shield (ablation)",
    catAxisOrientation: "maxMin", objectName: "Ablation chart",
  });
  card(s, 6.0, 3.4, 3.5, 1.7, "Recall card");
  s.addText(pct1(DATA.recall.full), { x: 6.15, y: 3.48, w: 1.6, h: 0.6, fontSize: 28, bold: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addText(pct1(DATA.recall.detector_only), { x: 7.8, y: 3.48, w: 1.6, h: 0.6, fontSize: 28, bold: true, color: C.accent3, margin: 0, isTextBox: true });
  s.addText("obstacle recall, fusion", { x: 6.15, y: 4.05, w: 1.6, h: 0.45, fontSize: 11, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  s.addText("detector only", { x: 7.8, y: 4.05, w: 1.6, h: 0.45, fontSize: 11, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  s.addText(`Unknown fallen boxes: ${pct(DATA.recall.unknown_detector_only)} → ${pct(DATA.recall.unknown_full)} found`,
    { x: 6.15, y: 4.55, w: 3.25, h: 0.45, fontSize: 12, bold: true, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addNotes(
    "Each row is a true obstacle type, each column is what our perception said, counted per frame in simulation. " +
    "The detector alone never reports the fallen boxes because they are not a trained class, so 100% of them are missed, " +
    "and it misses about 10% of people and pallets. With depth fusion, almost nothing is missed: a person the detector missed " +
    "still appears as an 'unknown obstacle', so the robot still avoids it. The bar chart is our ablation study: " +
    "with detector only, the robot crashes into what it cannot see. Note: the detector here is simulated with 80-97% recall; " +
    "our real YOLO11n baseline on LOCO is mAP@50 31.7%, which we are still improving.");

  // =========================================================================
  // 6. Uncertainty + safety + next steps
  pres.addSection({ title: "Next steps" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Next steps" });
  s.addText("Shiny wrap raises uncertainty, so the robot slows", { placeholder: "title" });
  s.addText("Uncertainty score when a pallet is within 3 m in front of the robot (share of frames)", { placeholder: "body" });
  const U = DATA.uncertainty;
  const bins = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"];
  const norm = (h) => { const tot = h.reduce((a, b) => a + b, 0) || 1; return h.map((v) => Math.round(100 * v / tot)); };
  s.addChart(pres.charts.BAR, [
    { name: "Normal pallets", labels: bins, values: norm(U["normal pallets"].hist) },
    { name: "Stretch-wrapped (reflective)", labels: bins, values: norm(U["reflective pallets"].hist) },
  ], {
    x: 0.4, y: 1.3, w: 5.0, h: 2.75, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
    ...AXIS, chartColors: [HEX.accent5, HEX.accent1], showLegend: true, legendPos: "t",
    valAxisLabelFormatCode: '0"%"', catAxisTitle: "Uncertainty score (≥ 0.35 → slow down)",
    showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisTitleColor: HEX.dk2, catAxisTitleFontFace: "+mn-lt",
    objectName: "Uncertainty histogram",
  });
  const sm = [
    [pct(U["normal pallets"].slow_rate) + " → " + pct(U["reflective pallets"].slow_rate), "slow-down triggered: normal → shiny pallets"],
    [pct(DATA.shield_step_out), "worker-steps-out tests where the shield stopped in time"],
    [DATA.speed.policy_plus_shield_ms.toFixed(2) + " ms", "policy + shield decision time per frame on a CPU"],
  ];
  sm.forEach(([v, l], i) => {
    const x = 0.5 + i * 1.65;
    s.addText(v, { x, y: 4.15, w: 1.6, h: 0.45, fontSize: 18, bold: true, color: C.accent6, margin: 0, isTextBox: true });
    s.addText(l, { x, y: 4.58, w: 1.55, h: 0.62, fontSize: 10, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  card(s, 5.75, 1.3, 3.75, 3.85, "Next steps card");
  s.addText("Next week", { x: 5.95, y: 1.42, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const nxt = [
    "Add the COCO person class and retrain YOLO11n (mAP@50 now 31.7%)",
    "Run Depth Anything V2 on our hard-case videos and measure distance error",
    "Train 3 random seeds and report mean ± spread",
    "Reduce time-outs: train with the shield in the loop",
    "Webcam-on-a-trolley demo with live boxes, depth and actions",
  ];
  s.addText(nxt.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < nxt.length - 1 } })),
    { x: 5.95, y: 1.9, w: 3.4, h: 3.1, fontSize: 12, color: C.text1, paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true });
  s.addNotes(
    "Our uncertainty-aware slowdown: on shiny stretch-wrapped pallets the depth reading jumps between frames and disagrees " +
    "with the detector box, so the score shifts right and the robot slows down far more often, while normal pallets rarely " +
    "trigger it. In scripted tests where a worker suddenly steps out about one metre ahead, the shield stopped the robot every time. " +
    "Next week we connect the real models: YOLO with the person class, Depth Anything on our hard-case footage, and three seeds for robustness.");

  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
