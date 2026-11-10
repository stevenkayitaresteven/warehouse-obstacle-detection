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
  const Y = DATA.yolo || {};
  const YE = Y.eval || null;
  const YC = Y.curve || null;
  const DS = Y.dataset || null;
  const ICON = {
    camera: await icon(fa.FaCamera, HEX.dk1),
    eye: await icon(fa.FaEye, HEX.dk1),
    layer: await icon(fa.FaLayerGroup, HEX.dk1),
    brain: await icon(fa.FaBrain, HEX.dk1),
    shield: await icon(fa.FaShieldAlt, HEX.dk1),
    check: await icon(fa.FaCheckCircle, HEX.accent2),
    bug: await icon(fa.FaBug, HEX.accent3),
    db: await icon(fa.FaDatabase, HEX.accent6),
    arrow: await icon(fa.FaArrowRight, HEX.accent4),
  };
  const map50 = YE ? YE.fine_tuned.mAP50 : null;

  // =========================================================================
  // 1. Title
  pres.addSection({ title: "Overview" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
  s.addText("Weekly progress: a warehouse detector and a robot that drives itself",
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
    [pct(map50), "mAP@50 of our fine-tuned YOLO11n on the test images", C.accent1],
    [pct(full.success_rate), "of robot test runs reach the goal (PPO + shield)", C.accent5],
    [pct(full.collision_rate), "collisions with the safety shield on", C.accent2],
  ];
  stats.forEach(([v, l, col], i) => {
    const x = 0.5 + i * 3.05;
    s.addText(v, { x, y: 3.55, w: 2.85, h: 0.85, fontSize: 44, bold: true, color: col,
      margin: 0, isTextBox: true, objectName: `Stat ${i + 1}` });
    s.addText(l, { x, y: 4.4, w: 2.75, h: 0.55, fontSize: 12, color: C.background2,
      margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes(
    "This week we did two things. First, image recognition: we fine-tuned YOLO11n on a warehouse obstacle dataset " +
    "with people, forklifts and packages, and measured it on test images it never saw. Second, robotics: we trained a " +
    "reinforcement-learning robot that uses the detector's output to drive through a simulated warehouse, with a " +
    "safety shield. Navigation numbers are from simulation, not a physical robot.");

  // =========================================================================
  // 2. What we built
  pres.addSection({ title: "What we built" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "What we built" });
  s.addText("What we built this week", { placeholder: "title" });
  s.addText("All four layers of our design now run end to end", { placeholder: "body" });
  const stages = [
    ["Camera", "single RGB camera, 120° view", ICON.camera, C.accent5],
    ["YOLO11n + depth", "fine-tuned detector + Depth Anything V2", ICON.eye, C.accent5],
    ["Fusion", "unknown obstacles, uncertainty, tracking", ICON.layer, C.accent6],
    ["PPO policy", "GO · SLOW · STOP · LEFT · RIGHT", ICON.brain, C.accent6],
    ["Safety shield", "hard stop · person · uncertainty", ICON.shield, C.accent6],
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
  [["Existing models we fine-tune / reuse", C.accent5], ["New logic we added", C.accent6]].forEach(([t, col], i) => {
    const x = 0.5 + i * 3.4;
    s.addShape(pres.shapes.OVAL, { x, y: 3.1, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col } });
    s.addText(t, { x: x + 0.24, y: 3.04, w: 3.0, h: 0.28, fontSize: 11, color: C.text2, margin: 0, isTextBox: true });
  });
  const nTrain = DS ? DS.images.train : 0;
  const done = [
    `Built a 3-class warehouse obstacle dataset (person, forklift, package) and fine-tuned YOLO11n on ${nTrain.toLocaleString()} images`,
    `Warehouse simulator + PPO robot trained ${DATA.total_steps_m} M steps with domain randomisation`,
    "Safety shield with decision log + path replanning around seen obstacles",
    "Interactive Gradio demo: click to drop obstacles, upload photos for detection",
  ];
  done.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6, y = 3.55 + row * 0.75;
    s.addImage({ data: ICON.check, x, y: y + 0.05, w: 0.26, h: 0.26 });
    s.addText(t, { x: x + 0.38, y, w: 4.0, h: 0.62, fontSize: 12, color: C.text1,
      valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    "Blue boxes are existing models we fine-tune or reuse; orange boxes are our new logic, the same colour code as our design document. " +
    "The detector is trained on real images. The robot's policy never sees pixels, only the obstacle map built from detections and depth, " +
    "which is our sim-to-real argument.");

  // =========================================================================
  // 3. Dataset + fine-tuning curve
  pres.addSection({ title: "Image recognition" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Image recognition" });
  s.addText("Fine-tuning YOLO11n on warehouse obstacles", { placeholder: "title" });
  s.addText("Dataset: workers and forklifts on industrial sites + parcels; split into train / val / test", { placeholder: "body" });
  if (DS) {
    const cls = ["person", "forklift", "package"];
    s.addChart(pres.charts.BAR, ["train", "val", "test"].map((sp) => ({
      name: sp, labels: cls, values: cls.map((c) => DS.instances[sp][c]) })), {
      x: 0.4, y: 1.3, w: 4.2, h: 2.6, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50,
      ...AXIS, chartColors: [HEX.accent6, HEX.accent5, HEX.accent4], showValue: true,
      dataLabelPosition: "outEnd", showLegend: true, legendPos: "t",
      showTitle: true, title: "Labelled objects per class", objectName: "Dataset chart",
    });
    const im = DS.images;
    s.addText([
      { text: `${im.train.toLocaleString()} train · ${im.val} val · ${im.test} test images`, options: { bold: true, breakLine: true } },
      { text: "Test images are never used for training", options: {} },
    ], { x: 0.55, y: 4.0, w: 4.0, h: 0.6, fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    s.addText("Sources: Ultralytics construction-ppe (person, forklift) and package-seg (parcels). LOCO is next.",
      { x: 0.55, y: 4.62, w: 4.0, h: 0.55, fontSize: 10, color: C.accent4, valign: "top", margin: 0, isTextBox: true });
  }
  if (YC) {
    s.addChart(pres.charts.LINE, [
      { name: "mAP@50", labels: YC.epoch.map(String), values: YC.map50 },
      { name: "mAP@50-95", labels: YC.epoch.map(String), values: YC.map50_95 },
    ], {
      x: 4.85, y: 1.3, w: 4.75, h: 3.85, ...AXIS, chartColors: [HEX.accent6, HEX.accent4],
      lineSize: 2.5, lineDataSymbol: "none", showLegend: true, legendPos: "t",
      showTitle: true, title: "Validation accuracy per training epoch",
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
      catAxisTitle: "Epoch", showCatAxisTitle: true, catAxisTitleFontSize: 11,
      catAxisTitleColor: HEX.dk2, catAxisTitleFontFace: "+mn-lt", objectName: "YOLO curve",
    });
  }
  s.addNotes(
    "We fine-tuned the COCO-pretrained YOLO11n, 2.6 million parameters, at 416 pixels on a CPU. The left chart shows how many " +
    "labelled objects each class has; forklifts are the rarest, as in LOCO. The right chart is validation mAP after every epoch: " +
    "it climbs quickly because we start from pretrained weights, which is the benefit of fine-tuning. Our planned LOCO training " +
    "on Colab with pallets and stillages is the next step.");

  // =========================================================================
  // 4. Detection results: confusion matrix + before/after
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Image recognition" });
  s.addText("Fine-tuning taught the detector new classes", { placeholder: "title" });
  s.addText(`Test set: ${YE ? YE.test_images : "-"} unseen images`, { placeholder: "body" });
  if (YE) {
    const lab = YE.confusion_matrix.labels;            // [person, forklift, package, background]
    const M = YE.confusion_matrix.matrix_pred_rows_true_cols;  // rows = predicted, cols = true
    const n = lab.length;
    const colTot = Array.from({ length: n }, (_, j) => M.reduce((a, r) => a + r[j], 0) || 1);
    const head = ["True → / Predicted ↓", ...lab.map((l) => l === "background" ? "missed" : l)];
    const rows = [head.map((h) => ({ text: h, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, fontSize: 10, align: "center" } }))];
    for (let i = 0; i < n; i++) {
      const name = lab[i] === "background" ? "false alarm" : lab[i];
      const cells = [{ text: name, options: { bold: true, fontSize: 10, color: HEX.dk1, fill: { color: HEX.lt2 } } }];
      for (let j = 0; j < n; j++) {
        const isBgBg = i === n - 1 && j === n - 1;
        const p = M[i][j] / colTot[j];
        const diag = i === j && !isBgBg;
        const fill = isBgBg ? "FFFFFF" : p < 0.005 ? "FFFFFF" : diag ? (p > 0.6 ? "8FD3C9" : "CBECE7") : (p > 0.2 ? "F6D5DA" : "FBE3C4");
        const txt = isBgBg ? "" : (j === n - 1 ? String(M[i][j]) : (p < 0.005 ? "·" : `${Math.round(100 * p)}%`));
        cells.push({ text: txt, options: { fontSize: 11, align: "center", color: HEX.dk1, fill: { color: fill }, bold: diag } });
      }
      rows.push(cells);
    }
    s.addText("Confusion matrix (share of each true class)", { x: 0.5, y: 1.3, w: 4.6, h: 0.3, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addTable(rows, { x: 0.5, y: 1.65, w: 4.6, colW: [1.2, 0.85, 0.85, 0.85, 0.85], rowH: 0.36,
      border: { type: "solid", color: "D9DEE2", pt: 0.75 }, fontFace: THEME.bodyFontFace, objectName: "YOLO confusion matrix" });
    s.addText("Columns = true class. Diagonal = correct. Last row = false alarms on background (count).",
      { x: 0.5, y: 3.55, w: 4.6, h: 0.45, fontSize: 10, color: C.accent4, valign: "top", margin: 0, isTextBox: true });
    const bva = YE.before_vs_after_AP50;
    const cls = ["person", "forklift", "package"];
    s.addChart(pres.charts.BAR, [
      { name: "Before (COCO-pretrained)", labels: cls, values: cls.map((c) => Math.round(100 * (bva[c].coco_pretrained || 0))) },
      { name: "After fine-tuning", labels: cls, values: cls.map((c) => Math.round(100 * (bva[c].fine_tuned || 0))) },
    ], {
      x: 5.3, y: 1.3, w: 4.3, h: 2.75, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50,
      ...AXIS, chartColors: ["B8C2CA", HEX.accent2], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: '0"%"', valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"',
      showLegend: true, legendPos: "t", showTitle: true, title: "AP@50 per class on the test set",
      objectName: "Before after chart",
    });
    const ft = YE.fine_tuned;
    const sm = [
      [pct(ft.mAP50), "mAP@50"], [pct(ft.mAP50_95), "mAP@50-95"],
      [pct(ft.precision), "precision"], [pct(ft.recall), "recall"],
      [Math.round(YE.cpu_fps_imgsz416) + " FPS", "on a laptop CPU"],
    ];
    sm.forEach(([v, l], i) => {
      const x = 0.5 + i * 1.83;
      card(s, x, 4.2, 1.7, 0.95, `Metric ${l}`);
      s.addText(v, { x: x + 0.1, y: 4.25, w: 1.5, h: 0.5, fontSize: 22, bold: true, color: C.accent6, margin: 0, isTextBox: true });
      s.addText(l, { x: x + 0.1, y: 4.74, w: 1.5, h: 0.35, fontSize: 11, color: C.text2, margin: 0, isTextBox: true });
    });
  }
  s.addNotes(
    "Left: the confusion matrix on the test images. Each column is a true class; the diagonal shows how often it was detected with the " +
    "right label; 'missed' means the object was not found. The last row counts false alarms on background. " +
    "Right: before fine-tuning, the COCO model can find people, but it has no forklift or package class at all, so it scores zero there. " +
    "After fine-tuning it detects all three classes. These are measured results on images the model never saw. Our earlier LOCO baseline was mAP@50 31.7%.");

  // =========================================================================
  // 5. Navigation
  pres.addSection({ title: "Robot navigation" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Robot navigation" });
  s.addText("RL + safety shield beats the rule-based baseline", { placeholder: "title" });
  s.addText(`Simulated warehouse, ${DATA.episodes} test runs per controller on unseen layouts`, { placeholder: "body" });
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
  const side = [
    [pct(ctl.ppo_shield.collision_rate), "collisions with PPO + shield", C.accent2],
    [pct(ctl.ppo.collision_rate), "collisions with PPO alone: the shield is needed", C.accent3],
    [`${DATA.total_steps_m} M`, "PPO training steps; bug found: robot drove in circles to farm reward", C.accent6],
  ];
  side.forEach(([v, l, col], i) => {
    const y = 1.35 + i * 1.27;
    card(s, 6.75, y, 2.75, 1.12, `Result ${i + 1}`);
    s.addText(v, { x: 6.9, y: y + 0.08, w: 2.5, h: 0.55, fontSize: 28, bold: true, color: col, margin: 0, isTextBox: true });
    s.addText(l, { x: 6.9, y: y + 0.62, w: 2.5, h: 0.46, fontSize: 11, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    "The robot learns with PPO in a simulated warehouse: reward for progress and reaching the goal, penalties for collisions, near-misses and harsh braking. " +
    "Green is reaching the goal, red is a collision, grey is running out of time. The rule-based baseline is 'stop if closer than 1 m'. " +
    "PPO alone still collides sometimes; with the shield it never collided. During training we found reward hacking: the robot drove in " +
    "circles to collect progress reward when its path was replanned; we fixed the reward and retrained.");

  // =========================================================================
  // 6. Fusion + safety + next steps
  pres.addSection({ title: "Next steps" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Next steps" });
  s.addText("Depth fusion catches what the detector misses", { placeholder: "title" });
  s.addText("Ablation in simulation: PPO + shield with more of our new logic switched on", { placeholder: "body" });
  const ab = DATA.ablation;
  const abOrder = [["detector_only", "Detector only"], ["fusion", "+ depth fusion"], ["fusion_uncertainty", "+ uncertainty"], ["full", "+ tracking (full)"]];
  s.addChart(pres.charts.BAR, [{ name: "Collision rate", labels: abOrder.map((a) => a[1]),
    values: abOrder.map((a) => Math.round(100 * ab[a[0]].collision_rate)) }], {
    x: 0.4, y: 1.3, w: 5.0, h: 2.3, barDir: "bar", ...AXIS, chartColors: [HEX.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisHidden: true,
    valGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Collision rate",
    catAxisOrientation: "maxMin", objectName: "Ablation chart",
  });
  const U = DATA.uncertainty;
  const sm6 = [
    [`${pct(DATA.recall.unknown_detector_only)} → ${pct(DATA.recall.unknown_full)}`, "unknown fallen boxes found (detector → fusion)"],
    [`${pct(U["normal pallets"].slow_rate)} → ${pct(U["reflective pallets"].slow_rate)}`, "slow-down rate: normal → shiny wrapped pallets"],
    [pct(DATA.shield_step_out), "worker-steps-out tests where the shield stopped in time"],
  ];
  sm6.forEach(([v, l], i) => {
    const x = 0.5 + i * 1.67;
    s.addText(v, { x, y: 3.8, w: 1.6, h: 0.45, fontSize: 18, bold: true, color: C.accent6, margin: 0, isTextBox: true });
    s.addText(l, { x, y: 4.26, w: 1.55, h: 0.8, fontSize: 10, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  card(s, 5.75, 1.3, 3.75, 3.85, "Next steps card");
  s.addText("Next week", { x: 5.95, y: 1.42, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const nxt = [
    "Fine-tune on LOCO (pallets, stillages) on Colab GPU, 640 px, 100 epochs",
    "Plug the fine-tuned detector into the robot's perception",
    "Run Depth Anything V2 on our hard-case videos; measure distance error",
    "Train 3 random seeds; report mean ± spread",
    "Webcam-on-a-trolley demo with live boxes, depth and actions",
  ];
  s.addText(nxt.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < nxt.length - 1 } })),
    { x: 5.95, y: 1.9, w: 3.4, h: 3.1, fontSize: 12, color: C.text1, paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true });
  s.addNotes(
    "This is our ablation study. With the detector alone, the robot crashes into fallen boxes because they are not a trained class. " +
    "Adding the depth branch finds them, and collisions drop to zero. On shiny stretch-wrapped pallets the uncertainty score rises, so " +
    "the robot slows down far more often than near normal pallets. In scripted tests where a worker suddenly steps out about one metre ahead, " +
    "the shield stopped the robot every time. Next week we fine-tune on LOCO and connect the real detector to the robot.");

  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
