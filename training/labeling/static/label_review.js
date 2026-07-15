"use strict";
// Model-assisted REVIEW UI — read-only judging. Shows the YOLO prediction as a
// thin, crisp contour with a faint transparent fill so it's easy to judge at a
// glance. No editing: the only decisions are Accept / Reject / Skip. One mode
// per session (from ?mode=; default spill). Points are IMAGE (natural) pixels;
// the canvas backing store is the natural image size and CSS scales it to fit.
//
//   Accept — the mask is right. Approves the predicted shapes into training; if
//            the model found NOTHING, confirms the image as clean (negative).
//   Reject — the mask is wrong -> image goes to the hand labeler (app_multi.py).
//   Skip   — no decision, move on.

const MODE_META = {
  standard: { name: "REVIEW · STANDARD (in-cup)", dataset: "→ smoothie_dataset_std", color: "#39c07a", icon: "🟢" },
  spill:    { name: "REVIEW · SPILL",             dataset: "→ spill_dataset",         color: "#e8833a", icon: "🟠" },
  logo:     { name: "REVIEW · LOGO",              dataset: "→ logo_dataset",          color: "#a06cd5", icon: "🟣" },
};
const params = new URLSearchParams(location.search);
const mode = MODE_META[params.get("mode")] ? params.get("mode") : "spill";
// File order by default so traversal + resume are predictable; ?sort=conf_asc
// for a lowest-confidence-first triage session.
const sort = params.get("sort") || "file";

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const doneEl = document.getElementById("done");
const frameEl = document.getElementById("frame");
const fileidEl = document.getElementById("fileid");
const confEl = document.getElementById("conf");
const badgeEl = document.getElementById("reviewbadge");
const emptyEl = document.getElementById("emptyflag");

let fileId = null;
let shapes = [];           // [[[x,y],...], ...]  model prediction (display only)
let confidences = [];
let reviewStatus = null;
let hasShapes = false;     // model found at least one polygon
let img = new Image();

// ---- rendering: thin crisp contour + faint transparent fill -----------------
function cssScale() {
  const rect = canvas.getBoundingClientRect();
  return canvas.width ? rect.width / canvas.width : 1;
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const color = MODE_META[mode].color;
  const s = cssScale();
  for (const p of shapes) {
    if (p.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(p[0][0], p[0][1]);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
    if (p.length >= 3) ctx.closePath();
    // faint transparent fill so the covered area reads at a glance
    if (p.length >= 3) { ctx.fillStyle = hexA(color, 0.14); ctx.fill(); }
    // dark hairline underneath keeps the contour crisp on light backgrounds
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 2.2 / s;
    ctx.stroke();
    // thin bright contour on top
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.0 / s;
    ctx.stroke();
  }

  if (!hasShapes) {
    ctx.save();
    ctx.fillStyle = "#ffcc00";
    ctx.font = `${Math.round(22 / s)}px system-ui, sans-serif`;
    ctx.fillText("model found nothing — Accept = clean, Reject = it missed something",
                 12 / s, 40 / s);
    ctx.restore();
  }
}

// ---- keyboard: Accept / Reject / Skip + navigation --------------------------
document.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "arrowleft")  { e.preventDefault(); goPrev(); return; }
  if (k === "arrowright") { e.preventDefault(); goNext(); return; }
  if (!fileId) return;
  switch (k) {
    case "a": case "enter": accept(); break;
    case "r": reject(); break;
    case "s": skip(); break;
  }
});

// ---- decisions --------------------------------------------------------------
function accept() {
  // Model is right: approve its shapes, or (nothing found) confirm clean.
  if (hasShapes) {
    const v = shapes.filter((p) => p.length >= 3).map((p) => ({ polygon: p }));
    decide("approve", v);
  } else {
    decide("clean", []);
  }
}
function reject() { decide("reject", []); }
function skip() { nextPending(fileId); }   // leave pending, show next pending

function decide(decision, v) {
  const wasPending = !reviewStatus || reviewStatus === "pending";
  const at = fileId;
  fetch("/api/review/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: at, mode, decision, shapes: v }),
  }).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(() => {
      refreshProgress();
      // Fresh review of a pending image -> flow on to the next pending.
      // Re-deciding an already-decided image (traversing back) -> just step one,
      // so you stay where you were inspecting instead of being yanked forward.
      if (wasPending) nextPending(at); else seek("next");
    })
    .catch((err) => flashWarn("Save failed: " + err.message));
}

// ---- navigation -------------------------------------------------------------
// Two tracks:
//   nextPending() — jump to the first/next PENDING image (resume + decision flow)
//   seek(dir)     — step one image in file order through ALL images (any status),
//                   so you can go back and change a past decision
function nextPending(after) {
  const a = after != null ? `&after=${after}` : "";
  fetch(`/api/review/next?mode=${mode}&sort=${sort}${a}`)
    .then((r) => r.json()).then((d) => {
      if (d.done) { showDone(); return; }
      setItem(d);
    });
}
function goNext() { fileId == null ? nextPending(null) : seek("next"); }
function goPrev() { if (fileId != null) seek("prev"); }
function seek(dir) {
  if (fileId == null) { nextPending(null); return; }
  fetch(`/api/review/seek?mode=${mode}&file_id=${fileId}&dir=${dir}`)
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => {
      if (d.edge) { flashWarn(dir === "prev" ? "At the first image." : "At the last image."); return; }
      setItem(d);
    })
    .catch((err) => flashWarn("Navigation failed (" + err.message + ")."));
}

function setItem(d) {
  fileId = d.file_id;
  reviewStatus = d.review_status || null;
  shapes = (d.shapes && d.shapes.length) ? d.shapes.map((p) => p.slice()) : [];
  confidences = d.confidences || [];
  hasShapes = shapes.some((p) => p.length >= 3);
  doneEl.style.display = "none"; frameEl.style.display = "";
  fileidEl.textContent = `#${d.file_id}`;
  emptyEl.style.display = hasShapes ? "none" : "inline-block";
  updateBadge(); updateConf(); updateButtons();

  img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    draw();
  };
  img.src = `/image/${d.file_id}?t=${d.file_id}`;
}

function showDone() {
  fileId = null; frameEl.style.display = "none"; doneEl.style.display = "block";
  fileidEl.textContent = ""; confEl.textContent = ""; badgeEl.textContent = "";
  emptyEl.style.display = "none"; updateButtons();
}

// ---- sidebar / progress -----------------------------------------------------
function flashWarn(msg) {
  const w = document.getElementById("warn");
  w.textContent = msg; w.style.display = "block";
  clearTimeout(flashWarn._t);
  flashWarn._t = setTimeout(() => { w.style.display = "none"; }, 3500);
}
function updateBadge() {
  const s = reviewStatus || "pending";
  badgeEl.textContent = s.toUpperCase();
  badgeEl.className = s;
}
function updateConf() {
  const cs = confidences.filter((c) => c != null);
  if (!hasShapes || !cs.length) { confEl.textContent = "conf: — (no detection)"; return; }
  confEl.textContent = `top conf ${Math.max(...cs).toFixed(2)} · ${cs.length} pred`;
}
function updateButtons() {
  ["btnApprove", "btnReject", "btnSkip"].forEach((id) =>
    (document.getElementById(id).disabled = !fileId));
}
function refreshProgress() {
  fetch(`/api/review/progress?mode=${mode}`).then((r) => r.json()).then((p) => {
    const c = p.counts;
    document.getElementById("counts").innerHTML =
      `<div><b>${mode}</b> — ${c.total} reviewed</div>` +
      `<div>pending: <b>${c.pending}</b></div>` +
      `<div>approved: <b>${c.approved}</b></div>` +
      `<div>rejected: <b>${c.rejected}</b></div>`;
  });
}

// ---- boot -------------------------------------------------------------------
document.getElementById("btnApprove").onclick = accept;
document.getElementById("btnReject").onclick = reject;
document.getElementById("btnSkip").onclick = skip;

document.documentElement.dataset.mode = mode;
document.getElementById("modename").textContent = MODE_META[mode].name;
document.getElementById("dataset").textContent = MODE_META[mode].dataset;
document.title = MODE_META[mode].icon + " review · " + mode;
refreshProgress();
nextPending(null);   // resume at the first pending image
