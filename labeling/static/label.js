"use strict";
// Review mode  — image + SAM mask shown as a semi-transparent green fill.
// Edit mode (E) — image + editable polygon with draggable vertex handles.
// Polygon points are kept in IMAGE (natural) pixel coordinates; the canvas
// backing store is the natural image size and CSS scales the element to fit.

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const doneEl = document.getElementById("done");
const modeEl = document.getElementById("mode");
const fileidEl = document.getElementById("fileid");

let item = null;        // {file_id, width, height, polygon:[[x,y]...], verdict}
let img = new Image();
let maskOverlay = null; // off-screen canvas: RGBA tinted mask for review mode
let editMode = false;
let cutMode = false;    // sub-mode of edit: click a horizontal line to slice the top flat
let cutY = null;        // current cut-line y (image px) tracked from the mouse
let dirty = false;      // polygon was edited -> verdict becomes "corrected"
let dragIndex = -1;
let undoStack = [];
const visited = [];
let histIndex = -1;

// ---- coordinate helpers -----------------------------------------------------
function cssScale() {
  const rect = canvas.getBoundingClientRect();
  return rect.width / canvas.width;
}
function toImageCoords(evt) {
  const rect = canvas.getBoundingClientRect();
  const s = rect.width / canvas.width;
  return [(evt.clientX - rect.left) / s, (evt.clientY - rect.top) / s];
}
function handleRadius() { return 6 / cssScale(); }
function hitTol()       { return 9 / cssScale(); }

// ---- mask overlay -----------------------------------------------------------
// Converts a grayscale SAM mask PNG into an RGBA canvas where white pixels
// become semi-transparent green and black pixels become fully transparent.
function buildMaskOverlay(maskImg) {
  const off = document.createElement("canvas");
  off.width  = maskImg.naturalWidth;
  off.height = maskImg.naturalHeight;
  const octx = off.getContext("2d");
  octx.drawImage(maskImg, 0, 0);
  const id = octx.getImageData(0, 0, off.width, off.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const isMask = d[i] > 128;
    d[i]   = 57;              // R
    d[i+1] = 192;             // G
    d[i+2] = 122;             // B
    d[i+3] = isMask ? 120 : 0; // A: ~47 % opacity inside mask
  }
  octx.putImageData(id, 0, 0);
  maskOverlay = off;
}

// ---- rendering --------------------------------------------------------------
function draw() {
  if (!item) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const p = item.polygon;

  if (!editMode) {
    // Review mode: show the SAM mask as a flat overlay (no polygon border/handles).
    if (maskOverlay) {
      ctx.drawImage(maskOverlay, 0, 0, canvas.width, canvas.height);
    } else if (p && p.length >= 3) {
      // Fallback: mask PNG not available, draw filled polygon without border.
      ctx.fillStyle = "rgba(57,192,122,0.35)";
      ctx.beginPath();
      ctx.moveTo(p[0][0], p[0][1]);
      for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
      ctx.closePath();
      ctx.fill();
    }
    return;
  }

  // Edit mode: draw polygon outline + draggable vertex handles.
  if (p && p.length) {
    ctx.lineWidth   = 2 / cssScale();
    ctx.strokeStyle = dirty ? "#4a90e0" : "#39c07a";
    ctx.fillStyle   = dirty ? "rgba(74,144,224,0.15)" : "rgba(57,192,122,0.15)";
    ctx.beginPath();
    ctx.moveTo(p[0][0], p[0][1]);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Vertex handles are hidden in cut mode (the polygon is about to be re-cut).
    if (!cutMode) {
      const r = handleRadius();
      for (let i = 0; i < p.length; i++) {
        ctx.beginPath();
        ctx.arc(p[i][0], p[i][1], r, 0, Math.PI * 2);
        ctx.fillStyle   = i === dragIndex ? "#ffd166" : "#4a90e0";
        ctx.fill();
        ctx.lineWidth   = 1 / cssScale();
        ctx.strokeStyle = "#fff";
        ctx.stroke();
      }
    }
  }

  // Empty polygon in edit mode: prompt the user to start drawing.
  if ((!p || !p.length) && !cutMode) {
    ctx.save();
    ctx.fillStyle = "#ffcc00";
    ctx.font = `${Math.round(22 / cssScale())}px system-ui, sans-serif`;
    ctx.fillText("click to place points  (Z = undo scrap)",
                 12 / cssScale(), 40 / cssScale());
    ctx.restore();
  }

  // Cut-line guide: a full-width horizontal line at the mouse Y. Everything
  // ABOVE it (the foam/rim) is removed on click.
  if (cutMode) {
    const y = cutY == null ? canvas.height * 0.15 : cutY;
    ctx.save();
    ctx.lineWidth = 2 / cssScale();
    ctx.setLineDash([10 / cssScale(), 6 / cssScale()]);
    ctx.strokeStyle = "#ffcc00";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
    // shade the region that will be cut away
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(224,82,74,0.20)";
    ctx.fillRect(0, 0, canvas.width, y);
    ctx.restore();
  }
}

// ---- edit operations --------------------------------------------------------
function snapshot() { undoStack.push(JSON.parse(JSON.stringify(item.polygon))); }
function markDirty() { dirty = true; updateMode(); }

// Slice everything ABOVE a horizontal line at image-y = Y, capping the polygon
// with a flat top edge at Y (Sutherland–Hodgman clip against the half-plane
// y >= Y; y grows downward, so "keep below the line"). Used to exclude the foam
// band in one click. Returns the clipped ring, or null if it would degenerate.
function clipTop(poly, Y) {
  const out = [];
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const cur = poly[i], nxt = poly[(i + 1) % n];
    const curIn = cur[1] >= Y, nxtIn = nxt[1] >= Y;
    if (curIn) out.push([cur[0], cur[1]]);
    if (curIn !== nxtIn) {
      const t = (Y - cur[1]) / (nxt[1] - cur[1]);
      out.push([Math.round(cur[0] + t * (nxt[0] - cur[0])), Math.round(Y)]);
    }
  }
  return out.length >= 3 ? out : null;
}

function applyCut(Y) {
  Y = Math.max(0, Math.min(item.height, Math.round(Y)));
  const clipped = clipTop(item.polygon, Y);
  if (!clipped) return;              // cut would erase the whole shape — ignore
  snapshot();
  item.polygon = clipped;
  markDirty();
  cutMode = false;                   // one cut, then back to vertex editing
  updateMode();
  draw();
}

function nearestVertex(pt) {
  const p = item.polygon, tol = hitTol();
  let best = -1, bestD = tol;
  for (let i = 0; i < p.length; i++) {
    const d = Math.hypot(p[i][0] - pt[0], p[i][1] - pt[1]);
    if (d <= bestD) { bestD = d; best = i; }
  }
  return best;
}
function segDist(pt, a, b) {
  const vx = b[0]-a[0], vy = b[1]-a[1];
  const wx = pt[0]-a[0], wy = pt[1]-a[1];
  const len2 = vx*vx + vy*vy || 1e-9;
  let t = (wx*vx + wy*vy) / len2;
  t = Math.max(0, Math.min(1, t));
  return { d: Math.hypot(pt[0]-(a[0]+t*vx), pt[1]-(a[1]+t*vy)) };
}
function nearestEdge(pt) {
  const p = item.polygon, tol = hitTol();
  let best = -1, bestD = tol;
  for (let i = 0; i < p.length; i++) {
    const { d } = segDist(pt, p[i], p[(i+1) % p.length]);
    if (d <= bestD) { bestD = d; best = i; }
  }
  return best;
}

// ---- pointer events ---------------------------------------------------------
canvas.addEventListener("contextmenu", (e) => {
  if (!editMode) return;
  e.preventDefault();
  const pt = toImageCoords(e);
  const vi = nearestVertex(pt);
  if (vi >= 0 && item.polygon.length > 3) {
    snapshot(); item.polygon.splice(vi, 1); markDirty(); draw();
  }
});
canvas.addEventListener("pointerdown", (e) => {
  if (!editMode || e.button !== 0) return;
  const pt = toImageCoords(e);
  if (cutMode) { applyCut(pt[1]); return; }   // click = slice the top flat here
  const vi = nearestVertex(pt);
  if (vi >= 0) {
    snapshot(); dragIndex = vi; canvas.setPointerCapture(e.pointerId); draw(); return;
  }
  if (item.polygon.length) {
    const ei = nearestEdge(pt);
    if (ei >= 0) {
      snapshot();
      item.polygon.splice(ei+1, 0, [Math.round(pt[0]), Math.round(pt[1])]);
      dragIndex = ei+1; markDirty();
      canvas.setPointerCapture(e.pointerId); draw(); return;
    }
  }
  snapshot();
  item.polygon.push([Math.round(pt[0]), Math.round(pt[1])]);
  dragIndex = item.polygon.length - 1; markDirty();
  canvas.setPointerCapture(e.pointerId); draw();
});
canvas.addEventListener("pointermove", (e) => {
  if (cutMode) { cutY = toImageCoords(e)[1]; draw(); return; }
  if (dragIndex < 0) return;
  const pt = toImageCoords(e);
  item.polygon[dragIndex] = [
    Math.max(0, Math.min(item.width,  Math.round(pt[0]))),
    Math.max(0, Math.min(item.height, Math.round(pt[1]))),
  ];
  markDirty(); draw();
});
canvas.addEventListener("pointerup", () => { dragIndex = -1; draw(); });

// ---- keyboard ---------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if (!item && !["ArrowLeft","ArrowRight"].includes(e.key)) return;
  switch (e.key.toLowerCase()) {
    case "a": saveVerdict("good"); break;
    case "r": saveVerdict("bad");  break;
    case "s": saveVerdict("skip"); break;
    case "e":
      editMode = !editMode;
      if (!editMode) cutMode = false;   // leaving edit also leaves cut
      updateMode(); draw(); break;
    case "c":
      // Toggle the cut-line tool. Auto-enters edit mode if we're reviewing.
      if (!editMode) editMode = true;
      cutMode = !cutMode;
      updateMode(); draw(); break;
    case "escape":
      if (cutMode) { cutMode = false; updateMode(); draw(); }
      break;
    case "x":
      // Scrap the polygon entirely — start from scratch. After this, click on
      // the image to drop new vertices one by one (Z undoes the scrap).
      if (!editMode) break;
      if (item.polygon.length) {
        snapshot();
        item.polygon = [];
        cutMode = false; dragIndex = -1;
        markDirty(); draw();
      }
      break;
    case "z":
      if (undoStack.length) {
        item.polygon = undoStack.pop();
        if (!undoStack.length) dirty = false;
        updateMode(); draw();
      }
      break;
    case "arrowleft":  e.preventDefault(); goPrev(); break;
    case "arrowright": e.preventDefault(); goNext(); break;
  }
});

// ---- verdict + navigation ---------------------------------------------------
function saveVerdict(verdict) {
  if (!item) return;
  if (dirty && verdict === "good") verdict = "corrected";
  fetch("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_id: item.file_id,
      verdict,
      polygon: dirty ? item.polygon : null,
    }),
  }).then(() => { refreshProgress(); goNext(); });
}

function goNext() {
  if (histIndex >= 0 && histIndex < visited.length - 1) {
    histIndex++;
    loadItem(visited[histIndex]);
    return;
  }
  const after = item ? item.file_id : null;
  fetch(after != null ? `/api/next?after=${after}` : "/api/next")
    .then(r => r.json())
    .then(d => {
      if (d.done) { showDone(); return; }
      setItem(d);
      if (visited[visited.length - 1] !== d.file_id) visited.push(d.file_id);
      histIndex = visited.length - 1;
    });
}
function goPrev() {
  if (histIndex > 0) { histIndex--; loadItem(visited[histIndex]); }
}
function loadItem(fileId) {
  fetch(`/api/item/${fileId}`).then(r => r.json()).then(setItem);
}

function setItem(d) {
  item = { file_id: d.file_id, width: d.width, height: d.height,
           polygon: d.polygon || [], verdict: d.verdict || null };
  editMode = false; cutMode = false; cutY = null;
  dirty = false; dragIndex = -1; undoStack = [];
  maskOverlay = null;
  doneEl.style.display = "none"; canvas.style.display = "";
  canvas.width  = d.width;
  canvas.height = d.height;
  fileidEl.textContent = `#${d.file_id}` + (d.verdict ? ` (was: ${d.verdict})` : "");
  updateMode();

  // Load photo and mask in parallel; draw when both are ready.
  let photoReady = false, maskReady = false;
  const tryDraw = () => { if (photoReady && maskReady) draw(); };

  img = new Image();
  img.onload = () => { photoReady = true; tryDraw(); };
  img.src = `/image/${d.file_id}?t=${d.file_id}`;

  const maskImg = new Image();
  maskImg.onload = () => {
    buildMaskOverlay(maskImg);
    maskReady = true; tryDraw();
  };
  maskImg.onerror = () => { maskReady = true; tryDraw(); }; // no mask on disk yet
  maskImg.src = `/mask/${d.file_id}`;
}

function showDone() {
  item = null; canvas.style.display = "none"; doneEl.style.display = "block";
  fileidEl.textContent = "";
}
function updateMode() {
  let label = editMode ? "edit" : "review";
  if (editMode && cutMode) label = "edit • CUT-LINE (click to slice top)";
  if (dirty) label += " • corrected";
  modeEl.textContent = label;
  modeEl.className   = editMode ? "edit" : "";
}

// ---- progress ---------------------------------------------------------------
function refreshProgress() {
  fetch("/api/progress").then(r => r.json()).then(p => {
    document.getElementById("progress").innerHTML =
      `<b>${p.labeled}</b> / ${p.total} labeled`;
    document.getElementById("counts").innerHTML =
      `<span class="pill good">good ${p.good}</span>
       <span class="pill corrected">corrected ${p.corrected}</span>
       <span class="pill bad">bad ${p.bad}</span>
       <span class="pill skip">skip ${p.skip}</span>`;
  });
}

// ---- boot -------------------------------------------------------------------
refreshProgress();
goNext();
