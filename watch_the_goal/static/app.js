"use strict";

// ---------------------------------------------------------------------------
// State

let data = { sources: [], clips: [], order: [], exports: [], jobs: [], settings: {} };
let lastJson = "";
const sel = { clipId: null, sourceId: null };
const activity = {}; // sourceId -> { key, audio, motion }
const uploads = []; // { name, progress, error }
let dragging = false;
let resetArmed = false;
let reelQueue = null; // { ids, index } while previewing the reel
let stopAt = null; // seconds; pause when the player reaches it
let loaded = { sourceId: null, preview: null };
let watchedExportJob = null;

const $ = (id) => document.getElementById(id);
const player = $("player");

// ---------------------------------------------------------------------------
// Helpers

function fmt(t, precise = true) {
  t = Math.max(0, t || 0);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const ss = precise ? s.toFixed(1).padStart(4, "0") : String(Math.floor(s)).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
function fmtDur(t) {
  return t >= 60 ? `${Math.floor(t / 60)}m ${String(Math.round(t % 60)).padStart(2, "0")}s` : `${t.toFixed(1)}s`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
const srcById = (id) => data.sources.find((s) => s.id === id);
const clipById = (id) => data.clips.find((c) => c.id === id);
const clipsFor = (sid) => data.clips.filter((c) => c.source_id === sid).sort((a, b) => a.start - b.start);
const reel = () => data.clips.filter((c) => c.status === "kept");
const clipLen = (c) => c.end - c.start;
const jobFor = (sid, kind) => data.jobs.find((j) => j.target === sid && j.kind === kind && j.status === "running");
const LABEL_TEXT = { likely_goal: "likely goal", save: "save", near_miss: "near miss", other: "not a highlight", uncertain: "uncertain" };
function bestFirst() {
  try { return localStorage.getItem("wtg-best-first") === "1"; } catch { return false; }
}
function reviewOrder() {
  // Timeline order: sources as imported, clips by time. "Strongest first"
  // visits each recording's suggestions by rank (after AI review, likely goals first).
  return data.sources.flatMap((s) => {
    const clips = clipsFor(s.id);
    if (!bestFirst()) return clips;
    const rank = (c) => (c.origin === "detected" && c.rank != null ? c.rank : 1e6 + c.start);
    return [...clips].sort((a, b) => rank(a) - rank(b));
  });
}
function frameUrl(sid, t, w = 320) {
  return `/api/sources/${sid}/frame?t=${t.toFixed(2)}&w=${w}`;
}

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let payload = {};
  try { payload = await res.json(); } catch { /* empty */ }
  if (!res.ok) throw new Error(payload.error || `${res.status} ${res.statusText}`);
  return payload;
}

let toastTimer;
function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast" + (isError ? " err" : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), isError ? 7000 : 3500);
}

// ---------------------------------------------------------------------------
// Sync with the server

let stateRequest = 0;
async function refresh(force = false) {
  const request = ++stateRequest;
  let state;
  try {
    state = await api("GET", "/api/state");
  } catch {
    return;
  }
  if (dragging || request !== stateRequest) return;
  const json = JSON.stringify(state);
  if (!force && json === lastJson) return;
  lastJson = json;
  data = state;
  if (sel.clipId && !clipById(sel.clipId)) sel.clipId = null;
  if (sel.sourceId && !srcById(sel.sourceId)) sel.sourceId = null;
  if (!sel.sourceId && data.sources.length) selectSource(data.sources[0].id, 0, false);
  watchExport();
  render();
}

function schedulePoll() {
  const busy = data.jobs.some((j) => j.status === "running") || uploads.length;
  setTimeout(async () => {
    if (!dragging) await refresh();
    schedulePoll();
  }, busy ? 700 : 2500);
}

// ---------------------------------------------------------------------------
// Rendering

function render() {
  shapeForPreview = outputLayout();
  renderSources();
  renderTimeline();
  renderEditor();
  renderReel();
  renderExports();
  const has = data.sources.length > 0;
  $("player-empty").hidden = has;
  $("new-clip-btn").disabled = !sel.sourceId;
  $("missed-btn").disabled = !loaded.sourceId;
  $("drop-hint").hidden = data.sources.length > 2;
  $("undo-btn").disabled = !data.can_undo;
  $("template-select").value = data.settings.template || "plain";
  $("details-btn").hidden = data.settings.template !== "recap";
  $("layout-select").value = data.settings.layout || "auto";
}

function renderSources() {
  const html = data.sources.map((s) => {
    const info = s.info;
    const badges = [];
    badges.push(s.kind === "long" ? `<span class="badge long">Long recording</span>` : `<span class="badge">Short clip</span>`);
    if (info.orientation === "portrait") badges.push(`<span class="badge">Portrait</span>`);
    if (info.rotation) badges.push(`<span class="badge">Rotated ${info.rotation}°</span>`);
    if (info.fps >= 48) badges.push(`<span class="badge">${Math.round(info.fps)} fps</span>`);
    badges.push(`<span class="badge">${info.width}×${info.height}</span>`);
    if (!info.has_audio) badges.push(`<span class="badge warn">No audio</span>`);
    if (s.preview === "pending") badges.push(`<span class="badge warn">${esc(info.video_codec.toUpperCase())} · preparing preview</span>`);
    if (s.goal_areas.length) badges.push(`<span class="badge">${s.goal_areas.length} goal${s.goal_areas.length > 1 ? "s" : ""} marked</span>`);

    const job = jobFor(s.id, "analyze");
    let status = "";
    const a = s.analysis || {};
    if (job) {
      status = `<div class="status">${esc(job.message)} · ${Math.round(job.progress * 100)}%</div>
        <div class="progress"><i style="width:${job.progress * 100}%"></i></div>`;
    } else if (a.status === "done") {
      const clips = clipsFor(s.id);
      if (s.kind === "long") {
        const cand = clips.filter((c) => c.origin === "detected").length;
        status = `<div class="status">${cand ? `${cand} suggested moment${cand > 1 ? "s" : ""}` : "No clear moments found. Scrub and add clips by hand."} · used ${esc(a.used.join(", "))}</div>`;
        for (const note of a.notes || []) status += `<div class="status err">${esc(note)}</div>`;
      } else {
        const trim = clips.find((c) => c.suggested_trim);
        status = `<div class="status">Included whole${trim ? " · trim suggested" : ""}</div>`;
      }
    } else if (a.status === "error" || a.status === "interrupted") {
      status = `<div class="status err">${esc(a.message || "Analysis failed")}</div>`;
    } else {
      status = `<div class="status">Waiting to analyse…</div>`;
    }
    const rjob = jobFor(s.id, "review");
    const rv = s.review || {};
    if (rjob) {
      status += `<div class="status">${esc(rjob.message)} · ${Math.round(rjob.progress * 100)}%</div>
        <div class="progress"><i style="width:${rjob.progress * 100}%"></i></div>`;
    } else if (rv.status === "done" || rv.status === "partial") {
      const labels = Object.entries(rv.labels || {}).map(([k, n]) => `${n} ${LABEL_TEXT[k] || k}`).join(", ");
      status += `<div class="status">AI review (${esc(rv.model)}): ${rv.reviewed} checked${labels ? ` · ${esc(labels)}` : ""}${rv.failed ? ` · ${rv.failed} not reviewed` : ""}</div>`;
      if (rv.message) status += `<div class="status">${esc(rv.message)}</div>`;
    } else if (rv.status === "unavailable") {
      status += `<div class="status err">AI review unavailable, suggestions unchanged: ${esc(rv.message)}</div>`;
    }
    const reviewOn = data.settings.ai_review?.enabled && s.kind === "long";
    const thumbT = Math.min(info.duration * 0.3, 8);
    return `<div class="source ${s.id === sel.sourceId ? "active" : ""}" data-src="${s.id}">
      <div class="thumb" style="background-image:url('${frameUrl(s.id, thumbT, 480)}')"><span class="dur">${fmt(info.duration, false)}</span></div>
      <div class="meta">
        <div class="name" title="${esc(s.path)}">${esc(s.name)}</div>
        <div class="badges">${badges.join("")}</div>
        ${status}
        ${s.kind === "long" ? playSummary(s) : ""}
        ${coachHtml(s, !!(job || rjob))}
        <div class="actions">
          ${s.kind === "long" ? `<button class="btn small" data-act="trim" data-help="Set kick-off, halftime and full time so warm-ups and breaks are left out.">Trim match</button>` : ""}
          ${s.kind === "long" ? `<button class="btn small" data-act="goals" data-help="Draw boxes around the goal mouths in fixed-camera footage.">Mark goals</button>` : ""}
          <button class="btn small" data-act="analyze" data-help="Scan this recording again using the current analysis settings." ${job ? "disabled" : ""}>Re-analyse</button>
          ${reviewOn ? `<button class="btn small" data-act="review" data-help="Ask the local vision model about the strongest untouched suggestions." ${job || rjob ? "disabled" : ""}>AI review</button>` : ""}
          <button class="btn small ghost danger" data-act="remove" data-help="Remove this recording from the project. The original file stays on disk.">Remove</button>
        </div>
      </div>
    </div>`;
  });
  const up = uploads.map((u) => `<div class="source"><div class="meta">
      <div class="name">${esc(u.name)}</div>
      <div class="status ${u.error ? "err" : ""}">${u.error ? esc(u.error) : `Copying into workspace… ${Math.round(u.progress * 100)}%`}</div>
      <div class="progress"><i style="width:${u.progress * 100}%"></i></div></div></div>`);
  $("sources").innerHTML = html.join("") + up.join("");
}

function renderTimeline() {
  if (dragging) return;
  const el = $("timeline");
  if (!data.sources.length) {
    el.innerHTML = `<div class="timeline-empty">Clips and suggestions appear here once footage is imported.</div>`;
    return;
  }
  const focusedClip = document.activeElement.closest(".block")?.dataset.clip;
  el.innerHTML = data.sources.map((s) => {
    const d = s.info.duration;
    const blocks = clipsFor(s.id).map((c) => {
      const left = pct(s, c.start);
      const width = spanPct(s, clipLen(c));
      const tip = `${fmt(c.start)}–${fmt(c.end)} · ${c.reasons.join("; ")}`;
      return `<button aria-label="${esc(s.name)} · ${fmt(c.start)} to ${fmt(c.end)}" class="block ${c.status} ${c.id === sel.clipId ? "sel" : ""}" data-clip="${c.id}" title="${esc(tip)}" style="left:${left}%;width:${width}%"></button>`;
    });
    return `<div class="track ${s.id === sel.sourceId ? "active" : ""}" data-src="${s.id}">
      <div class="track-label" title="${esc(s.name)}">${esc(s.name)}<br><span class="d">${fmt(d, false)}</span></div>
      <div class="track-lane">
        <div class="track-ruler" aria-hidden="true">${rulerHtml(s)}</div>
        <div class="track-bar" data-src="${s.id}"><canvas></canvas>${s.kind === "long" ? offPlay(s) : ""}${blocks.join("")}${callFlags(s)}<div class="ph" hidden></div></div>
        ${overviewHtml(s)}
      </div>
    </div>`;
  }).join("");
  if (focusedClip) el.querySelector(`[data-clip="${focusedClip}"]`)?.focus({preventScroll: true});
  for (const s of data.sources) {
    const canvas = el.querySelector(`.track-bar[data-src="${s.id}"] canvas`);
    drawActivity(canvas, s, ...viewOf(s));
    drawEvents(canvas, s);
    loadActivity(s);
    loadEvidence(s);
  }
  renderNav();
  updatePlayheads();
}

function loadActivity(s) {
  const key = `${s.analysis?.status}:${s.analysis?.seconds}`;
  if (s.analysis?.status !== "done" || activity[s.id]?.key === key) return;
  activity[s.id] = { key };
  api("GET", `/api/sources/${s.id}/activity`).then((a) => {
    activity[s.id] = { key, ...a };
    renderTimeline();
    renderEditor();
  }).catch(() => {});
}

function drawActivity(canvas, s, t0, t1) {
  if (!canvas) return;
  const a = activity[s.id];
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!a || (!a.audio && !a.motion)) return;
  const draw = (series, color, fill) => {
    if (!series) return;
    ctx.beginPath();
    const n = series.length;
    for (let x = 0; x <= w; x++) {
      const t = t0 + (x / w) * (t1 - t0);
      const i0 = Math.floor(t), i1 = Math.max(i0 + 1, Math.floor(t + (t1 - t0) / w));
      let v = 0;
      for (let i = i0; i < Math.min(i1, n); i++) v = Math.max(v, series[i] ?? 0);
      const y = h - 2 - v * (h - 6);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    if (fill) {
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      ctx.fillStyle = color; ctx.fill();
    } else {
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke();
    }
  };
  draw(a.audio, "rgba(120, 170, 255, 0.28)", true);
  draw(a.motion, "rgba(200, 220, 210, 0.45)", false);
}

// Zoomed window around the selected clip, fixed while dragging.
let zoomWin = null;
function zoomWindow(c, d) {
  const len = clipLen(c);
  const before = Math.max(15, len * 0.6), after = Math.max(8, len * 0.35);
  return [Math.max(0, Math.min(c.start, eventTime(c) ?? c.start) - before),
          Math.min(d, Math.max(c.end, eventTime(c) ?? c.end) + after)];
}

const waves = new Map();
function drawWave(s, start, end) {
  const canvas = $("zoom-wave");
  if (!canvas) return;
  const key = [s.id, start.toFixed(3), end.toFixed(3)].join("/");
  if (!waves.has(key)) {
    if (waves.size >= 32) waves.delete(waves.keys().next().value);
    waves.set(key, null);
    api("GET", `/api/sources/${s.id}/waveform?start=${start.toFixed(3)}&end=${end.toFixed(3)}`).then(({ peaks }) => {
      waves.set(key, peaks);
      if ($("zoom-wave")?.dataset.key === key) drawWave(s, start, end);
    }).catch(() => waves.delete(key));
  }
  canvas.dataset.key = key;
  const peaks = waves.get(key);
  if (!peaks) return;
  canvas.width = peaks.length; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#84bca8";
  peaks.forEach((v, x) => {
    const height = Math.max(1, 52 * Math.sqrt(v));
    ctx.fillRect(x, (64 - height) / 2, 1, height);
  });
}
const eventTime = c => c.anchor ?? c.suggested_trim?.anchor;
let proposal = null;
function editControls(c) {
  if (!data.settings.ai_review?.enabled) return "";
  const pending = proposal?.id === c.id && proposal.expected[0] === c.start && proposal.expected[1] === c.end;
  if (pending) {
    const v = proposal.result;
    return v ? `<div class="suggest" role="status">Current ${fmt(c.start)}–${fmt(c.end)} → Proposed ${fmt(v.start)}–${fmt(v.end)} · ${esc(v.reason)}
      <button class="btn small" data-act="accept">Apply edit</button><button class="btn small" data-act="dismiss">Dismiss</button></div>`
      : '<div class="suggest" role="status">Local reviewer is checking the footage… <button class="btn small" data-act="dismiss">Dismiss</button></div>';
  }
  return '<div class="suggest">Ask reviewer <button class="btn small" data-act="buildup">Include more of the buildup</button><button class="btn small" data-act="sooner">Cut sooner after the goal</button></div>';
}
async function requestEdit(c, request) {
  const current = { id: c.id, expected: [c.start, c.end] };
  proposal = current;
  renderEditor();
  try {
    current.result = await api("POST", `/api/clips/${c.id}/proposal`, { request });
  } catch (err) {
    if (proposal === current) { proposal = null; toast(err.message, true); }
  }
  renderEditor();
}
function renderEditor() {
  if (dragging) return;
  const ed = $("editor");
  const c = clipById(sel.clipId);
  if (!c) { ed.hidden = true; ed.innerHTML = ""; return; }
  const s = srcById(c.source_id);
  const d = s.info.duration;
  zoomWin = zoomWindow(c, d);
  const origin = { detected: "Suggested moment", "short-clip": "Phone clip", manual: "Your clip" }[c.origin];
  let reasons = c.reasons.map((r) => `<span class="reason ${c.origin === "detected" ? "" : "plain"}">${esc(r)}</span>`).join("");
  const rv = c.review;
  if (rv?.status === "done") {
    reasons += `<span class="reason ai" title="${esc(rv.model)} looked at ${fmt(rv.window[0])}–${fmt(rv.window[1])}">AI: ${esc(LABEL_TEXT[rv.label] || rv.label)} · ${esc(rv.reason)}</span>`;
  } else if (rv && rv.status !== "done") {
    reasons += `<span class="reason ai fail" title="${esc(rv.message || "")}">AI review ${rv.status === "timeout" ? "timed out" : "gave no usable answer"}; detector suggestion kept</span>`;
  }
  const rank = c.origin === "detected" && c.rank != null ? ` #${c.rank}` : "";
  const trim = c.suggested_trim;
  const trimHtml = trim && (Math.abs(trim.start - c.start) > 0.2 || Math.abs(trim.end - c.end) > 0.2)
    ? `<div class="suggest">Suggested trim ${fmt(trim.start)}–${fmt(trim.end)} (${esc(trim.reasons.join("; "))})
        <button class="btn small" data-act="apply-trim">Apply</button></div>`
    : "";
  const active = ed.contains(document.activeElement) ? document.activeElement : null;
  const focusSelector = active?.id ? `#${active.id}` : active?.dataset.act ? `[data-act="${active.dataset.act}"]` : null;
  const moment = eventTime(c);
  ed.hidden = false;
  ed.innerHTML = `
    <div class="editor-head">
      <span class="editor-title" title="${rank ? "Rank among this recording's suggestions, strongest first" : ""}">${origin}${rank} · ${esc(s.name)}</span>
      <span class="spacer"></span>
      <button class="btn small" data-act="play-clip" data-help="Play just the selected clip from its In point to its Out point.">↻ Replay range</button>
      <button class="btn small keep ${c.status === "kept" ? "on" : ""}" data-act="keep" data-help="Add this clip to the reel (K).">✓ Keep</button>
      <button class="btn small skip ${c.status === "skipped" ? "on" : ""}" data-act="skip" data-help="Leave this clip out of the reel (X).">Skip</button>
      ${c.origin !== "short-clip" ? `<button class="btn small ghost danger" data-act="delete" title="Delete this clip">Delete</button>` : ""}
    </div>
    <div class="reasons">${reasons}</div>
    ${callsRow(c)}
    <div class="range-meta"><span>In <b id="start-label"></b></span>
      ${moment != null ? `<button class="event-jump" data-act="finish" data-help="Jump to the suggested finish without changing the trim.">◆ Finish ${fmt(moment)}</button>` : ""}
      <span>Out <b id="end-label"></b></span></div>
    <div class="zoom" id="zoom">${s.info.has_audio ? '<canvas id="zoom-wave" aria-label="Audio waveform"></canvas>' : '<canvas></canvas>'}
      <div class="sel ${c.status}" id="zoom-sel"></div>
      ${moment != null ? `<div class="moment" id="zoom-moment" title="Detected moment"></div>` : ""}
      <div class="handle" id="h-start" role="slider" tabindex="0" aria-label="Clip start" title="Drag to move the start"></div>
      <div class="handle" id="h-end" role="slider" tabindex="0" aria-label="Clip end" title="Drag to move the end"></div>
      <div class="ph" id="zoom-ph"></div>
      <div id="zoom-ticks"></div>
    </div>
    <div class="trim-row">
      <div class="trim-side">
        <button class="btn small" data-act="start-5" data-help="Include five more seconds of buildup (Shift+←).">⟵ 5s earlier</button>
        <button class="btn small" data-act="start-1" data-help="Move the In point one second earlier.">−1s</button>
        <button class="btn small" data-act="start+1" data-help="Move the In point one second later.">+1s</button>
        <button class="btn small" data-act="start-here" data-help="Set the In point to the playhead ([).">Start here</button>

      </div>
      <div class="trim-len" id="len-label">${clipLen(c).toFixed(1)}s</div>
      <div class="trim-side end">

        <button class="btn small" data-act="end-here" data-help="Set the Out point to the playhead (]).">End here</button>
        <button class="btn small" data-act="end-1" data-help="Move the Out point one second earlier.">−1s</button>
        <button class="btn small" data-act="end+1" data-help="Move the Out point one second later.">+1s</button>
        <button class="btn small" data-act="end+5" data-help="Move the Out point five seconds later (Shift+→).">5s later ⟶</button>
      </div>
    </div>
    ${trimHtml}${editControls(c)}
    ${outputLayout() === "portrait" ? `<div class="framing"><label>Vertical framing
      <select id="frame-mode"><option value="fit">Fit full shot</option><option value="crop" ${c.crop != null ? "selected" : ""}>Crop to fill</option></select></label>
      ${c.crop != null ? `<label>Position <input id="crop-position" type="range" min="0" max="100" value="${Math.round(c.crop * 100)}"></label>` : ""}</div>` : ""}`;
  if (focusSelector) ed.querySelector(focusSelector)?.focus({preventScroll: true});
  layoutZoom(c, s);
}

function layoutZoom(c, s) {
  const [w0, w1] = zoomWin;
  const pct = (t) => ((t - w0) / (w1 - w0)) * 100;
  $("zoom-sel").style.left = pct(c.start) + "%";
  $("zoom-sel").style.width = pct(c.end) - pct(c.start) + "%";
  $("h-start").style.left = pct(c.start) + "%";
  $("h-end").style.left = pct(c.end) + "%";
  for (const edge of ["start", "end"]) {
    const handle = $("h-" + edge);
    handle.setAttribute("aria-valuemin", edge === "start" ? 0 : Math.min(s.info.duration, c.start + 1));
    handle.setAttribute("aria-valuemax", edge === "start" ? Math.max(0, c.end - 1) : s.info.duration);
    handle.setAttribute("aria-valuenow", c[edge]);
    handle.setAttribute("aria-valuetext", fmt(c[edge]));
  }
  const m = $("zoom-moment");
  if (m) m.style.left = pct(eventTime(c)) + "%";
  $("start-label").textContent = fmt(c.start);
  $("end-label").textContent = fmt(c.end);
  $("len-label").textContent = clipLen(c).toFixed(1) + "s";
  const span = w1 - w0;
  const step = span > 120 ? 30 : span > 60 ? 10 : 5;
  let ticks = "";
  for (let t = Math.ceil(w0 / step) * step; t <= w1; t += step) ticks += `<span class="tick" style="left:${pct(t)}%">${fmt(t, false)}</span>`;
  $("zoom-ticks").innerHTML = ticks;
  if (s.info.has_audio) drawWave(s, w0, w1);
  else drawActivity(document.querySelector("#zoom canvas"), s, w0, w1);
  updatePlayheads();
}

function renderReel() {
  if (dragging) return;
  const items = reel();
  const recap = items.length && data.settings.template === "recap";
  const total = items.reduce((a, c) => a + clipLen(c), (recap && data.settings.title ? 2 : 0) + (recap && data.settings.end_card ? 3 : 0));
  const scale = Math.max(90, total * 1.1);
  let note;
  if (!items.length) note = "Keep clips to build the reel. Aim for 30–60 seconds.";
  else if (total < 30) note = `${(30 - total).toFixed(0)}s under the 30–60s sweet spot.`;
  else if (total <= 60) note = "In the 30–60s sweet spot for a group chat.";
  else note = `${(total - 60).toFixed(0)}s over 60s. Skip or trim a clip, or export anyway.`;
  $("meter").innerHTML = `
    <div class="total">${fmtDur(total)}</div>
    <div class="bar"><div class="band" style="left:${(30 / scale) * 100}%;width:${(30 / scale) * 100}%"></div>
      <div class="fill ${total > 60 ? "over" : ""}" style="width:${Math.min(100, (total / scale) * 100)}%"></div></div>
    <div class="note">${note}</div>
    ${items.length ? `<div class="mix">${reelMix(items)}</div>` : ""}`;
  const moveFocus = document.activeElement.closest("[data-move]");
  const focusId = moveFocus?.closest(".reel-item")?.dataset.clip;
  const playingId = reelQueue ? reelQueue.ids[reelQueue.index] : null;
  $("reel-list").innerHTML = items.length
    ? items.map((c, i) => {
      const s = srcById(c.source_id);
      return `<li class="reel-item ${c.id === sel.clipId ? "sel" : ""} ${c.id === playingId ? "playing" : ""}" draggable="true" data-clip="${c.id}">
        <span class="n" title="Drag to reorder">${i + 1}</span>
        <img src="${frameUrl(s.id, Math.min(c.anchor ?? c.start + clipLen(c) / 2, s.info.duration - 0.1), 160)}" alt="" loading="lazy">
        <div class="t"><div>${kindOf(c) ? `<span class="kind" data-kind="${kindOf(c)}">${kindOf(c)}</span> ` : ""}${esc(s.name)}</div><div class="s">${fmt(c.start, false)}–${fmt(c.end, false)} · ${clipLen(c).toFixed(1)}s</div>${recap && c.caption ? `<div class="cap">“${esc(c.caption)}”</div>` : ""}</div>
        <div class="mv"><button data-move="-1" aria-label="Move up" data-help="Move this clip earlier in the reel.">▲</button><button data-move="1" aria-label="Move down" data-help="Move this clip later in the reel.">▼</button></div>
      </li>`;
    }).join("")
    : `<li class="reel-empty">Nothing kept yet. Keep clips one by one, or auto-fill from the suggestions above.</li>`;
  if (focusId) $("reel-list").querySelector(`[data-clip="${focusId}"] [data-move="${moveFocus.dataset.move}"]`)?.focus({preventScroll: true});
  const exporting = data.jobs.some((j) => j.kind === "export" && j.status === "running");
  $("export-btn").disabled = !items.length || exporting;
  $("preview-reel-btn").disabled = !items.length;
  const suggested = data.clips.filter((c) => c.origin === "detected" && c.status === "candidate").length;
  $("auto-btn").disabled = !suggested;
  $("auto-btn").dataset.help = suggested
    ? `Keep the strongest of ${suggested} suggestion${suggested > 1 ? "s" : ""} until the reel is about this long. One Undo puts them back.`
    : "No suggestions left to add. Import a long recording, or un-skip one.";
  $("preview-reel-btn").textContent = reelQueue ? "Stop preview" : "Preview reel";
}

function renderExports() {
  const job = data.jobs.find((j) => j.kind === "export" && (j.status === "running" || j.id === watchedExportJob));
  const st = $("export-status");
  if (job && job.status === "running") {
    st.innerHTML = `<div>${esc(job.message)} · ${Math.round(job.progress * 100)}%</div><div class="progress"><i style="width:${job.progress * 100}%"></i></div>`;
  } else if (job && job.status === "error") {
    st.innerHTML = `<div class="err">${esc(job.error)}</div>`;
  } else {
    st.innerHTML = "";
  }
  const html = data.exports.slice(0, 3).map((e, i) => `
    <div class="export-card">
      ${i === 0 ? `<video src="/exports/${encodeURIComponent(e.name)}" controls preload="metadata" playsinline></video>` : ""}
      <div class="info"><div><strong>${esc(e.name)}</strong><br>${fmtDur(e.duration)} · ${e.width}×${e.height} · ${e.clips} clip${e.clips > 1 ? "s" : ""}</div>
        <span class="spacer"></span><a class="btn small" href="/exports/${encodeURIComponent(e.name)}" download>Download</a></div>
      ${i === 0 ? `<div class="path" title="Saved here">${esc(e.path)}</div>` : ""}
    </div>`).join("");
  const box = $("exports");
  if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
}

// ---------------------------------------------------------------------------
// Player

function ensureSource(sid) {
  const s = srcById(sid);
  if (!s) return Promise.resolve();
  if (loaded.sourceId === sid && loaded.preview === s.preview) return Promise.resolve();
  const keep = loaded.sourceId === sid ? player.currentTime : null;
  loaded = { sourceId: sid, preview: s.preview };
  player.src = `/media/${sid}?v=${s.preview}`;
  const badge = $("player-badge");
  badge.hidden = s.preview !== "pending";
  badge.textContent = "Preview is being converted for this browser…";
  return new Promise((resolve) => {
    const done = () => { player.removeEventListener("loadedmetadata", done); player.removeEventListener("error", done); if (keep != null) player.currentTime = keep; resolve(); };
    player.addEventListener("loadedmetadata", done);
    player.addEventListener("error", done);
  });
}

let seekSequence = 0;
async function seek(sid, t) {
  const sequence = ++seekSequence;
  await ensureSource(sid);
  if (sequence !== seekSequence || loaded.sourceId !== sid) return false;
  player.currentTime = Math.max(0, t);
  return true;
}

async function playRange(sid, a, b) {
  if (!await seek(sid, a)) return;
  stopAt = b;
  try { await player.play(); } catch { /* autoplay blocked */ }
}

function stopReel() {
  if (!reelQueue) return;
  ++seekSequence;
  clearTimeout(reelQueue.timer);
  reelQueue = null;
  stopAt = null;
  player.pause();
  $("transport-note").textContent = "";
  renderReel();
}

function playReelAt(index) {
  stopAt = null;
  reelQueue.intro = false;
  const ids = reelQueue.ids;
  if (index >= ids.length) {
    if (data.settings.template !== "recap" || !data.settings.end_card || reelQueue.outro) { stopReel(); return; }
    // Hold the end card on the last clip's canvas, as the export does.
    const queue = reelQueue;
    player.pause();
    stopAt = null;
    queue.outro = true;
    $("transport-note").textContent = "Reel preview · end card";
    queue.timer = setTimeout(() => { if (reelQueue === queue) stopReel(); }, 3000);
    return;
  }
  reelQueue.index = index;
  const c = clipById(ids[index]);
  if (!c) { playReelAt(index + 1); return; }
  $("transport-note").textContent = `Reel preview · clip ${index + 1} of ${ids.length}`;
  sel.clipId = c.id; sel.sourceId = c.source_id;
  render();
  playRange(c.source_id, c.start, c.end);
}

function tick() {
  compose();
  liveTick();
  const t = player.currentTime || 0;
  $("time-readout").textContent = fmt(t);
  if (stopAt != null && ((!player.paused && t >= stopAt - 0.02) || player.ended)) {
    if (reelQueue) {
      playReelAt(reelQueue.index + 1);
    } else {
      player.pause();
      stopAt = null;
    }
  }
  followPlayhead();
  updatePlayheads();
  requestAnimationFrame(tick);
}

function updatePlayheads() {
  const t = player.currentTime || 0;
  document.querySelectorAll(".track-bar").forEach((bar) => {
    const ph = bar.querySelector(".ph");
    const s = srcById(bar.dataset.src);
    if (!ph || !s) return;
    const p = pct(s, t);
    ph.hidden = s.id !== loaded.sourceId || p < 0 || p > 100;
    ph.style.left = p + "%";
  });
  const zph = $("zoom-ph");
  if (zph && zoomWin) {
    const c = clipById(sel.clipId);
    const inWin = c && c.source_id === loaded.sourceId && t >= zoomWin[0] && t <= zoomWin[1];
    zph.hidden = !inWin;
    if (inWin) zph.style.left = ((t - zoomWin[0]) / (zoomWin[1] - zoomWin[0])) * 100 + "%";
  }
  $("play-btn").textContent = player.paused ? "▶" : "❚❚";
}

// ---------------------------------------------------------------------------
// Actions

function selectSource(sid, t = null, doRender = true) {
  stopAt = null;
  player.pause();
  sel.sourceId = sid;
  sel.clipId = null;
  if (t != null) seek(sid, t); else ensureSource(sid);
  if (doRender) render();
}

function selectClip(id, { play = false } = {}) {
  const c = clipById(id);
  if (!c) return;
  stopAt = null;
  player.pause();
  sel.clipId = id;
  sel.sourceId = c.source_id;
  render();
  if (play) playRange(c.source_id, c.start, c.end);
  else seek(c.source_id, c.anchor != null ? Math.max(c.start, c.anchor - 3) : c.start);
}

async function patchClip(id, fields) {
  const c = clipById(id);
  if (!c) return;
  if ("start" in fields || "end" in fields || fields.apply_trim) { player.pause(); stopAt = null; }
  Object.assign(c, fields); // optimistic
  render();
  try {
    const { clip } = await api("PATCH", `/api/clips/${id}`, fields);
    Object.assign(c, clip);
    if (sel.clipId === id && (fields.expected || fields.apply_trim)) {
      await seek(c.source_id, Math.max(c.start, Math.min(c.end, player.currentTime)));
    }
    render();
  } catch (e) {
    toast(e.message, true);
  }
  refresh(true);
}

function nudge(which, delta) {
  const c = clipById(sel.clipId);
  if (!c) return;
  const d = srcById(c.source_id).info.duration;
  let { start, end } = c;
  if (which === "start") start = Math.max(0, Math.min(start + delta, end - 1));
  else end = Math.min(d, Math.max(end + delta, start + 1));
  patchClip(c.id, { start: +start.toFixed(2), end: +end.toFixed(2) });
  seek(c.source_id, which === "start" ? start : Math.max(start, end - 2));
}

function setAtPlayhead(which) {
  const c = clipById(sel.clipId);
  if (!c || c.source_id !== loaded.sourceId) return;
  const t = player.currentTime;
  if (which === "start") patchClip(c.id, { start: +Math.max(0, Math.min(t, c.end - 1)).toFixed(2) });
  else patchClip(c.id, { end: +Math.max(t, c.start + 1).toFixed(2) });
}

function setStatus(status, advance = true) {
  const c = clipById(sel.clipId);
  if (!c) return;
  patchClip(c.id, { status: c.status === status ? "candidate" : status });
  if (advance && c.status !== "candidate") {
    const next = reviewOrder().find((x) => x.status === "candidate" && x.id !== c.id && (x.source_id !== c.source_id || x.start > c.start))
      || reviewOrder().find((x) => x.status === "candidate" && x.id !== c.id);
    if (next) setTimeout(() => selectClip(next.id), 150);
  }
}

function step(dir) {
  stopReel();
  const list = reviewOrder();
  if (!list.length) return;
  let i = list.findIndex((c) => c.id === sel.clipId);
  i = i < 0 ? (dir > 0 ? 0 : list.length - 1) : Math.min(list.length - 1, Math.max(0, i + dir));
  selectClip(list[i].id);
}

async function newClipHere() {
  if (!sel.sourceId) return;
  const t = loaded.sourceId === sel.sourceId ? player.currentTime : 0;
  try {
    const { clip } = await api("POST", "/api/clips", { source_id: sel.sourceId, start: t - 10, end: t + 5 });
    await refresh(true);
    selectClip(clip.id);
    toast("Clip added to the reel. Drag its edges to adjust.");
  } catch (e) { toast(e.message, true); }
}

async function moveInReel(id, dir) {
  const ids = reel().map((c) => c.id);
  const i = ids.indexOf(id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= ids.length) return;
  [ids[i], ids[j]] = [ids[j], ids[i]];
  await saveReelOrder(ids);
}

async function saveReelOrder(ids) {
  // Reel order is the project order of kept clips; others keep their place after.
  try {
    await api("POST", "/api/order", { ids });
    const previous = new Map([...document.querySelectorAll(".reel-item")].map(el => [el.dataset.clip, el.offsetTop]));
    stopReel();
    await refresh(true);
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      for (const el of document.querySelectorAll(".reel-item")) {
        const delta = previous.get(el.dataset.clip) - el.offsetTop;
        if (delta) el.animate([{ transform: `translateY(${delta}px)` }, { transform: "none" }], { duration: 160, easing: "ease-out" });
      }
    }
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------------------
// Import

function uploadFile(file) {
  return new Promise((resolve) => {
    const u = { name: file.name, progress: 0, error: null };
    uploads.push(u);
    renderSources();
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { u.progress = e.loaded / e.total; renderSources(); } };
    const finish = (err) => {
      uploads.splice(uploads.indexOf(u), 1);
      if (err) toast(`${file.name}: ${err}`, true);
      refresh(true).then(resolve);
    };
    xhr.onload = () => {
      let msg = null;
      if (xhr.status >= 400) { try { msg = JSON.parse(xhr.responseText).error; } catch { msg = xhr.statusText; } }
      finish(msg);
    };
    xhr.onerror = () => finish("Upload failed. Is Watch the Goal still running?");
    xhr.send(file);
  });
}

async function importFiles(files) {
  const vids = [...files].filter((f) => /\.(mp4|mov|m4v|mkv|webm|avi|3gp)$/i.test(f.name) || f.type.startsWith("video/"));
  if (!vids.length) { toast("Those don't look like video files.", true); return; }
  for (const f of vids) await uploadFile(f);
}

async function importPaths(text) {
  const paths = text.split(/\n/).map((p) => p.trim()).filter(Boolean);
  if (!paths.length) return;
  try {
    const res = await api("POST", "/api/import", { paths });
    if (res.errors.length) toast(res.errors.join(" · "), true);
    else toast(`Imported ${res.added.length} video${res.added.length === 1 ? "" : "s"}.`);
    $("path-input").value = "";
    await refresh(true);
    if (res.added.length) selectSource(res.added[0].id, 0);
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------------------
// Goal marking

let goalDraft = [];
let goalSource = null;

function openGoals(sid) {
  const s = srcById(sid);
  goalSource = sid;
  goalDraft = s.goal_areas.map((a) => ({ ...a }));
  const t = loaded.sourceId === sid ? player.currentTime : s.info.duration * 0.25;
  $("goal-frame").src = frameUrl(sid, t, 1280);
  drawGoalBoxes();
  $("goal-dialog").showModal();
}

function drawGoalBoxes() {
  $("goal-boxes").innerHTML = goalDraft.map((a) => {
    const label = a.x + a.w / 2 < 0.5 ? "left goal" : "right goal";
    return `<div class="goal-box" style="left:${a.x * 100}%;top:${a.y * 100}%;width:${a.w * 100}%;height:${a.h * 100}%"><span>${label}</span></div>`;
  }).join("");
}

function setupGoalDrawing() {
  const wrap = document.querySelector(".goal-canvas-wrap");
  let startPt = null;
  const norm = (e) => {
    const r = $("goal-frame").getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };
  wrap.addEventListener("pointerdown", (e) => {
    startPt = norm(e);
    if (goalDraft.length >= 2) goalDraft.shift();
    goalDraft.push({ x: startPt.x, y: startPt.y, w: 0, h: 0 });
    try { wrap.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
  });
  wrap.addEventListener("pointermove", (e) => {
    if (!startPt) return;
    const p = norm(e);
    const box = goalDraft[goalDraft.length - 1];
    box.x = Math.min(startPt.x, p.x); box.y = Math.min(startPt.y, p.y);
    box.w = Math.abs(p.x - startPt.x); box.h = Math.abs(p.y - startPt.y);
    drawGoalBoxes();
  });
  wrap.addEventListener("pointerup", () => {
    startPt = null;
    goalDraft = goalDraft.filter((a) => a.w > 0.01 && a.h > 0.01);
    drawGoalBoxes();
  });
  $("goal-clear").onclick = () => { goalDraft = []; drawGoalBoxes(); };
  $("goal-save").onclick = async () => {
    try {
      await api("POST", `/api/sources/${goalSource}/goals`, { areas: goalDraft });
      $("goal-dialog").close();
      const calls = srcById(goalSource)?.calls?.length;
      toast((goalDraft.length ? "Goals saved. Re-analysing with goal-area motion…" : "Goal areas cleared. Re-analysing…")
        + (calls ? " Re-tune afterwards to apply your calls to the new evidence." : ""));
      refresh(true);
    } catch (e) { toast(e.message, true); }
  };
}

// ---------------------------------------------------------------------------
// Zoom strip dragging

function setupZoomDrag() {
  $("editor").addEventListener("pointerdown", (e) => {
    const zoom = e.target.closest("#zoom");
    if (!zoom) return;
    const c = clipById(sel.clipId);
    if (!c || !zoomWin) return;
    const s = srcById(c.source_id);
    const rect = zoom.getBoundingClientRect();
    const toT = (x) => zoomWin[0] + Math.min(1, Math.max(0, (x - rect.left) / rect.width)) * (zoomWin[1] - zoomWin[0]);
    const handle = e.target.id === "h-start" ? "start" : e.target.id === "h-end" ? "end" : null;
    if (!handle) { stopReel(); stopAt = null; seek(c.source_id, e.target.id === "zoom-moment" ? eventTime(c) : toT(e.clientX)); return; }
    e.preventDefault();
    dragging = true;
    const original = { start: c.start, end: c.end };
    stopAt = null;
    stopReel();
    player.pause();
    try { zoom.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    let lastSeek = 0;
    const move = (ev) => {
      const t = toT(ev.clientX);
      if (handle === "start") c.start = +Math.max(0, Math.min(t, c.end - 1)).toFixed(2);
      else c.end = +Math.min(s.info.duration, Math.max(t, c.start + 1)).toFixed(2);
      layoutZoom(c, s);
      const now = performance.now();
      if (now - lastSeek > 60) { lastSeek = now; seek(c.source_id, handle === "start" ? c.start : c.end); }
    };
    const up = (ev) => {
      zoom.removeEventListener("pointermove", move);
      zoom.removeEventListener("pointerup", up);
      zoom.removeEventListener("pointercancel", up);
      dragging = false;
      if (ev.type === "pointercancel") {
        Object.assign(c, original);
        render();
        seek(c.source_id, c[handle]);
        return;
      }
      seek(c.source_id, handle === "start" ? c.start : c.end);
      patchClip(c.id, { start: c.start, end: c.end });
    };
    zoom.addEventListener("pointermove", move);
    zoom.addEventListener("pointerup", up);
    zoom.addEventListener("pointercancel", up);
  });
}

// ---------------------------------------------------------------------------
// Export

async function autoFill() {
  stopReel();
  const value = $("auto-target").value;
  try {
    const res = await api("POST", "/api/reel/auto", { target: value === "all" ? null : Number(value) });
    await refresh(true);
    if (!res.kept.length) {
      toast(res.left ? `Your reel is already ${fmtDur(res.seconds)}. Pick a longer length to add more.` : "No suggestions left to add.");
      return;
    }
    const n = res.kept.length;
    toast(`Kept ${n} suggestion${n > 1 ? "s" : ""}. The reel is now ${fmtDur(res.seconds)}. Undo puts ${n > 1 ? "them" : "it"} back.`);
    if (reel().length) selectClip(res.kept[0]);
  } catch (e) { toast(e.message, true); }
}

async function startExport() {
  stopReel();
  try {
    const { job } = await api("POST", "/api/export", { layout: $("layout-select").value });
    watchedExportJob = job.id;
    refresh(true);
  } catch (e) { toast(e.message, true); }
}

let exportNotified = null;
function watchExport() {
  if (!watchedExportJob) return;
  const job = data.jobs.find((j) => j.id === watchedExportJob);
  if (!job || job.status === "running" || exportNotified === job.id) return;
  exportNotified = job.id;
  if (job.status === "done") toast(`Reel ready: ${fmtDur(job.result.duration)}, ${job.result.width}×${job.result.height}. Play it on the right.`);
  else toast(job.error, true);
}

// ---------------------------------------------------------------------------
// Wiring

// ---------------------------------------------------------------------------
// Analysis settings

let modelInfo = [];
async function openSettings() {
  const st = data.settings;
  const rv = st.ai_review || {};
  $("detector-select").value = st.detector || "none";
  $("review-enabled").checked = !!rv.enabled;
  $("review-model").value = rv.model || "qwen3.5:9b";
  $("review-endpoint").value = rv.endpoint || "http://127.0.0.1:11434";
  $("review-max").value = rv.max_candidates || 6;
  $("review-status").textContent = "";
  $("settings-dialog").showModal();
  try { modelInfo = (await api("GET", "/api/models")).models; } catch { modelInfo = []; }
  renderModelStatus();
}

function renderModelStatus() {
  const name = $("detector-select").value;
  const el = $("model-status");
  const m = modelInfo.find((x) => x.name === name);
  const job = data.jobs.find((j) => j.kind === "download" && j.target === name && j.status === "running");
  if (name === "none" || !m) { el.innerHTML = ""; return; }
  if (job) { el.textContent = `Downloading… ${Math.round(job.progress * 100)}%`; return; }
  el.innerHTML = m.downloaded
    ? `Model ready (${esc(m.license)}, ${esc(m.source)}).`
    : `Not downloaded yet (${(m.size / 1e6).toFixed(0)} MB, ${esc(m.license)}). <button type="button" class="btn small" id="model-download">Download</button>`;
}

async function saveSettings() {
  try {
    await api("POST", "/api/settings", {
      detector: $("detector-select").value,
      ai_review: {
        enabled: $("review-enabled").checked,
        model: $("review-model").value.trim(),
        endpoint: $("review-endpoint").value.trim(),
        max_candidates: +$("review-max").value || 6,
      },
    });
    $("settings-dialog").close();
    toast("Saved. Re-analyse a recording to apply.");
    refresh(true);
  } catch (er) { toast(er.message, true); }
}

async function undoEdit() {
  if (!data.can_undo) return;
  stopReel(); player.pause(); stopAt = null;
  try {
    const edit = await api("POST", "/api/undo", {});
    await refresh(true);
    const c = clipById(edit.clip_id);
    if (c) {
      const t = sel.clipId === c.id ? player.currentTime : c.start;
      sel.clipId = c.id; sel.sourceId = c.source_id;
      await seek(c.source_id, Math.max(c.start, Math.min(c.end, t)));
      render();
    }
    toast({ trim: "Trim undone", status: "Auto-fill undone" }[edit.kind] || "Reel order restored");
  } catch (err) { toast(err.message, true); await refresh(true); }
}
function wire() {
  $("undo-btn").onclick = undoEdit;
  $("template-select").onchange = async e => {
    stopReel();
    if (e.target.value === "recap") return openRecap();
    try {
      await api("POST", "/api/recap", { template: "plain", title: data.settings.title });
      await refresh(true);
    } catch (err) { toast(err.message, true); }
  };
  $("details-btn").onclick = openRecap;
  $("writer-provider").onchange = writerChanged;
  $("writer-btn").onclick = writeRecap;
  $("recap-cancel").onclick = () => $("recap-dialog").close();
  $("recap-dialog").onclose = () => { $("template-select").value = data.settings.template; };
  $("recap-form").onsubmit = async e => {
    e.preventDefault();
    const sources = {};
    for (const row of $("recap-sources").children) {
      sources[row.dataset.sid] = Object.fromEntries([...row.querySelectorAll("input")].map(input => [input.name, input.value]));
    }
    try {
      const captions = Object.fromEntries([...$("recap-captions").querySelectorAll("[data-clip]")]
        .map(row => [row.dataset.clip, row.querySelector("input").value]));
      await api("POST", "/api/recap", { template: "recap", title: $("recap-title").value, end: $("recap-end").value, sources, captions });
      await refresh(true);
      $("recap-dialog").close();
    } catch (err) { toast(err.message, true); }
  };
  $("editor").addEventListener("input", e => {
    if (e.target.id === "crop-position") clipById(sel.clipId).crop = Number(e.target.value) / 100;
  });
  $("editor").addEventListener("change", e => {
    if (e.target.id === "frame-mode") patchClip(sel.clipId, { crop: e.target.value === "fit" ? null : .5 });
    if (e.target.id === "crop-position") patchClip(sel.clipId, { crop: Number(e.target.value) / 100 });
  });
  $("settings-btn").addEventListener("click", openSettings);
  $("settings-save").addEventListener("click", saveSettings);
  $("detector-select").addEventListener("change", renderModelStatus);
  $("model-status").addEventListener("click", async (e) => {
    if (e.target.id !== "model-download") return;
    const name = $("detector-select").value;
    try {
      await api("POST", `/api/models/${name}/download`);
      await refresh(true);
      renderModelStatus();
      const poll = setInterval(async () => {
        await refresh(true);
        const running = data.jobs.find((j) => j.kind === "download" && j.target === name && j.status === "running");
        if (!running) {
          clearInterval(poll);
          modelInfo = (await api("GET", "/api/models")).models;
          const failed = data.jobs.find((j) => j.kind === "download" && j.target === name && j.status === "error");
          if (failed) toast(failed.error, true);
        }
        renderModelStatus();
      }, 1000);
    } catch (er) { toast(er.message, true); }
  });
  $("review-check").addEventListener("click", async () => {
    const out = $("review-status");
    out.textContent = "Checking…";
    try {
      // Check what's typed, without saving it as enabled.
      await api("POST", "/api/settings", { ai_review: { model: $("review-model").value.trim(), endpoint: $("review-endpoint").value.trim() } });
      const r = await api("GET", "/api/review/check");
      out.textContent = r.ok ? `Ready: ${r.model} can read images.` : r.message;
    } catch (er) { out.textContent = er.message; }
  });
  try { $("best-first").checked = bestFirst(); } catch {}
  $("best-first").addEventListener("change", (e) => {
    try { localStorage.setItem("wtg-best-first", e.target.checked ? "1" : "0"); } catch {}
  });

  $("file-input").addEventListener("change", (e) => { importFiles(e.target.files); e.target.value = ""; });
  $("path-form").addEventListener("submit", (e) => { e.preventDefault(); importPaths($("path-input").value); });
  $("reset-btn").addEventListener("click", async () => {
    const btn = $("reset-btn");
    if (!resetArmed) {
      resetArmed = true;
      btn.textContent = "Click again to clear";
      setTimeout(() => { resetArmed = false; btn.textContent = "New project"; }, 3000);
      return;
    }
    resetArmed = false;
    btn.textContent = "New project";
    stopReel();
    await api("POST", "/api/reset");
    player.removeAttribute("src"); player.load();
    loaded = { sourceId: null, preview: null };
    sel.clipId = sel.sourceId = null;
    refresh(true);
  });

  // Drag and drop anywhere.
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { depth++; document.body.classList.add("dragging"); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) document.body.classList.remove("dragging"); });
  window.addEventListener("dragover", (e) => { if (e.dataTransfer?.types.includes("Files")) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!e.dataTransfer?.files.length) return;
    e.preventDefault(); depth = 0; document.body.classList.remove("dragging");
    importFiles(e.dataTransfer.files);
  });

  $("sources").addEventListener("click", (e) => {
    const card = e.target.closest(".source[data-src]");
    if (!card) return;
    const sid = card.dataset.src;
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "goals") return openGoals(sid);
    if (act === "trim") return openTrim(sid);
    if (act === "analyze") return api("POST", `/api/sources/${sid}/analyze`).then(() => refresh(true)).catch((er) => toast(er.message, true));
    if (act === "review") return api("POST", `/api/sources/${sid}/review`).then(() => refresh(true)).catch((er) => toast(er.message, true));
    if (act === "retune") return retune(sid);
    if (act === "untune") return api("POST", `/api/sources/${sid}/tuning/reset`).then(() => refresh(true)).catch((er) => toast(er.message, true));
    if (e.target.closest("a[download]")) return;
    if (act === "remove") {
      return api("DELETE", `/api/sources/${sid}`).then(() => {
        if (loaded.sourceId === sid) { player.removeAttribute("src"); player.load(); loaded = { sourceId: null, preview: null }; }
        if (sel.sourceId === sid) { sel.sourceId = null; sel.clipId = null; }
        refresh(true);
      });
    }
    stopReel();
    const first = clipsFor(sid)[0];
    if (first) selectClip(first.id); else selectSource(sid, 0);
  });

  $("timeline").addEventListener("click", (e) => {
    stopReel();
    const flag = e.target.closest(".call-flag");
    if (flag) {
      const s = srcById(flag.dataset.src);
      const call = s.calls.find((c) => c.id === flag.dataset.callId);
      if (call.clip_id && clipById(call.clip_id)) selectClip(call.clip_id); else selectSource(s.id, call.t);
      return explainAt(s.id, call.t, call.label, `Your call: ${CALL_TEXT[call.label]} at ${fmt(call.t, false)}`);
    }
    const block = e.target.closest(".block");
    if (block) return selectClip(block.dataset.clip);
    const bar = e.target.closest(".track-bar");
    if (!bar) return;
    const s = srcById(bar.dataset.src);
    selectSource(s.id, timeAt(s, bar, e.clientX));
  });

  $("editor").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-act]");
    const act = btn?.dataset.act;
    if (!act) return;
    const c = clipById(sel.clipId);
    if (!c) return;
    stopReel();
    const map = {
      call: () => makeCall(c.source_id, btn.dataset.call, { clipId: c.id }),
      uncall: () => removeCall(c.source_id, callFor(c).id),
      why: () => explainAt(c.source_id, eventTime(c) ?? (c.start + c.end) / 2, callFor(c)?.label,
        `What it measured at ${fmt(eventTime(c) ?? (c.start + c.end) / 2, false)}`),
      "start-5": () => nudge("start", -5), "start-1": () => nudge("start", -1), "start+1": () => nudge("start", 1),
      "end-1": () => nudge("end", -1), "end+1": () => nudge("end", 1), "end+5": () => nudge("end", 5),
      "start-here": () => setAtPlayhead("start"), "end-here": () => setAtPlayhead("end"),
      keep: () => setStatus("kept"), skip: () => setStatus("skipped"),
      "play-clip": () => playRange(c.source_id, c.start, c.end),
      finish: () => { stopAt = null; player.pause(); seek(c.source_id, eventTime(c)); },
      buildup: () => requestEdit(c, "buildup"), sooner: () => requestEdit(c, "sooner"),
      dismiss: () => { proposal = null; renderEditor(); },
      accept: () => {
        const { start, end, expected } = proposal.result;
        proposal = null;
        patchClip(c.id, { start, end, expected });
      },
      "apply-trim": () => patchClip(c.id, { apply_trim: true }),
      delete: () => api("DELETE", `/api/clips/${c.id}`).then(() => { sel.clipId = null; refresh(true); }),
    };
    map[act]?.();
  });
  setupZoomDrag();

  // Reel list: select, move, drag to reorder.
  const list = $("reel-list");
  let dragId = null;
  list.addEventListener("click", (e) => {
    const li = e.target.closest(".reel-item");
    if (!li) return;
    const mv = e.target.closest("[data-move]");
    if (mv) return moveInReel(li.dataset.clip, +mv.dataset.move);
    stopReel();
    selectClip(li.dataset.clip, { play: true });
  });
  list.addEventListener("dragstart", (e) => { dragId = e.target.closest(".reel-item")?.dataset.clip; e.dataTransfer.effectAllowed = "move"; });
  list.addEventListener("dragover", (e) => {
    const li = e.target.closest(".reel-item");
    if (!dragId || !li) return;
    e.preventDefault();
    list.querySelectorAll(".drag-over").forEach((x) => x.classList.remove("drag-over"));
    li.classList.add("drag-over");
  });
  list.addEventListener("drop", (e) => {
    const li = e.target.closest(".reel-item");
    if (!dragId || !li) return;
    e.preventDefault();
    e.stopPropagation();
    const ids = reel().map((c) => c.id).filter((id) => id !== dragId);
    ids.splice(ids.indexOf(li.dataset.clip) + (reel().findIndex((c) => c.id === dragId) < reel().findIndex((c) => c.id === li.dataset.clip) ? 1 : 0), 0, dragId);
    dragId = null;
    saveReelOrder(ids);
  });
  list.addEventListener("dragend", () => { dragId = null; list.querySelectorAll(".drag-over").forEach((x) => x.classList.remove("drag-over")); });

  $("play-btn").addEventListener("click", () => { if (player.paused) player.play().catch(() => {}); else { player.pause(); stopReel(); } });
  $("new-clip-btn").addEventListener("click", newClipHere);
  $("export-btn").addEventListener("click", startExport);
  $("auto-btn").addEventListener("click", autoFill);
  try { $("auto-target").value = localStorage.getItem("wtg-auto-target") || "60"; } catch { /* private mode */ }
  $("auto-target").addEventListener("change", (e) => {
    try { localStorage.setItem("wtg-auto-target", e.target.value); } catch { /* private mode */ }
  });
  $("preview-reel-btn").addEventListener("click", async () => {
    if (reelQueue) return stopReel();
    const ids = reel().map((c) => c.id);
    if (!ids.length) return;
    reelQueue = { ids, index: 0 };
    if (data.settings.template === "recap" && data.settings.title) {
      selectClip(ids[0]); player.pause();
      reelQueue.intro = true;
      $("transport-note").textContent = "Reel preview · opening title";
      const queue = reelQueue;
      const titleLayer = layer("title", outputLayout(), data.settings.title);
      try { await titleLayer.decode(); } catch { toast("Could not load the preview title.", true); }
      if (reelQueue === queue) queue.timer = setTimeout(() => playReelAt(0), 2000);
    } else playReelAt(0);
  });
  $("layout-select").addEventListener("change", async e => {
    stopReel();
    try {
      await api("POST", "/api/settings", { layout: e.target.value });
      await refresh(true);
      if (!sel.clipId && reel().length) selectClip(reel()[0].id);
    } catch (err) { toast(err.message, true); }
  });
  player.addEventListener("seeked", () => { lastFrame = ""; compose(); });
  player.addEventListener("pause", updatePlayheads);
  player.addEventListener("error", () => {
    const s = srcById(loaded.sourceId);
    if (s && s.preview === "pending") $("player-badge").hidden = false;
    else if (s) toast(`Can't preview ${s.name} in this browser. Export still works.`, true);
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea, dialog")) return;
    const k = e.key;
    if (e.target.matches(".handle") && (k === "ArrowLeft" || k === "ArrowRight")) {
      e.preventDefault();
      nudge(e.target.id === "h-start" ? "start" : "end", (k === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 5 : .1));
      return;
    }
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z" && !e.shiftKey) { e.preventDefault(); undoEdit(); return; }
    if (e.target.closest("button") && (k === " " || k === "Enter")) return;
    if (k === " ") { e.preventDefault(); $("play-btn").click(); }
    else if (k === "j" || k === "J") step(-1);
    else if (k === "l" || k === "L") step(1);
    else if (k === "k" || k === "K") setStatus("kept");
    else if (k === "x" || k === "X") setStatus("skipped");
    else if (k === "[") setAtPlayhead("start");
    else if (k === "]") setAtPlayhead("end");
    else if (k === "n" || k === "N") newClipHere();
    else if (k === "g" || k === "G") goalHere();
    else if (k === "d" || k === "D") setLive(!live.on);
    else if (k === ",") stepFrame(-1);
    else if (k === ".") stepFrame(1);
    else if (k === "<") stepSpeed(-1);
    else if (k === ">") stepSpeed(1);
    else if (k === "=" || k === "+") { const s = focusSource(); if (s) zoomBy(s, 0.5); }
    else if (k === "-") { const s = focusSource(); if (s) zoomBy(s, 2); }
    else if (/^[0-5]$/.test(k)) callKey(k);
    else if (k === "ArrowLeft" && e.shiftKey) { e.preventDefault(); nudge("start", -5); }
    else if (k === "ArrowRight" && e.shiftKey) { e.preventDefault(); nudge("end", 5); }
    else if (k === "ArrowLeft") { e.preventDefault(); player.currentTime = Math.max(0, player.currentTime - 2); }
    else if (k === "ArrowRight") { e.preventDefault(); player.currentTime += 2; }
  });
  window.addEventListener("resize", () => { renderTimeline(); renderEditor(); });
  setupGoalDrawing();
  wireCoach();
  wireMatch();
  wireTimelineNav();
}

wire();
refresh(true).then(schedulePoll);
requestAnimationFrame(tick);
