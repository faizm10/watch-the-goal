"use strict";
// Live detection (what the detector hears and sees at the playhead) and the
// user's calls (what really happened), which re-tune a recording's analysis.
// Shares globals with app.js; nothing here runs before app.js has loaded.

// [label, button text, key]
const CALLS = [["goal", "Goal", "1"], ["save", "Save", "2"], ["near_miss", "Near miss", "3"], ["chance", "Chance", "4"],
  ["tackle", "Tackle", "5"], ["other", "Not a highlight", "0"]];
const CALL_TEXT = { goal: "goal", save: "save", near_miss: "near miss", chance: "chance", tackle: "tackle", other: "not a highlight" };
const CALL_PLURAL = { goal: "goals", save: "saves", near_miss: "near misses", chance: "chances", tackle: "tackles", other: "not highlights" };
// What a reel is made of (coach.KIND): a save and a near miss are both shots.
const KIND = { goal: "goal", save: "shot", near_miss: "shot", chance: "chance", tackle: "tackle" };
const KINDS = [["goal", "goals"], ["shot", "shots"], ["chance", "chances"], ["tackle", "tackles"]];
const kindOf = (c) => KIND[callFor(c)?.label] || null;
const EVENT_COLORS = { audio: "#78aaff", goal: "#00c8ff", lull: "#9aa7a0" };
const LIVE_SIDE = 960; // px; both detectors downscale further (640 and 384)
const GOAL_MARGIN = 0.5; // as vision/players.py: "near" is the goal box widened by half its size

const evidence = {}; // sourceId -> { key, data } from /api/sources/:id/evidence
const live = { on: false, busy: false, result: null, sentAt: 0, retryAt: 0, hud: "" };
try { live.on = localStorage.getItem("wtg-live") === "1"; } catch { /* private mode */ }

const callFor = (c) => (srcById(c.source_id)?.calls || []).find((x) => x.clip_id === c.id);

// ---------------------------------------------------------------------------
// Evidence on the timeline

function loadEvidence(s) {
  const key = `${s.analysis?.status}:${s.analysis?.seconds}:${JSON.stringify(s.tuning)}`;
  if (s.kind !== "long" || s.analysis?.status !== "done" || evidence[s.id]?.key === key) return;
  evidence[s.id] = { key };
  api("GET", `/api/sources/${s.id}/evidence`).then((d) => {
    evidence[s.id] = { key, data: d };
    renderTimeline();
  }).catch(() => {});
}

function drawEvents(canvas, s) {
  // Every raw event the detector heard or saw, suggested or not: short ticks along the top.
  const ev = evidence[s.id]?.data;
  if (!canvas || !ev?.events) return;
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = canvas.clientWidth;
  for (const e of ev.events) {
    ctx.fillStyle = EVENT_COLORS[e.kind] || "#fff";
    ctx.fillRect(Math.round((pct(s, e.t) / 100) * w), 0, 1.5, e.kind === "lull" ? 4 : 8);
  }
}

function callFlags(s) {
  return (s.calls || []).map((c) => `<button class="call-flag ${c.label}" data-call-id="${c.id}" data-src="${s.id}"
    style="left:${pct(s, c.t)}%" title="Your call: ${CALL_TEXT[c.label]} at ${fmt(c.t)}"
    aria-label="Your call: ${CALL_TEXT[c.label]} at ${fmt(c.t)}"></button>`).join("");
}

// ---------------------------------------------------------------------------
// Calls

function callsRow(c) {
  const call = callFor(c);
  const buttons = CALLS.map(([k, label, key]) => `<button class="btn small call ${call?.label === k ? "on" : ""}" data-act="call"
    data-call="${k}" aria-pressed="${call?.label === k}" title="${label} (${key})">${label}</button>`).join("");
  return `<div class="calls-row" role="group" aria-label="Your call"><span class="lbl">Your call</span>${buttons}
    ${call ? `<button class="btn small ghost" data-act="uncall" title="Remove your call">Clear</button>` : ""}
    <button class="btn small ghost" data-act="why" title="What the detector measured at this moment">Why?</button></div>`;
}

async function makeCall(sid, label, { clipId = null, t = null, select = false } = {}) {
  try {
    const res = await api("POST", `/api/sources/${sid}/calls`, { label, clip_id: clipId, t });
    await refresh(true);
    if (select && res.call.clip_id && clipById(res.call.clip_id)) {
      sel.clipId = res.call.clip_id; sel.sourceId = sid;
      render();
    }
    showCoachNote(sid, `Your call: ${CALL_TEXT[label]} at ${fmt(res.call.t, false)}`, res.explain, res.calls);
  } catch (e) { toast(e.message, true); }
}

async function removeCall(sid, callId) {
  try {
    await api("DELETE", `/api/sources/${sid}/calls/${callId}`);
    await refresh(true);
  } catch (e) { toast(e.message, true); }
}

async function explainAt(sid, t, label, title) {
  try {
    const q = new URLSearchParams({ t: t.toFixed(2), ...(label ? { label } : {}) });
    const ex = await api("GET", `/api/sources/${sid}/explain?${q}`);
    showCoachNote(sid, title, ex, (srcById(sid)?.calls || []).length);
  } catch (e) { toast(e.message, true); }
}

function showCoachNote(sid, title, ex, calls) {
  const s = srcById(sid);
  const el = $("coach-note");
  const retune = s?.kind === "long" && calls
    ? `<div class="row"><button class="btn small" data-act="retune" data-src="${sid}">Re-tune from my calls</button>
        <span class="muted small">${calls} call${calls === 1 ? "" : "s"} on this recording. Re-tuning moves its bars just far enough to agree with them.</span></div>`
    : "";
  el.innerHTML = `<h4><span>${esc(title)}</span><span class="spacer"></span>
      <button class="btn small ghost" data-act="close-note" aria-label="Close">✕</button></h4>
    <ul>${(ex.lines || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>${retune}`;
  el.dataset.src = sid;
  el.hidden = false;
}

async function retune(sid) {
  try {
    await api("POST", `/api/sources/${sid}/retune`);
    $("coach-note").hidden = true;
    toast("Re-tuning from your calls…");
    await refresh(true);
  } catch (e) { toast(e.message, true); }
}

function coachHtml(s, busy) {
  if (s.kind !== "long") return "";
  const calls = s.calls || [];
  const counts = CALLS.map(([k]) => [k, calls.filter((c) => c.label === k).length]).filter(([, n]) => n);
  const summary = counts.map(([k, n]) => `${n} ${n === 1 ? CALL_TEXT[k] : CALL_PLURAL[k]}`).join(" · ");
  let html = `<div class="coach"><div>Your calls: ${summary || "none yet"}${s.tuning ? ' <span class="badge tuned" title="Thresholds learned from your calls">tuned</span>' : ""}</div>`;
  const co = s.coaching;
  if (co) {
    const highs = co.calls.filter((c) => c.label !== "other");
    const found = highs.filter((c) => c.verdict === "suggested").length;
    const lines = [...co.changes, ...co.notes];
    const falses = co.calls.filter((c) => c.label === "other");
    const dropped = falses.filter((c) => c.verdict !== "suggested").length;
    const parts = [`${co.before} → ${co.after} suggestions`];
    if (highs.length) parts.push(`finds ${found} of your ${highs.length} highlight${highs.length > 1 ? "s" : ""}`);
    if (falses.length) parts.push(`drops ${dropped} of ${falses.length} you rejected`);
    if (co.moved) parts.push(`${co.moved} ranked by resemblance`);
    html += `<div class="muted">Re-tuned: ${parts.join(" · ")}</div>`;
    const basis = calls.map((c) => `${c.id}:${c.label}`).sort().join(",");
    if (co.basis != null && basis !== co.basis) html += `<div class="status">You've made calls since. Re-tune to use them.</div>`;
    if (lines.length) html += `<ul>${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
  } else if (!calls.length) {
    html += `<div class="muted">Tag suggestions with 1–5, or press the same keys on a moment it missed. Then re-tune.</div>`;
  }
  html += `<div class="actions">
      <button class="btn small" data-act="retune" ${!calls.length || busy ? "disabled" : ""} title="Move this recording's thresholds to agree with your calls, then re-analyse">Re-tune from my calls</button>
      ${s.tuning ? `<button class="btn small ghost" data-act="untune" ${busy ? "disabled" : ""} title="Back to the default thresholds">Use defaults</button>` : ""}
      ${calls.length ? `<a class="btn small ghost" href="/api/sources/${s.id}/labels.json" download title="Your calls in the format scripts/eval_highlights.py reads">labels.json</a>` : ""}
    </div></div>`;
  return html;
}

// ---------------------------------------------------------------------------
// Live detection

function setLive(on) {
  live.on = on;
  try { localStorage.setItem("wtg-live", on ? "1" : "0"); } catch { /* private mode */ }
  $("live-btn").setAttribute("aria-pressed", String(on));
  live.hud = "";
  if (!on) { $("live-overlay").hidden = true; $("live-hud").hidden = true; }
}

function liveTick() {
  const show = live.on && !!loaded.sourceId && $("composition").hidden && player.readyState >= 2;
  $("live-overlay").hidden = !show;
  $("live-hud").hidden = !show;
  if (!show) return;
  pumpLive();
  drawLiveOverlay();
  renderHud();
}

let liveCanvas = null;
function pumpLive() {
  // One frame in flight at a time: about as many per second as the detector can manage.
  const now = performance.now();
  if (live.busy || now < live.retryAt) return;
  const sid = loaded.sourceId, t = player.currentTime;
  const last = live.result;
  if (last && last.sid === sid && Math.abs(last.t - t) < 0.04) return; // paused on a frame we've seen
  const vw = player.videoWidth, vh = player.videoHeight;
  if (!vw || !vh) return;
  const scale = Math.min(1, LIVE_SIDE / Math.max(vw, vh));
  liveCanvas ||= document.createElement("canvas");
  liveCanvas.width = Math.round(vw * scale);
  liveCanvas.height = Math.round(vh * scale);
  liveCanvas.getContext("2d").drawImage(player, 0, 0, liveCanvas.width, liveCanvas.height);
  live.busy = true;
  live.sentAt = now;
  liveCanvas.toBlob(async (blob) => {
    try {
      const res = await fetch(`/api/sources/${sid}/live?t=${t.toFixed(3)}`, {
        method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob,
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.error || res.statusText);
      live.result = { ...d, sid, t };
      // Without a downloaded model there's nothing to show; check again now and then.
      if (!d.detector) live.retryAt = performance.now() + 5000;
    } catch {
      live.retryAt = performance.now() + 2000;
    } finally {
      live.busy = false;
    }
  }, "image/jpeg", 0.8);
}

function videoRect(canvas) {
  const wrap = canvas.getBoundingClientRect(), pr = player.getBoundingClientRect();
  const scale = Math.min(pr.width / player.videoWidth, pr.height / player.videoHeight);
  const w = player.videoWidth * scale, h = player.videoHeight * scale;
  return { x: pr.left - wrap.left + (pr.width - w) / 2, y: pr.top - wrap.top + (pr.height - h) / 2, w, h };
}

function drawLiveOverlay() {
  const canvas = $("live-overlay");
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const r = videoRect(canvas);
  const s = srcById(loaded.sourceId);
  const res = live.result?.sid === loaded.sourceId ? live.result : null;
  const areas = s?.goal_areas || [];
  ctx.font = "600 11px -apple-system, system-ui, sans-serif";
  areas.forEach((a, i) => {
    const x = r.x + a.x * r.w, y = r.y + a.y * r.h, w = a.w * r.w, h = a.h * r.h;
    ctx.strokeStyle = "#00c8ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeRect(x - GOAL_MARGIN * w, y - GOAL_MARGIN * h, w * (1 + 2 * GOAL_MARGIN), h * (1 + 2 * GOAL_MARGIN));
    ctx.setLineDash([]);
    const near = res?.near?.[i];
    const text = near ? `${near.label} · ${near.count} near${near.blocked ? " · lens blocked" : ""}`
      : (a.x + a.w / 2 < 0.5 ? "left goal" : "right goal");
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = near?.blocked ? "#ff6b5e" : "#00c8ff";
    ctx.fillRect(x, y - 17, tw + 10, 16);
    ctx.fillStyle = "#03161c";
    ctx.fillText(text, x + 5, y - 5);
  });
  if (!res?.boxes) return;
  // Boxes from an older frame fade, so a slow detector never looks like it's tracking.
  ctx.globalAlpha = Math.abs(res.t - player.currentTime) > 0.6 ? 0.35 : 1;
  for (const [x0, y0, x1, y1] of res.boxes) {
    const fx = (x0 + x1) / 2, fy = y1;
    const near = areas.some((a) => fx >= a.x - GOAL_MARGIN * a.w && fx <= a.x + (1 + GOAL_MARGIN) * a.w
      && fy >= a.y - GOAL_MARGIN * a.h && fy <= a.y + (1 + GOAL_MARGIN) * a.h);
    ctx.strokeStyle = near ? "#3fd17a" : "#ffdc00";
    ctx.lineWidth = near ? 2 : 1.5;
    ctx.strokeRect(r.x + x0 * r.w, r.y + y0 * r.h, (x1 - x0) * r.w, (y1 - y0) * r.h);
  }
  ctx.globalAlpha = 1;
}

function hudMeter(label, ratio, text) {
  // The white tick is the bar the signal has to clear; the scale runs to twice that.
  const fill = Math.max(0, Math.min(1, (ratio ?? 0) / 2)) * 100;
  return `<div class="hud-row"><span class="k">${esc(label)}</span><div class="m">
    <div class="mbar ${ratio >= 1 ? "hot" : ""}"><i style="width:${fill.toFixed(1)}%"></i><b></b></div>
    <span class="mval">${esc(text)}</span></div></div>`;
}

const dB = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB`;

function hudVerdict(s, t, ev) {
  const out = [];
  const clip = clipsFor(s.id).find((c) => c.start <= t && t <= c.end);
  if (clip) {
    const what = clip.origin === "detected" ? `Suggestion${clip.rank ? ` #${clip.rank}` : ""}` : clip.origin === "manual" ? "Your clip" : "Clip";
    out.push(`<span class="hit">● ${what} covers this (${clip.status === "candidate" ? "suggested" : clip.status})</span>`);
  }
  const heard = (ev?.events || []).filter((e) => e.t - 1 <= t && t <= (e.end ?? e.t) + 1);
  for (const e of heard.slice(0, 2)) out.push(`<span class="heard">◆ ${esc(e.reason)}</span>`);
  const call = (s.calls || []).find((c) => Math.abs(c.t - t) < 3);
  if (call) out.push(`<span class="call">⚑ Your call: ${CALL_TEXT[call.label]}</span>`);
  if (!out.length) {
    const i = ev?.step ? Math.floor(t / ev.step) : -1;
    out.push(ev?.audio_ratio?.[i] >= 1
      ? `<span class="muted">Loud for a moment, but not held long enough to count as crowd noise</span>`
      : `<span class="muted">Nothing flagged here</span>`);
  }
  return `<div class="hud-verdict">${out.join("")}</div>`;
}

function renderHud() {
  const s = srcById(loaded.sourceId);
  if (!s) return;
  const t = player.currentTime;
  const ev = evidence[s.id]?.data;
  const res = live.result?.sid === s.id ? live.result : null;
  const rows = [];
  const series = ev?.flow || ev?.audio_db;
  if (ev?.step && series) {
    const i = Math.max(0, Math.min(Math.floor(t / ev.step), series.length - 1));
    if (ev.audio_db) rows.push(hudMeter("Crowd noise", ev.audio_ratio[i], `${dB(ev.audio_db[i])} · counts at ${dB(ev.audio_needs_db)}`));
    else rows.push(`<div class="hud-row"><span class="k">Crowd noise</span><span class="mval muted">no audio</span></div>`);
    for (const g of ev.goals) {
      const label = g.label[0].toUpperCase() + g.label.slice(1);
      rows.push(hudMeter(label, g.ratio[i], `${Math.round(Math.max(0, g.ratio[i]) * 100)}% of the bar`));
    }
    if (ev.flow) rows.push(hudMeter("Tempo", ev.flow[i] / ev.surge, `${ev.flow[i].toFixed(1)}× usual · surge at ${ev.surge}×`));
  } else if (s.kind === "long") {
    rows.push(`<div class="muted">Crowd noise and motion appear once analysis finishes.</div>`);
  }
  if (res?.detector) {
    const typ = res.typical;
    const near = (res.near || []).map((n) => ` · ${n.count} near ${n.label}${typ ? ` (usually ${Math.round(typ.near_goal)})` : ""}`).join("");
    const bunch = res.bunch >= 2 ? ` · bunch of ${res.bunch}${typ ? ` (usually ${Math.round(typ.bunch)})` : ""}` : "";
    rows.push(`<div class="hud-row"><span class="k">Players</span><span class="mval" title="${esc(res.people + " seen" + near + bunch)}">${res.people} seen${near}${bunch}</span></div>`);
  } else if (res && !res.detector) {
    rows.push(`<div class="hud-row"><span class="k">Players</span><span class="mval muted">${esc(res.message || "no detector")}</span></div>`);
  }
  const head = `<div class="hd"><span class="dot">●</span><span>Live</span><span class="spacer"></span>
    ${res?.detector ? `<span>${esc(res.detector)} · ${res.ms} ms</span>` : ""}</div>`;
  const html = head + rows.join("") + hudVerdict(s, t, ev);
  if (html !== live.hud) { $("live-hud").innerHTML = html; live.hud = html; }
}

// ---------------------------------------------------------------------------
// Wiring (called from app.js wire())

function wireCoach() {
  $("live-btn").setAttribute("aria-pressed", String(live.on));
  $("live-btn").addEventListener("click", () => setLive(!live.on));
  $("missed-btn").addEventListener("click", goalHere);
  $("coach-note").addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "close-note") $("coach-note").hidden = true;
    if (act === "retune") retune(e.target.closest("[data-src]").dataset.src);
  });
}

function goalHere() {
  if (!loaded.sourceId) return;
  stopReel();
  makeCall(loaded.sourceId, "goal", { t: player.currentTime, select: true });
}

// A number key calls the selected clip while the playhead is in it. Anywhere
// else it tags the moment at the playhead, adding a kept clip for a highlight.
function callKey(key) {
  const entry = CALLS.find(([, , k]) => k === key);
  if (!entry) return;
  const c = clipById(sel.clipId);
  const t = player.currentTime;
  if (c && (c.source_id !== loaded.sourceId || (t >= c.start && t <= c.end))) {
    makeCall(c.source_id, entry[0], { clipId: c.id });
  } else if (loaded.sourceId) {
    stopReel();
    makeCall(loaded.sourceId, entry[0], { t, select: true });
  }
}

function reelMix(items) {
  const counts = Object.fromEntries(KINDS.map(([k]) => [k, 0]));
  let untagged = 0;
  for (const c of items) {
    const k = kindOf(c);
    if (k) counts[k] += 1; else untagged += 1;
  }
  const have = KINDS.filter(([k]) => counts[k]).map(([k, plural]) =>
    `<span class="kind" data-kind="${k}">${counts[k]} ${counts[k] === 1 ? k : plural}</span>`);
  if (untagged) have.push(`<span class="kind">${untagged} untagged</span>`);
  const missing = KINDS.filter(([k]) => !counts[k]).map(([, plural]) => plural);
  const tail = missing.length
    ? `<span class="muted">No ${missing.length > 1 ? `${missing.slice(0, -1).join(", ")} or ${missing.at(-1)}` : missing[0]} yet. Tag them as you watch: 1 goal, 2–3 shot, 4 chance, 5 tackle.</span>`
    : "";
  return `${have.join(" ")} ${tail}`;
}
