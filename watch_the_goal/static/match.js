// Trimming a recording to the match: kick-off, full time and the halftime
// break. Suggestions, auto-fill and re-tuning ignore everything else.

let trimming = null; // { sid, start, end, breakStart, breakEnd } while the trim bar is open

function playOf(s) {
  const p = s.play || {};
  return { start: p.start ?? 0, end: p.end ?? s.info.duration, breakStart: p.break?.[0] ?? null, breakEnd: p.break?.[1] ?? null };
}

// The parts of a recording outside the match, for shading the timeline.
function offPlay(s) {
  const p = trimming?.sid === s.id ? trimming : playOf(s);
  const d = s.info.duration;
  const gaps = [[0, p.start], [p.end, d]];
  if (p.breakStart != null && p.breakEnd != null) gaps.push([p.breakStart, p.breakEnd]);
  return gaps.filter(([a, b]) => b - a > 0.5).map(([a, b]) =>
    `<div class="offplay" style="left:${pct(s, a)}%;width:${spanPct(s, b - a)}%" aria-hidden="true"></div>`).join("");
}

function playSummary(s) {
  if (!s.play) return "";
  const p = playOf(s);
  const brk = p.breakStart != null ? ` · halftime ${fmt(p.breakStart, false)}–${fmt(p.breakEnd, false)}` : "";
  return `<div class="status">Match ${fmt(p.start, false)}–${fmt(p.end, false)}${brk}</div>`;
}

function openTrim(sid) {
  const s = srcById(sid);
  if (!s) return;
  trimming = { sid, ...playOf(s) };
  selectSource(sid, trimming.start);
  renderTrim();
}

function closeTrim() {
  trimming = null;
  renderTrim();
  renderTimeline();
}

function renderTrim() {
  const bar = $("trim-bar");
  const s = trimming && srcById(trimming.sid);
  bar.hidden = !s;
  if (!s) return;
  const p = trimming;
  const t = (v) => (v == null ? "—" : fmt(v, false));
  bar.innerHTML = `
    <h4>Trim ${esc(s.name)} to the match</h4>
    <div class="muted">Scrub to each moment and set it from the playhead. Suggestions, auto-fill and re-tuning ignore the rest.</div>
    <div class="match-row">
      <span class="lbl">Kick-off</span><span class="t">${t(p.start)}</span>
      <button class="btn small" data-trim="start">Set here</button>
      <span class="lbl">Halftime</span><span class="t">${t(p.breakStart)}–${t(p.breakEnd)}</span>
      <button class="btn small" data-trim="breakStart">Starts here</button>
      <button class="btn small" data-trim="breakEnd">Ends here</button>
      ${p.breakStart != null || p.breakEnd != null ? `<button class="btn small ghost" data-trim="noBreak">No halftime</button>` : ""}
      <span class="lbl">Full time</span><span class="t">${t(p.end)}</span>
      <button class="btn small" data-trim="end">Set here</button>
    </div>
    <div class="match-row">
      <button class="btn small primary" data-trim="save">Save and re-rank</button>
      ${s.play ? `<button class="btn small ghost" data-trim="whole">Use the whole recording</button>` : ""}
      <button class="btn small ghost" data-trim="cancel">Cancel</button>
    </div>`;
}

async function saveTrim(play) {
  const sid = trimming.sid;
  try {
    const res = await api("POST", `/api/sources/${sid}/play`, { play });
    closeTrim();
    await refresh(true);
    toast(res.play ? `Trimmed to ${fmt(res.play.start, false)}–${fmt(res.play.end, false)}. Re-ranking suggestions…`
      : "Using the whole recording. Re-ranking suggestions…");
  } catch (e) { toast(e.message, true); }
}

function trimAction(what) {
  const p = trimming;
  if (!p) return;
  const t = loaded.sourceId === p.sid ? player.currentTime : null;
  if (["start", "end", "breakStart", "breakEnd"].includes(what)) {
    if (t == null) return toast("Play this recording first, then set the time from the playhead.", true);
    p[what] = Math.round(t * 10) / 10;
  } else if (what === "noBreak") {
    p.breakStart = p.breakEnd = null;
  } else if (what === "save") {
    if ((p.breakStart == null) !== (p.breakEnd == null)) return toast("Set where halftime ends as well as where it starts.", true);
    return saveTrim({ start: p.start, end: p.end, break: p.breakStart == null ? null : [p.breakStart, p.breakEnd] });
  } else if (what === "whole") {
    return saveTrim(null);
  } else if (what === "cancel") {
    return closeTrim();
  }
  renderTrim();
  renderTimeline();
}

function wireMatch() {
  $("trim-bar").addEventListener("click", (e) => {
    const what = e.target.closest("[data-trim]")?.dataset.trim;
    if (what) trimAction(what);
  });
}
