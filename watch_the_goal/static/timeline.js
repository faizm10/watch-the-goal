// Getting around a recording: zoom and pan the timeline, drag along a track to
// scrub, hover for a frame preview, and change speed or skip in the transport.

const views = {}; // source id -> [t0, t1] while zoomed in
const MIN_SPAN = 20; // seconds across the track at the deepest zoom
const SPEEDS = [0.5, 1, 1.5, 2, 4];
const TICK_STEPS = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
let scrubbing = null; // source id while dragging along a track

function viewOf(s) {
  return views[s.id] || [0, s.info.duration];
}
// Where t sits across a track, in percent of its width (outside 0–100 when off screen).
function pct(s, t) {
  const [a, b] = viewOf(s);
  return ((t - a) / (b - a)) * 100;
}
function spanPct(s, len) {
  const [a, b] = viewOf(s);
  return (len / (b - a)) * 100;
}
function timeAt(s, bar, clientX) {
  const r = bar.getBoundingClientRect();
  const [a, b] = viewOf(s);
  return a + Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * (b - a);
}

function setView(s, t0, t1) {
  const d = s.info.duration;
  const span = Math.min(d, Math.max(Math.min(MIN_SPAN, d), t1 - t0));
  const a = Math.min(Math.max(0, t0), d - span);
  if (span >= d - 0.01) delete views[s.id]; else views[s.id] = [a, a + span];
  renderTimeline();
}

// Zoom by ``factor`` (below 1 zooms in) keeping ``at`` (default: the playhead) where it is.
function zoomBy(s, factor, at = null) {
  const [a, b] = viewOf(s);
  if (at == null) at = loaded.sourceId === s.id ? player.currentTime : (a + b) / 2;
  const span = (b - a) * factor;
  const f = (at - a) / (b - a);
  setView(s, at - f * span, at - f * span + span);
}

function focusSource() {
  return srcById(sel.sourceId) || srcById(loaded.sourceId) || data.sources[0] || null;
}

function rulerHtml(s) {
  const [a, b] = viewOf(s);
  const step = TICK_STEPS.find((x) => (b - a) / x <= 6) || 3600;
  const out = [];
  for (let t = Math.ceil(a / step) * step; t <= b + 0.01; t += step) {
    const p = pct(s, t);
    const edge = p < 4 ? "start" : p > 96 ? "end" : "";
    out.push(`<span class="${edge}" style="left:${p}%">${fmt(t, false)}</span>`);
  }
  return out.join("");
}

// A strip under a zoomed track showing which part of the recording is on screen; drag it to pan.
function overviewHtml(s) {
  if (!views[s.id]) return "";
  const d = s.info.duration;
  const [a, b] = views[s.id];
  return `<div class="track-overview" data-src="${s.id}" aria-label="Visible part of ${esc(s.name)}; drag to pan">
    <div class="win" style="left:${(a / d) * 100}%;width:${((b - a) / d) * 100}%"></div></div>`;
}

function zoomLabel() {
  const s = focusSource();
  if (!s) return "";
  const [a, b] = viewOf(s);
  return views[s.id] ? `${fmt(a, false)}–${fmt(b, false)}` : "Whole recording";
}

function renderNav() {
  const s = focusSource();
  $("zoom-out").disabled = !s || !views[s.id];
  $("zoom-fit").disabled = !s || !views[s.id];
  $("zoom-in").disabled = !s || (viewOf(s)[1] - viewOf(s)[0]) <= MIN_SPAN + 0.01;
  $("zoom-label").textContent = zoomLabel();
}

// Keep the playhead on screen while a zoomed track plays.
function followPlayhead() {
  const s = srcById(loaded.sourceId);
  if (!s || !views[s.id] || player.paused || scrubbing) return;
  const [a, b] = views[s.id];
  const t = player.currentTime;
  if (t > b - (b - a) * 0.02 || t < a) setView(s, t - (b - a) * 0.1, t + (b - a) * 0.9);
}

// --- hover preview ------------------------------------------------------------

const scrubVideo = document.createElement("video");
scrubVideo.muted = true;
scrubVideo.preload = "auto";
let preview = { sid: null, want: null, busy: false };

function showPreview(s, bar, clientX) {
  const tip = $("scrub-tip");
  const t = timeAt(s, bar, clientX);
  const wrap = tip.parentElement.getBoundingClientRect(); // .timeline-wrap; offsetParent is null while hidden
  const r = bar.getBoundingClientRect();
  tip.hidden = false;
  tip.style.left = `${Math.min(Math.max(clientX - wrap.left - 82, 4), wrap.width - 168)}px`;
  tip.style.top = `${r.top - wrap.top - 112}px`;
  tip.querySelector(".t").textContent = fmt(t, false);
  if (s.preview === "pending") {
    // Not playable in this browser yet: ask the server for the frame instead.
    const img = tip.querySelector("img");
    img.hidden = false;
    tip.querySelector("canvas").hidden = true;
    img.src = frameUrl(s.id, Math.round(t), 160);
    return;
  }
  tip.querySelector("img").hidden = true;
  tip.querySelector("canvas").hidden = false;
  if (preview.sid !== s.id) {
    preview = { sid: s.id, want: null, busy: false };
    scrubVideo.src = `/media/${s.id}?v=${s.preview}`;
  }
  preview.want = t;
  if (!preview.busy && scrubVideo.readyState >= 1) seekPreview();
}

function seekPreview() {
  if (preview.want == null) return;
  preview.busy = true;
  scrubVideo.currentTime = preview.want;
  preview.want = null;
}

scrubVideo.addEventListener("loadedmetadata", seekPreview);
scrubVideo.addEventListener("seeked", () => {
  const canvas = $("scrub-tip").querySelector("canvas");
  canvas.getContext("2d").drawImage(scrubVideo, 0, 0, canvas.width, canvas.height);
  preview.busy = false;
  seekPreview(); // the pointer moved on while that frame was decoding
});

function hidePreview() {
  $("scrub-tip").hidden = true;
}

// --- transport ----------------------------------------------------------------

function setSpeed(rate) {
  player.playbackRate = rate;
  $("speed").value = String(rate);
  try { localStorage.setItem("wtg-speed", String(rate)); } catch { /* private mode */ }
}

function stepSpeed(dir) {
  const i = SPEEDS.indexOf(player.playbackRate);
  setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, Math.max(0, (i < 0 ? 1 : i) + dir))]);
}

function skip(seconds) {
  if (!loaded.sourceId) return;
  stopReel();
  const s = srcById(loaded.sourceId);
  player.currentTime = Math.min(Math.max(0, player.currentTime + seconds), s.info.duration);
}

// One frame at a time while paused (most footage is 25–30 fps).
function stepFrame(dir) {
  if (!loaded.sourceId) return;
  player.pause();
  skip(dir / 30);
}

function wireTimelineNav() {
  const tl = $("timeline");
  $("zoom-in").addEventListener("click", () => { const s = focusSource(); if (s) zoomBy(s, 0.5); });
  $("zoom-out").addEventListener("click", () => { const s = focusSource(); if (s) zoomBy(s, 2); });
  $("zoom-fit").addEventListener("click", () => { const s = focusSource(); if (s) setView(s, 0, s.info.duration); });
  $("skip-back").addEventListener("click", () => skip(-10));
  $("skip-fwd").addEventListener("click", () => skip(10));
  $("speed").addEventListener("change", (e) => setSpeed(Number(e.target.value)));
  try { setSpeed(Number(localStorage.getItem("wtg-speed")) || 1); } catch { setSpeed(1); }
  // The browser resets the rate when a new source loads.
  player.addEventListener("loadedmetadata", () => { player.playbackRate = Number($("speed").value) || 1; });

  // Ctrl/⌘ + wheel (and trackpad pinch) zooms around the pointer; sideways or Shift + wheel pans.
  tl.addEventListener("wheel", (e) => {
    const bar = e.target.closest(".track-bar");
    const s = bar && srcById(bar.dataset.src);
    if (!s) return;
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      zoomBy(s, Math.exp(e.deltaY * 0.01), timeAt(s, bar, e.clientX));
    } else if (views[s.id] && (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY))) {
      e.preventDefault();
      const [a, b] = views[s.id];
      const dx = (e.shiftKey ? e.deltaY : e.deltaX) / bar.clientWidth * (b - a);
      setView(s, a + dx, b + dx);
    }
  }, { passive: false });

  // Drag along a track to scrub; a plain click still seeks (see the click handler).
  // Moves are tracked on the window: the timeline re-renders while panning.
  const barOf = (sid) => tl.querySelector(`.track-bar[data-src="${sid}"]`);
  tl.addEventListener("pointerdown", (e) => {
    const bar = e.target.closest(".track-bar");
    if (!bar || e.button !== 0 || e.target.closest(".block, .call-flag")) return;
    scrubbing = bar.dataset.src;
    e.preventDefault(); // no text selection while dragging
  });
  window.addEventListener("pointermove", (e) => {
    const bar = scrubbing ? barOf(scrubbing) : e.target.closest?.(".track-bar");
    const s = bar && srcById(bar.dataset.src);
    if (!s) return hidePreview();
    showPreview(s, bar, e.clientX);
    if (scrubbing) {
      stopReel();
      seek(s.id, timeAt(s, bar, e.clientX));
    }
  });
  const endScrub = () => { if (scrubbing) { scrubbing = null; hidePreview(); } };
  window.addEventListener("pointerup", endScrub);
  window.addEventListener("pointercancel", endScrub);

  // Pan by dragging the window in the overview strip (or click to jump there).
  tl.addEventListener("pointerdown", (e) => {
    const strip = e.target.closest(".track-overview");
    if (!strip) return;
    e.preventDefault();
    const s = srcById(strip.dataset.src);
    const r = strip.getBoundingClientRect();
    const [a, b] = viewOf(s);
    const d = s.info.duration;
    const grab = e.target.closest(".win") ? (e.clientX - r.left) / r.width * d - a : (b - a) / 2;
    const move = (ev) => {
      const t = (ev.clientX - r.left) / r.width * d - grab;
      setView(s, t, t + (b - a));
    };
    move(e);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", () => window.removeEventListener("pointermove", move), { once: true });
  });
}
