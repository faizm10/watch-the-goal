// The editor and reel preview share this canvas. Artwork is the exact PNG used by FFmpeg,
// and captions and fades follow the same timing as the export (see recap.py).
const CAPTION_IN = 0.4, CAPTION_EASE = 0.35, CAPTION_HOLD = 4.0, FADE = 0.25;
const layouts = { landscape: [1920, 1080], portrait: [1080, 1920], square: [1080, 1080] };
let shapeForPreview = "landscape";
function outputLayout() {
  if (layouts[data.settings.layout]) return data.settings.layout;
  let balance = 0;
  for (const c of reel()) {
    const info = srcById(c.source_id).info;
    balance += clipLen(c) * (info.height > info.width ? 1 : -1);
  }
  return balance > 0 ? "portrait" : "landscape";
}

let lastFrame = "";
const images = new Map();
function layer(kind, layout, text) {
  const url = "/api/artwork?" + new URLSearchParams({ kind, layout, text });
  if (!images.has(url)) {
    if (images.size >= 16) images.delete(images.keys().next().value);
    const img = new Image();
    img.onload = () => { lastFrame = ""; };
    img.src = url;
    images.set(url, img);
  }
  return images.get(url);
}

function compose() {
  const canvas = $("composition");
  const c = clipById(sel.clipId);
  const shape = shapeForPreview;
  const recap = data.settings.template === "recap";
  const show = !!c && (recap || reelQueue || shape === "portrait");
  canvas.hidden = !show;
  player.style.visibility = show ? "hidden" : "";
  if (!show) { lastFrame = ""; return; }
  const src = srcById(c.source_id);
  const board = src.scoreboard || {};
  const text = [[board.home, board.away].filter(Boolean).join("  /  "), board.score].filter(Boolean).join("    ");
  const card = reelQueue?.intro ? "title" : reelQueue?.outro ? "end" : "";
  const title = card === "title" ? data.settings.title : card === "end" ? data.settings.end_card : "";
  const caption = recap && !title ? c.caption || "" : "";
  if (!title && player.readyState < 2 && lastFrame) return; // Keep the last picture while scrubbing.
  const key = [player.currentTime, player.readyState, player.videoWidth, c.id, c.crop, shape, recap, text, card, title, caption].join("|");
  if (key === lastFrame) return;
  lastFrame = key;
  const [ow, oh] = layouts[shape];
  const w = Math.round(ow / 2), h = Math.round(oh / 2);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#101e1a";
  ctx.fillRect(0, 0, w, h);
  if (!title && player.readyState >= 2 && player.videoWidth && loaded.sourceId === c.source_id) {
    const vw = player.videoWidth, vh = player.videoHeight;
    const cover = Math.max(w / vw, h / vh), fit = Math.min(w / vw, h / vh);
    if (shape === "portrait" && c.crop != null) {
      ctx.drawImage(player, (w - vw * cover) * c.crop, (h - vh * cover) / 2, vw * cover, vh * cover);
    } else {
      if (Math.abs(cover - fit) > .001) {
        ctx.filter = 'blur(12px) brightness(.9)';
        ctx.drawImage(player, (w - vw * cover) / 2 - 12, (h - vh * cover) / 2 - 12, vw * cover + 24, vh * cover + 24);
        ctx.filter = 'none';
      }
      ctx.drawImage(player, (w - vw * fit) / 2, (h - vh * fit) / 2, vw * fit, vh * fit);
    }
  }
  if (title || (recap && text)) {
    const img = layer(card || "board", shape, title || text);
    if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, w, h);
  }
  const into = player.currentTime - c.start, len = clipLen(c);
  if (caption && into >= 0) {
    const [a, b] = captionWindow(len);
    const alpha = Math.max(0, Math.min(1, (into - a) / CAPTION_EASE, 1 - (into - b) / CAPTION_EASE));
    const img = layer("caption", shape, caption);
    if (alpha > 0 && img.complete && img.naturalWidth) {
      ctx.globalAlpha = alpha;
      ctx.drawImage(img, -w * 0.04 * Math.max(0, 1 - Math.max(0, into - a) / CAPTION_EASE), 0, w, h);
      ctx.globalAlpha = 1;
    }
  }
  if (recap && reelQueue && !title) { // dip to black at the joins, as exported
    const dark = 1 - Math.min(1, into / FADE, (len - into) / FADE);
    if (dark > 0) { ctx.fillStyle = `rgba(0,0,0,${dark})`; ctx.fillRect(0, 0, w, h); }
  }
}

function captionWindow(len) {
  const a = Math.min(CAPTION_IN, len * 0.2);
  return [a, Math.max(a + CAPTION_EASE, Math.min(len - CAPTION_EASE - 0.2, a + CAPTION_EASE + CAPTION_HOLD))];
}

function openRecap() {
  $("recap-title").value = data.settings.title || "";
  $("recap-end").value = data.settings.end_card || "";
  $("recap-captions").innerHTML = reel().length ? reel().map((c, i) => {
    const s = srcById(c.source_id);
    return `<label class="cap-row" data-clip="${c.id}"><img src="${frameUrl(s.id, Math.min(c.anchor ?? c.start + clipLen(c) / 2, s.info.duration - 0.1), 160)}" alt="Clip ${i + 1}">
      <input maxlength="60" value="${esc(c.caption || "")}" placeholder="Clip ${i + 1} · ${fmt(c.start, false)}" aria-label="Caption for clip ${i + 1}"></label>`;
  }).join("") : `<p class="muted">Keep some clips to caption them.</p>`;
  const pick = $("writer-provider");
  const was = pick.value;
  pick.innerHTML = (data.writers || []).map(w => `<option value="${w.id}">${esc(w.label)}</option>`).join("");
  if (was && pick.querySelector(`[value="${was}"]`)) pick.value = was;
  writerChanged();
  $("recap-sources").innerHTML = data.sources.map(s => {
    const b = s.scoreboard || {};
    return `<fieldset data-sid="${s.id}"><legend>${esc(s.name)}</legend>${[
      ["home", "Team one", 28], ["away", "Team two", 28], ["score", "Score (optional)", 12]
    ].map(([k, label, limit]) => `<label>${label}<input name="${k}" maxlength="${limit}" value="${esc(b[k] || "")}"></label>`).join("")}</fieldset>`;
  }).join("");
  $("recap-dialog").showModal();
}

// Drafting the words. "tags and detection" needs nothing; a provider gets only the text
// facts (teams, score, minutes, tags, detector notes) and a key that stays on this machine.
function writerChanged() {
  const w = (data.writers || []).find(x => x.id === $("writer-provider").value) || { id: "detection" };
  const ai = w.id !== "detection";
  $("writer-key-row").hidden = $("writer-model-row").hidden = !ai;
  $("writer-model").placeholder = w.model || "";
  $("writer-key").placeholder = w.ready ? "Key already set · paste to replace" : "Paste a key";
  $("writer-note").textContent = ai
    ? `Sends only text to ${w.label}: team names, score, match minutes, your tags and the detector's notes. Never footage or file names. ` +
      `The key stays in this app's memory until you quit${w.env ? `, or set ${w.env} before starting it` : ""}.`
    : "Fills the lines from your tags (GOAL, Big save…), the detector's reasons and match minutes. Nothing leaves this machine.";
}

async function writeRecap() {
  const btn = $("writer-btn");
  btn.disabled = true;
  btn.textContent = "Writing…";
  try {
    const out = await api("POST", "/api/reel/draft", {
      provider: $("writer-provider").value, key: $("writer-key").value, model: $("writer-model").value });
    $("writer-key").value = "";
    await refresh(true); // the key now counts as set
    writerChanged();
    if (out.title) $("recap-title").value = out.title;
    if (out.end) $("recap-end").value = out.end;
    for (const row of $("recap-captions").querySelectorAll("[data-clip]")) {
      if (row.dataset.clip in out.captions) row.querySelector("input").value = out.captions[row.dataset.clip];
    }
    toast("Drafted. Edit anything, then Save recap.");
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
  btn.textContent = "Write";
}
