// The editor and reel preview share this canvas. Artwork is the exact PNG used by FFmpeg.
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
  const title = reelQueue?.intro ? data.settings.title : "";
  if (!title && player.readyState < 2 && lastFrame) return; // Keep the last picture while scrubbing.
  const key = [player.currentTime, player.readyState, player.videoWidth, c.id, c.crop, shape, recap, text, title].join("|");
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
    const img = layer(title ? "title" : "board", shape, title || text);
    if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, w, h);
  }
}

function openRecap() {
  $("recap-title").value = data.settings.title || "";
  $("recap-sources").innerHTML = data.sources.map(s => {
    const b = s.scoreboard || {};
    return `<fieldset data-sid="${s.id}"><legend>${esc(s.name)}</legend>${[
      ["home", "Team one", 28], ["away", "Team two", 28], ["score", "Score (optional)", 12]
    ].map(([k, label, limit]) => `<label>${label}<input name="${k}" maxlength="${limit}" value="${esc(b[k] || "")}"></label>`).join("")}</fieldset>`;
  }).join("");
  $("recap-dialog").showModal();
}
