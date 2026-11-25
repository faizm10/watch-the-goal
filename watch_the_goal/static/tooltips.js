(() => {
  const tip = document.createElement("div");
  tip.id = "control-help";
  tip.className = "control-help";
  tip.role = "tooltip";
  tip.hidden = true;

  let anchor, described, timer, leaveTimer;

  function hide() {
    clearTimeout(timer);
    clearTimeout(leaveTimer);
    described?.removeAttribute("aria-describedby");
    anchor = described = null;
    tip.hidden = true;
  }

  function show(control, focusTarget) {
    hide();
    anchor = control;
    tip.textContent = control.dataset.help;
    if (focusTarget && !focusTarget.hasAttribute("aria-describedby")) {
      described = focusTarget;
      described.setAttribute("aria-describedby", tip.id);
    }
    (control.closest("dialog") || document.body).append(tip);
    timer = setTimeout(() => {
      if (!control.isConnected || !control.getClientRects().length) return hide();
      tip.hidden = false;
      const rect = control.getBoundingClientRect();
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      tip.style.left = `${Math.max(8, Math.min(rect.left + (rect.width - width) / 2, innerWidth - width - 8))}px`;
      tip.style.top = `${rect.bottom + height + 16 <= innerHeight
        ? rect.bottom + 8 : Math.max(8, rect.top - height - 8)}px`;
    }, focusTarget ? 0 : 350);
  }

  document.addEventListener("pointerover", (event) => {
    const control = event.target.closest("[data-help]");
    if (event.pointerType !== "mouse") return;
    if (control === anchor || tip.contains(event.target)) clearTimeout(leaveTimer);
    else if (control) show(control);
  });
  document.addEventListener("pointerout", (event) => {
    if (!anchor || described || event.pointerType !== "mouse") return;
    const inside = (node) => anchor.contains(node) || tip.contains(node);
    if (inside(event.target) && !inside(event.relatedTarget)) {
      clearTimeout(leaveTimer);
      leaveTimer = setTimeout(hide, tip.hidden ? 0 : 150);
    }
  });
  document.addEventListener("focusin", (event) => {
    const control = event.target.closest("[data-help]");
    if (control && event.target.matches(":focus-visible")) show(control, event.target);
  });
  document.addEventListener("focusout", (event) => {
    if (described === event.target) hide();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hide();
  });
  window.addEventListener("scroll", hide, true);
  window.addEventListener("resize", hide);
})();
