/* Hash router and panel host. Plugin scripts load from this app only. */

const shell = { payload: null, jobTimer: null, route: "" };

function shellEscape(value) {
  if (typeof escapeHtml === "function") return escapeHtml(value);
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function showHost(html) {
  const evaluate = document.getElementById("view-evaluate");
  const view = document.getElementById("view-panel");
  if (evaluate) evaluate.classList.add("hidden");
  if (!view) return;
  view.classList.remove("hidden");
  view.innerHTML = html;
}

function setCurrent(id) {
  document.querySelectorAll("#side-nav .nav-item").forEach((button) => {
    if (button.dataset.slot === id) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
}

function showEvaluateView(tab) {
  const names = ["connect", "run", "results", "compare", "judges"];
  const name = names.indexOf(tab) >= 0 ? tab : "connect";
  const evaluate = document.getElementById("view-evaluate");
  const view = document.getElementById("view-panel");
  if (evaluate) evaluate.classList.remove("hidden");
  if (view) {
    view.classList.add("hidden");
    view.innerHTML = "";
  }
  if (typeof show === "function") show(name);
  setCurrent("evaluate");
}

function navSlot(id) {
  const nav = (shell.payload && shell.payload.nav) || [];
  return nav.find((slot) => slot.id === id) || null;
}

function findPanel(name) {
  const catalog = (shell.payload && shell.payload.catalog) || [];
  return catalog.find((row) => row.name === name) || null;
}

function hostStatus() {
  return (shell.payload && shell.payload.status) || {};
}

function badgeKind(availability) {
  if (availability === "ready") return "badge-ready";
  if (availability === "partial") return "badge-partial";
  return "badge-missing";
}

function othersTable(currentId) {
  const rows = (hostStatus().features || []).filter((slot) => slot.id !== currentId);
  const body = rows.map((slot) => (
    "<tr><td>" + shellEscape(slot.title) + "</td><td><span class=\"badge " + badgeKind(slot.availability) + "\">" +
    shellEscape(slot.badge || "") + "</span></td><td>" + shellEscape(slot.detail || "") + "</td></tr>"
  )).join("");
  return (
    '<section class="others"><h3>What else is available</h3>' +
    '<div class="table-wrap"><table><thead><tr><th scope="col">Feature</th><th scope="col">State</th><th scope="col">Detail</th></tr></thead><tbody>' +
    body + "</tbody></table></div></section>"
  );
}

function checkedText() {
  const status = hostStatus();
  const iso = status.checked_at || (shell.payload && shell.payload.checked_at) || "";
  const when = iso && typeof formatDate === "function" ? formatDate(iso) : iso;
  const found = status.found_label || "none";
  return "Last checked " + when + ". Found: " + found + ".";
}

function bindCardActions(command) {
  const again = document.getElementById("check-again");
  const copy = document.getElementById("copy-command");
  if (again) again.addEventListener("click", () => { checkAgain().catch((err) => showFailure(err)); });
  if (copy) copy.addEventListener("click", () => { copyCommand(command, copy); });
}

function renderMissing(slot) {
  setCurrent(slot.id);
  const command = slot.command || "";
  const explainer = slot.local_note || "It runs on this computer only.";
  showHost(
    '<header class="page-head"><h2>' + shellEscape(slot.title) + "</h2>" +
    '<p class="lede">' + shellEscape(slot.summary || "") + "</p></header>" +
    '<section class="panel availability-card">' +
    '<p class="state-pill">Not installed</p>' +
    "<h3>" + shellEscape(slot.title) + " comes with the Air-Gap AI Toolkit</h3>" +
    "<p>" + shellEscape(explainer) + "</p>" +
    '<p class="install-label">Install from the local wheel folder</p>' +
    '<pre class="command"><code id="install-command">' + shellEscape(command) + "</code></pre>" +
    '<p class="hint">' + shellEscape((shell.payload && shell.payload.restart_note) || "") + "</p>" +
    '<div class="card-actions"><div class="actions">' +
    '<button type="button" class="primary" id="check-again">Check again</button>' +
    '<button type="button" id="copy-command">Copy command</button></div>' +
    '<p class="hint checked-line" id="checked-line">' + shellEscape(checkedText()) + "</p></div>" +
    "</section>" +
    othersTable(slot.id)
  );
  bindCardActions(command);
}

function renderProblem(panel, slot) {
  const slotId = (slot && slot.id) || panel.slot || "more";
  setCurrent(document.querySelector('#side-nav .nav-item[data-slot="' + slotId + '"]') ? slotId : "more");
  const command = (slot && slot.command) || "";
  const title = panel.title || panel.name || "Panel";
  let body = "";
  if (panel.availability === "incompatible") {
    const host = (shell.payload && shell.payload.api_version) || "0.2.0";
    body =
      "<h2>Update needed</h2>" +
      '<p class="lede">' + shellEscape(title) + " " + shellEscape(panel.version || "") +
      " uses plugin API " + shellEscape(panel.api_version || "unknown") +
      ". This app loads plugin API " + shellEscape(host) + ".</p>" +
      "<p>" + shellEscape(panel.reason || "") + "</p>";
  } else {
    body =
      '<h2>This panel did not load</h2>' +
      '<p class="lede">' + shellEscape(title) + " is installed, but this app could not open it.</p>" +
      "<details><summary>Detail</summary><p>" + shellEscape(panel.reason || "Unknown error") + "</p></details>";
  }
  const commandBlock = command
    ? '<pre class="command"><code id="install-command">' + shellEscape(command) + "</code></pre>"
    : "";
  showHost(
    '<section class="panel availability-card" role="alert">' + body + commandBlock +
    '<div class="card-actions"><div class="actions"><button type="button" class="primary" id="check-again">Check again</button>' +
    (command ? '<button type="button" id="copy-command">Copy command</button>' : "") +
    '</div><p class="hint checked-line">' + shellEscape(checkedText()) + "</p></div></section>" +
    othersTable(slotId)
  );
  bindCardActions(command);
}

function ensureCss(href) {
  if (document.querySelector('link[data-plugin-css="' + href + '"]')) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  link.dataset.pluginCss = href;
  document.head.appendChild(link);
}

function checksBanner(panel) {
  const checks = (panel.status && panel.status.checks) || [];
  const failed = checks.filter((row) => !row.ok);
  if (!failed.length) return "";
  const items = failed.map((row) => (
    "<li><strong>" + shellEscape(row.label || row.id) + ".</strong> " +
    shellEscape(row.detail || "") + (row.fix ? " " + shellEscape(row.fix) : "") + "</li>"
  )).join("");
  return '<div class="banner" role="status"><ul class="check-list">' + items + "</ul></div>";
}

async function mountPlugin(panel) {
  const slotId = panel.slot || "more";
  setCurrent(document.querySelector('#side-nav .nav-item[data-slot="' + slotId + '"]') ? slotId : "more");
  const lede = panel.summary || "Runs on this computer only.";
  showHost(
    '<header class="page-head"><h2>' + shellEscape(panel.title || panel.name || "Panel") + "</h2>" +
    '<p class="lede">' + shellEscape(lede) + "</p></header>" +
    checksBanner(panel) +
    '<div class="panel plugin-card"><div id="plugin-mount"></div></div>'
  );
  const mount = document.getElementById("plugin-mount");
  if (!mount) return;
  if (panel.assets && panel.assets.css) ensureCss(panel.assets.css);
  if (panel.assets && panel.assets.js) {
    try {
      const mod = await import(panel.assets.js);
      if (mod && typeof mod.mount === "function") {
        await mod.mount(mount, {
          name: panel.name,
          title: panel.title,
          apiPrefix: panel.mounted_at || panel.api_prefix || "",
          status: panel.status,
        });
        return;
      }
      mount.textContent = "This panel has no script to show.";
      return;
    } catch (err) {
      mount.innerHTML = '<p role="alert">' +
        shellEscape(err && err.message ? err.message : "The panel script did not run.") + "</p>";
      return;
    }
  }
  mount.textContent = "This panel has no script to show.";
}

async function fillJobs(runs) {
  const table = document.getElementById("jobs-table");
  if (!table) return;
  let rows = runs;
  if (!rows) {
    const body = await api("/api/runs");
    rows = body.runs || [];
  }
  if (!rows.length) {
    table.innerHTML = '<p class="empty">No jobs yet. Start an evaluation from Evaluate.</p>';
    return;
  }
  const body = rows.map((run) => (
    "<tr><td>" + shellEscape(run.preset || run.run_id) + "</td><td>Evaluate</td><td>" +
    shellEscape(run.status || "") + "</td><td>" +
    shellEscape(typeof formatDate === "function" ? formatDate(run.created_at) : (run.created_at || "")) +
    "</td></tr>"
  )).join("");
  table.innerHTML =
    '<div class="table-wrap"><table><thead><tr><th scope="col">Job</th><th scope="col">From</th>' +
    '<th scope="col">Status</th><th scope="col">Started</th></tr></thead><tbody>' + body + "</tbody></table></div>";
}

async function renderJobs() {
  setCurrent("jobs");
  const slot = navSlot("jobs");
  showHost(
    '<header class="page-head"><h2>Jobs</h2><p class="lede">' +
    shellEscape(slot && slot.summary ? slot.summary : "Evaluation runs on this computer.") +
    '</p></header><div id="jobs-table" class="panel"><p class="hint">Loading jobs.</p></div>'
  );
  try {
    await fillJobs();
  } catch (err) {
    const table = document.getElementById("jobs-table");
    if (table) table.innerHTML = '<p class="banner banner-fail">' + shellEscape(err.message) + "</p>";
  }
}

function renderSettings() {
  setCurrent("settings");
  const slot = navSlot("settings");
  const banner = (shell.payload && shell.payload.classification) || {};
  const root = (shell.payload && shell.payload.toolkit_root) || "/opt/ai-toolkit";
  const offline = typeof state !== "undefined" && state.offline && state.offline.offline;
  const forced = typeof state !== "undefined" && state.offline && state.offline.forced;
  const version = (shell.payload && shell.payload.app_version) || "";
  const colors = ["green", "blue", "yellow", "orange", "red", "purple"];
  const options = colors.map((color) => (
    '<option value="' + color + '"' + (banner.color === color ? " selected" : "") + ">" + color + "</option>"
  )).join("");
  showHost(
    '<header class="page-head"><h2>Settings</h2><p class="lede">' +
    shellEscape(slot && slot.summary ? slot.summary : "Paths and offline mode.") +
    '</p></header><div class="panel form-narrow"><div class="field"><label for="settings-toolkit">Toolkit folder</label>' +
    '<input id="settings-toolkit" spellcheck="false" value="' + shellEscape(root) + '">' +
    '<p class="hint">Local wheel folder used by the install command. Default is /opt/ai-toolkit.</p></div>' +
    '<label class="check" for="settings-offline"><input id="settings-offline" type="checkbox"' +
    (offline ? " checked" : "") + (forced ? " disabled" : "") +
    "> Offline mode. Blocks the OpenAI cloud API and refuses weight downloads.</label>" +
    '<div class="field"><label for="settings-class-text">Classification banner</label>' +
    '<input id="settings-class-text" maxlength="80" value="' + shellEscape(banner.text || "") + '"></div>' +
    '<label class="check" for="settings-class-enabled"><input id="settings-class-enabled" type="checkbox"' +
    (banner.enabled ? " checked" : "") + "> Show the banner at the top and bottom of the page.</label>" +
    '<div class="field"><label for="settings-class-color">Banner color</label><select id="settings-class-color">' +
    options + "</select></div>" +
    '<div class="actions"><button type="button" class="primary" id="settings-save">Save settings</button></div>' +
    '<p id="settings-note" class="hint"></p></div>' +
    '<div class="panel"><h3>About</h3><p id="settings-about">LLM Eval Suite ' + shellEscape(version) +
    '. Runs on this computer. This app does not check for updates. Docs ship with the install.</p>' +
    '<p id="llama-attribution">Built with Llama.</p></div>'
  );
}

async function saveSettings() {
  const note = document.getElementById("settings-note");
  const saved = await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      toolkit_root: document.getElementById("settings-toolkit").value,
      classification: {
        enabled: document.getElementById("settings-class-enabled").checked,
        text: document.getElementById("settings-class-text").value,
        color: document.getElementById("settings-class-color").value,
      },
    }),
  });
  const offlineBox = document.getElementById("settings-offline");
  if (offlineBox && !offlineBox.disabled && typeof state !== "undefined") {
    const want = offlineBox.checked;
    const have = !!(state.offline && state.offline.offline);
    if (want !== have) {
      state.offline = await api("/api/offline", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ offline: want }),
      });
      if (typeof applyOfflineStatus === "function") applyOfflineStatus();
    }
  }
  if (shell.payload) {
    shell.payload.toolkit_root = saved.toolkit_root;
    shell.payload.classification = saved.classification;
    shell.payload = await api("/api/plugins");
    paintNav();
  }
  if (note) note.textContent = "Saved.";
}

function openHash(hash) {
  const catalog = (shell.payload && shell.payload.catalog) || [];
  const match = catalog.find((row) => row.route === hash);
  if (match && match.compatible) {
    mountPlugin(match);
    return;
  }
  if (match && !match.compatible) {
    renderProblem(match, navSlot(match.slot) || navSlot("more"));
    return;
  }
  const id = (hash.replace(/^#\//, "").split("/")[0]) || "evaluate";
  if (id === "evaluate") {
    showEvaluateView("connect");
    return;
  }
  const slot = navSlot(id);
  if (!slot) {
    showEvaluateView("connect");
    return;
  }
  if (id === "jobs") {
    renderJobs();
    return;
  }
  if (id === "settings") {
    renderSettings();
    return;
  }
  const panels = (slot.panels || []).map(findPanel).filter(Boolean);
  const ready = panels.filter((row) => row.compatible);
  if (ready.length) {
    mountPlugin(ready[0]);
    return;
  }
  if (panels.length) {
    renderProblem(panels[0], slot);
    return;
  }
  renderMissing(slot);
}

function applyRoute(force) {
  const hash = location.hash || "#/evaluate";
  const evaluate = hash === "#/evaluate" || hash.indexOf("#/evaluate/") === 0;
  const view = document.getElementById("view-panel");
  if (!force && shell.route === hash && !evaluate && view && !view.classList.contains("hidden")) return;
  if (evaluate) {
    shell.route = hash;
    showEvaluateView(hash.split("/")[2] || "connect");
    return;
  }
  if (!shell.payload) {
    showHost('<section class="panel"><p class="hint">Checking panels on this computer.</p></section>');
    const id = hash.replace(/^#\//, "").split("/")[0];
    if (id) setCurrent(id);
    return;
  }
  shell.route = hash;
  openHash(hash);
}

function paintEnvironment() {
  const line = document.getElementById("env-status");
  const text = hostStatus().environment || (shell.payload && shell.payload.environment);
  if (line && text) line.textContent = text;
}

function paintBanner() {
  const banner = (shell.payload && shell.payload.classification) || {};
  const on = !!(banner.enabled && banner.text);
  ["class-banner-top", "class-banner-bottom"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = banner.text || "";
    el.dataset.color = banner.color || "green";
    el.classList.toggle("is-on", on);
  });
}

function paintNav() {
  const nav = (shell.payload && shell.payload.nav) || [];
  const divider = document.querySelector("#side-nav .nav-divider");
  document.querySelectorAll('#side-nav .nav-item[data-slot="more"]').forEach((button) => button.remove());
  nav.forEach((slot) => {
    let button = document.querySelector('#side-nav .nav-item[data-slot="' + slot.id + '"]');
    if (!button && slot.id === "more" && divider) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "nav-item";
      button.dataset.slot = "more";
      button.innerHTML =
        '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="5" cy="10" r="1.2"/><circle cx="10" cy="10" r="1.2"/><circle cx="15" cy="10" r="1.2"/></svg>' +
        '<span class="nav-label"></span><span class="nav-tag hidden"></span><span class="status-dot hidden" aria-hidden="true"></span>';
      divider.before(button);
    }
    if (!button) return;
    button.dataset.route = slot.route || button.dataset.route;
    const label = button.querySelector(".nav-label");
    if (label) label.textContent = slot.title;
    const tag = button.querySelector(".nav-tag");
    if (tag) {
      tag.textContent = slot.tag || "";
      tag.classList.toggle("hidden", !slot.tag);
    }
    button.classList.toggle("is-muted", slot.availability === "not_installed");
    const dot = button.querySelector(".status-dot");
    const partial = slot.availability === "partial";
    if (dot) dot.classList.toggle("hidden", !partial);
    let note = label ? label.querySelector(".visually-hidden") : null;
    if (partial && label) {
      if (!note) {
        note = document.createElement("span");
        note.className = "visually-hidden";
        label.appendChild(note);
      }
      note.textContent = ", partly available";
    } else if (note) {
      note.remove();
    }
  });
  paintEnvironment();
  paintBanner();
}

async function checkAgain() {
  const button = document.getElementById("check-again");
  if (button) button.disabled = true;
  shell.payload = await api("/api/plugins");
  paintNav();
  applyRoute(true);
}

function copyCommand(command, button) {
  const done = () => {
    if (!button) return;
    const previous = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = previous; }, 1500);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(command).then(done).catch(() => fallbackCopy(command, done));
    return;
  }
  fallbackCopy(command, done);
}

function fallbackCopy(command, done) {
  const area = document.createElement("textarea");
  area.value = command;
  document.body.appendChild(area);
  area.select();
  try { document.execCommand("copy"); } catch (_err) { /* the command stays visible */ }
  area.remove();
  done();
}

function showFailure(err) {
  showHost(
    '<section class="panel" role="alert"><h2>Could not check panels</h2><p>' +
    shellEscape(err && err.message ? err.message : "The check failed.") + "</p></section>"
  );
}

async function refreshJobs() {
  try {
    const body = await api("/api/runs");
    const runs = body.runs || [];
    const active = runs.find((run) => run.status === "running" || run.status === "cancel_requested");
    const pill = document.getElementById("job-pill");
    if (pill) {
      if (!active) {
        pill.hidden = true;
        pill.textContent = "";
      } else {
        const label = (active.preset || "Evaluation") + (active.model ? " · " + active.model : "");
        pill.hidden = false;
        pill.setAttribute("role", "status");
        pill.innerHTML = '<span class="job-pill-label">' + shellEscape(label) +
          '</span><span class="job-pill-track" role="progressbar" aria-valuemin="0" aria-label="Running"><span class="job-pill-bar"></span></span>';
      }
    }
    if (document.getElementById("jobs-table")) await fillJobs(runs);
  } catch (_err) {
    /* the next poll retries */
  }
}

function scheduleJobs() {
  const delay = document.hidden ? 5000 : 1000;
  shell.jobTimer = window.setTimeout(async () => {
    await refreshJobs();
    scheduleJobs();
  }, delay);
}

document.getElementById("view-panel").addEventListener("click", (event) => {
  if (!event.target.closest("#settings-save")) return;
  saveSettings().catch((err) => {
    const note = document.getElementById("settings-note");
    if (note) note.textContent = err.message;
  });
});

document.getElementById("side-nav").addEventListener("click", (event) => {
  const button = event.target.closest(".nav-item");
  if (!button) return;
  const route = button.dataset.route || "#/evaluate";
  if (location.hash === route) applyRoute(true);
  else location.hash = route;
});

document.querySelectorAll(".subtabs .tab").forEach((button) => {
  button.addEventListener("click", () => {
    const tab = button.dataset.tab;
    if (!tab) return;
    const route = "#/evaluate/" + tab;
    if (location.hash !== route) history.pushState(null, "", route);
    const evaluate = document.getElementById("view-evaluate");
    const view = document.getElementById("view-panel");
    if (evaluate) evaluate.classList.remove("hidden");
    if (view) view.classList.add("hidden");
    setCurrent("evaluate");
  });
});

const subtabs = document.querySelector(".subtabs");
if (subtabs) {
  subtabs.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const tabs = Array.from(subtabs.querySelectorAll(".tab"));
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? 1 : -1;
    const next = tabs[(index + delta + tabs.length) % tabs.length];
    next.focus();
    next.click();
  });
}

window.addEventListener("hashchange", () => applyRoute());
window.addEventListener("popstate", () => applyRoute());

const offlineToggle = document.getElementById("offline-toggle");
if (offlineToggle) {
  offlineToggle.addEventListener("change", () => {
    window.setTimeout(() => {
      api("/api/plugins").then((payload) => {
        shell.payload = payload;
        paintNav();
      }).catch(() => { /* the status line stays until the next check */ });
    }, 200);
  });
}

async function bootShell() {
  applyRoute();
  try {
    shell.payload = await api("/api/plugins");
  } catch (err) {
    if (location.hash && location.hash.indexOf("#/evaluate") !== 0) showFailure(err);
    scheduleJobs();
    return;
  }
  paintNav();
  applyRoute();
  scheduleJobs();
}

bootShell();
