const state = {
  presets: [],
  connections: [],
  runs: [],
  judges: [],
  datasetId: "sample",
  activeRun: null,
  timer: null,
  passBarPercent: null,
};

function barPercent(card) {
  if (card && card.pass_bar_percent != null) state.passBarPercent = Number(card.pass_bar_percent);
  return state.passBarPercent;
}

const TABS = ["connect", "run", "results", "compare", "judges"];

const STATUS_TEXT = {
  running: "Running",
  cancel_requested: "Cancelling",
  completed: "Completed",
  cancelled: "Cancelled",
  failed: "Failed",
  invalid: "Invalid",
};

function $(id) { return document.getElementById(id); }

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try { body = JSON.parse(text); } catch (_err) { body = { detail: text }; }
  }
  if (!response.ok) {
    const detail = body && (body.detail || body.error) || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function show(tab) {
  for (const name of TABS) {
    $(name).classList.toggle("hidden", name !== tab);
  }
  document.querySelectorAll(".tab").forEach((button) => {
    if (button.dataset.tab === tab) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  window.scrollTo(0, 0);
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    show(button.dataset.tab);
    if (button.dataset.tab === "results" && $("results-run").value) {
      showResults().catch((err) => alert(err.message));
    }
  });
});

/* ---------- Formatting ---------- */

function formatSeconds(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "unknown";
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  if (hours > 0) return hours + "h " + minutes + "m";
  if (minutes <= 0) return rest + "s";
  return minutes + "m " + rest + "s";
}

function formatPercent(rate, digits) {
  if (rate == null || Number.isNaN(Number(rate))) return "n/a";
  return (Number(rate) * 100).toFixed(digits == null ? 1 : digits) + "%";
}

function formatDelta(delta, asPoints) {
  if (delta == null || Number.isNaN(Number(delta))) return { text: "n/a", cls: "delta-flat" };
  const value = Number(delta) * (asPoints ? 100 : 1);
  const digits = asPoints ? 1 : 2;
  if (Math.abs(value) < Math.pow(10, -digits) / 2) return { text: "0", cls: "delta-flat" };
  const sign = value > 0 ? "+" : "−";
  const text = sign + Math.abs(value).toFixed(digits) + (asPoints ? " pts" : "");
  return { text, cls: value > 0 ? "delta-up" : "delta-down" };
}

function formatScore(score) {
  if (score == null || score === "") return "";
  const value = Number(score);
  if (Number.isNaN(value)) return String(score);
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function statusBadge(status) {
  const text = STATUS_TEXT[status] || status || "Unknown";
  const kind = status === "completed" ? "badge-pass"
    : (status === "failed" || status === "invalid") ? "badge-fail"
      : (status === "cancelled" || status === "cancel_requested") ? "badge-fixture"
        : "badge-neutral";
  return '<span class="badge ' + kind + '">' + escapeHtml(text) + "</span>";
}

function sourceBadge(source, label) {
  const failed = source === "failed";
  const kind = source === "live" ? "badge-live" : failed ? "badge-fail" : "badge-fixture";
  const text = label || (source === "live" ? "Live" : failed ? "Failed live scan" : "Fixture or smoke, no live model call");
  return '<span class="badge ' + kind + '">' + escapeHtml(text) + "</span>";
}

function emptyNote(text) {
  return '<p class="empty">' + escapeHtml(text) + "</p>";
}

/* ---------- Connect ---------- */

function profileFromForm() {
  const context = $("conn-context").value.trim();
  return {
    name: $("conn-name").value.trim(),
    type: $("conn-type").value,
    base_url: $("conn-url").value.trim(),
    model: $("conn-model").value.trim(),
    api_key: $("conn-key").value,
    mode: $("conn-mode").value,
    max_context: context ? Number(context) : null,
    folder: $("conn-folder").value.trim(),
    precision: $("conn-precision") ? $("conn-precision").value : "",
    max_new_tokens: $("conn-max-new") && $("conn-max-new").value ? Number($("conn-max-new").value) : null,
    thinking_max_tokens: $("conn-thinking") && $("conn-thinking").value ? Number($("conn-thinking").value) : null,
    trust_remote_code: Boolean($("conn-trust") && $("conn-trust").checked),
    use_chat_template: ($("conn-chat-template") && $("conn-chat-template").value) || "auto",
  };
}

function updateLocalWeights() {
  const type = $("conn-type").value;
  $("local-weights").classList.toggle("hidden", type !== "hf" && type !== "nanogpt");
}

function updateCloudBanner() {
  const profile = profileFromForm();
  const local = !profile.base_url || /localhost|127\.0\.0\.1|\[::1\]/.test(profile.base_url);
  const banner = $("cloud-banner");
  const hide = local || profile.type === "hf" || profile.type === "nanogpt";
  const blocked = state.offline && state.offline.openai_cloud_blocked;
  banner.textContent = !hide && blocked
    ? "Offline mode blocks the OpenAI cloud API. This endpoint will be refused."
    : "This endpoint is not on this computer. Prompts will leave the machine.";
  banner.classList.toggle("hidden", hide);
  updateLocalWeights();
}

function applyOfflineStatus() {
  const status = state.offline || {};
  const banner = $("offline-banner");
  const toggle = $("offline-toggle");
  const note = $("rail-offline-note");
  if (toggle) {
    toggle.checked = !!status.offline;
    toggle.disabled = !!status.forced;
  }
  if (status.offline) {
    banner.textContent = status.message || "Offline mode is on. The OpenAI cloud API is blocked.";
    banner.classList.remove("hidden");
    if (note) note.textContent = "Offline mode is on. The OpenAI cloud API is blocked.";
  } else if (banner) {
    banner.textContent = "";
    banner.classList.add("hidden");
    if (note) note.textContent = "Runs on this computer. Nothing leaves it unless you choose a cloud endpoint.";
  }
  updateCloudBanner();
}

async function loadOffline() {
  state.offline = await api("/api/offline");
  applyOfflineStatus();
}

["conn-url", "conn-type"].forEach((id) => $(id).addEventListener("input", updateCloudBanner));
$("conn-type").addEventListener("change", updateCloudBanner);
$("offline-toggle").addEventListener("change", async () => {
  try {
    state.offline = await api("/api/offline", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ offline: $("offline-toggle").checked }),
    });
    applyOfflineStatus();
  } catch (err) {
    alert(err.message);
    await loadOffline();
  }
});

$("list-models").addEventListener("click", async () => {
  const button = $("list-models");
  button.disabled = true;
  try {
    const data = await api("/api/ollama/tags?base_url=" + encodeURIComponent($("conn-url").value.trim() || "http://127.0.0.1:11434"));
    const select = $("ollama-models");
    select.innerHTML = "";
    if (!data.models.length) {
      select.innerHTML = '<option value="">No models installed</option>';
    }
    for (const name of data.models) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (data.models[0]) $("conn-model").value = data.models[0];
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
});

$("ollama-models").addEventListener("change", () => {
  if ($("ollama-models").value) $("conn-model").value = $("ollama-models").value;
});

$("detect-path").addEventListener("click", async () => {
  try {
    const data = await api("/api/connections/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("conn-folder").value.trim() }),
    });
    $("detect-note").textContent = data.message;
  } catch (err) {
    $("detect-note").textContent = err.message;
  }
});

$("convert-nanogpt").addEventListener("click", async () => {
  $("detect-note").textContent = "Converting. This checks logits against the source checkpoint and can take a minute.";
  try {
    const data = await api("/api/connections/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ckpt_path: $("conn-folder").value.trim(),
        out_dir: $("convert-out").value.trim(),
      }),
    });
    $("detect-note").textContent = "Converted. Connect the Hugging Face folder at " + data.folder + ". Copy a local GPT-2 tokenizer into that folder before loading it.";
    $("conn-type").value = "hf";
    $("conn-folder").value = data.folder;
    updateLocalWeights();
  } catch (err) {
    $("detect-note").textContent = err.message;
  }
});

$("test-connection").addEventListener("click", async () => {
  const box = $("test-result");
  $("test-empty").classList.add("hidden");
  box.classList.remove("hidden");
  box.textContent = "Testing the connection...";
  try {
    const data = await api("/api/connections/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: profileFromForm() }),
    });
    if (!data.ok) {
      box.textContent = data.error || "Connection failed.";
      return;
    }
    box.textContent = [
      "Reply: " + data.reply,
      "Latency: " + data.latency_ms + " ms",
      "Tokens/s: " + (data.tokens_per_second == null ? "n/a" : data.tokens_per_second),
      "Tokens: " + (data.tokens_used == null ? "n/a" : data.tokens_used),
    ].join("\n");
  } catch (err) {
    box.textContent = err.message;
  }
});

$("save-connection").addEventListener("click", async () => {
  try {
    await api("/api/connections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profileFromForm()),
    });
    await loadConnections();
  } catch (err) {
    alert(err.message);
  }
});

const TYPE_TEXT = {
  ollama: "Ollama",
  openai: "OpenAI-compatible",
  hf: "Hugging Face folder",
  nanogpt: "nanoGPT checkpoint",
};

async function loadConnections() {
  const data = await api("/api/connections");
  state.connections = data.connections;
  const list = $("connection-list");
  const runSelect = $("run-connection");
  const previous = runSelect.value;
  runSelect.innerHTML = "";
  if (!state.connections.length) {
    list.innerHTML = emptyNote("No saved connections yet. Test a model, then save it to use it on the Run screen.");
    runSelect.innerHTML = '<option value="">Save a connection first</option>';
    return;
  }
  list.innerHTML = state.connections.map((row) => {
    const where = row.cloud
      ? '<span class="badge badge-fixture">Cloud</span>'
      : '<span class="badge badge-local">Local</span>';
    const target = row.model || row.folder || "";
    return '<div class="list-row"><div>' +
      '<div class="list-title">' + escapeHtml(row.name) + "</div>" +
      '<div class="list-sub">' + escapeHtml(TYPE_TEXT[row.type] || row.type) + (target ? ", " + escapeHtml(target) : "") + "</div>" +
      "</div>" + where + "</div>";
  }).join("");
  for (const row of state.connections) {
    const option = document.createElement("option");
    option.value = row.id;
    option.textContent = row.name + " (" + (row.model || row.type) + ")";
    runSelect.appendChild(option);
  }
  if (previous) runSelect.value = previous;
}

/* ---------- Run ---------- */

async function loadPresets() {
  const dataset = state.datasetId || "sample";
  const connection = $("run-connection").value || "";
  let url = "/api/presets?dataset_id=" + encodeURIComponent(dataset);
  if (connection) url += "&connection_id=" + encodeURIComponent(connection);
  const data = await api(url);
  state.presets = data.presets;
  const select = $("run-preset");
  select.innerHTML = "";
  for (const preset of state.presets) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = preset.label;
    select.appendChild(option);
  }
  showPreset();
}

function showPreset() {
  const preset = state.presets.find((row) => row.id === $("run-preset").value);
  if (!preset) return;
  $("preset-note").textContent = preset.description;
  const warning = $("preset-warning");
  warning.textContent = preset.warning || "";
  warning.classList.toggle("hidden", !preset.warning);
  const probes = preset.probe_count == null ? "Unspecified" : preset.probe_count;
  const prompts = preset.prompt_count == null ? "Unspecified" : preset.prompt_count;
  const facts = preset.factcheck_count == null ? 0 : preset.factcheck_count;
  let estimate = "Estimating…";
  let source = "No measured speed for this model, endpoint, and suite yet.";
  if (preset.estimate_source === "unbounded") {
    estimate = "Not estimated";
    source = "This preset has no prompt cap, so the time is not estimated.";
  } else if (preset.estimated_seconds != null && preset.estimate_source !== "estimating") {
    estimate = formatSeconds(preset.estimated_seconds);
    source = preset.estimate_source === "measured"
      ? "Estimate uses measured speed for this model and suite: " + preset.seconds_per_prompt + " s per prompt."
      : "Estimate assumes " + preset.seconds_per_prompt + " s per prompt until a run of this model is measured.";
  }
  const cell = (term, value) => "<div><dt>" + escapeHtml(term) + "</dt><dd>" + escapeHtml(value) + "</dd></div>";
  $("preset-estimate").innerHTML = '<dl class="estimate-grid">' +
    cell("Garak probes", probes) +
    cell("Prompt cap", prompts) +
    cell("Fact checks", facts) +
    cell("Estimated time", estimate) +
    '</dl><p class="estimate-source">' + escapeHtml(source) + "</p>";
}

$("run-preset").addEventListener("change", showPreset);
$("run-connection").addEventListener("change", () => {
  loadPresets().catch((err) => progressMessage(err.message));
});

$("run-dataset").addEventListener("change", async () => {
  state.datasetId = $("run-dataset").value;
  await loadPresets();
});

function progressMessage(text) {
  $("progress").innerHTML = '<p class="progress-message">' + escapeHtml(text) + "</p>";
}

$("start-demo").addEventListener("click", async () => {
  try {
    const data = await api("/api/demo/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder: $("demo-folder").value, preset: "quick" }),
    });
    state.demoRuns = (data.runs || []).map((row) => row.run_id);
    state.activeRun = state.demoRuns[0] || null;
    $("cancel-run").disabled = !state.activeRun;
    progressMessage("Demo pair started on the " + data.preset + " preset. Compare opens when both runs finish.");
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(pollDemo, 1000);
  } catch (err) {
    progressMessage(err.message);
  }
});

async function pollDemo() {
  if (!state.demoRuns || !state.demoRuns.length) return;
  try {
    const runs = [];
    for (const runId of state.demoRuns) runs.push(await api("/api/runs/" + runId));
    renderProgress(runs[0]);
    const pending = runs.some((run) => run.status === "running" || run.status === "cancel_requested");
    if (pending) return;
    clearInterval(state.timer);
    state.timer = null;
    $("cancel-run").disabled = true;
    await loadRuns();
    if (runs.length >= 2) {
      $("compare-left").value = runs[0].run_id;
      $("compare-right").value = runs[1].run_id;
      show("compare");
      $("do-compare").click();
    }
  } catch (err) {
    progressMessage(err.message);
  }
}

$("dataset-file").addEventListener("change", async () => {
  const file = $("dataset-file").files[0];
  if (!file) return;
  const text = await file.text();
  try {
    const data = await api("/api/datasets/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, text }),
    });
    if (!data.ok) {
      $("dataset-note").textContent = (data.errors || []).join(" ");
      return;
    }
    state.datasetId = data.dataset_id;
    const option = document.createElement("option");
    option.value = data.dataset_id;
    option.textContent = file.name + " (" + data.row_count + " rows)";
    $("run-dataset").appendChild(option);
    $("run-dataset").value = data.dataset_id;
    $("dataset-note").textContent = "Uploaded " + data.row_count + " rows.";
  } catch (err) {
    $("dataset-note").textContent = err.message;
  }
});

function renderProgress(run) {
  const suites = (run.progress && run.progress.suites) || [];
  let html = '<div class="progress-status">' + statusBadge(run.status) +
    '<span class="muted">' + escapeHtml(run.run_id || "") + "</span></div>";
  if (run.validity === "invalid") {
    html += '<div class="banner banner-fail">Invalid run. ' + escapeHtml(run.validity_reason || "Too many empty generations.") + "</div>";
  }
    const running = run.status === "running" || run.status === "cancel_requested";
  for (const suite of suites) {
    const done = suite.done == null ? 0 : Number(suite.done);
    const total = suite.total == null ? null : Number(suite.total);
    const pct = total ? Math.min(100, Math.round((done / total) * 100)) : null;
    const suiteDone = suite.status === "completed" || suite.status === "cancelled";
    const indeterminate = pct == null && running && !suiteDone;
    const finished = suiteDone || (!running && pct == null);
    const width = pct == null ? (finished ? 100 : 0) : pct;
    const raw = suite.status || suite.source || "";
    let eta = raw ? raw.charAt(0).toUpperCase() + raw.slice(1).replace(/_/g, " ") : "";
    const showEta = running && !suiteDone && suite.eta_seconds != null && Number(suite.eta_seconds) > 0;
    if (suite.status === "cancelled") {
      eta = "Stopped";
    } else if (suiteDone) {
      eta = suite.elapsed_seconds != null ? "Done in " + formatSeconds(suite.elapsed_seconds) : "Done";
    } else if (showEta) {
      const rate = suite.seconds_per_prompt == null ? "" : " at " + suite.seconds_per_prompt + " s per prompt";
      eta = "About " + formatSeconds(suite.eta_seconds) + " left" + rate;
    } else if (running && total != null && done >= total && total > 0) {
      eta = suite.elapsed_seconds != null ? "Done in " + formatSeconds(suite.elapsed_seconds) : "Done";
    } else if (running && !suiteDone) {
      eta = "Estimating…";
    }
    html += '<div class="progress-row">' +
      '<span class="name">' + escapeHtml(suite.name) + "</span>" +
      '<span class="count">' + done + " of " + (total == null ? "?" : total) + "</span>" +
      '<div class="progress-track"><div class="progress-fill' + (indeterminate ? " indeterminate" : "") +
      '" style="--v:' + width + '%"></div></div>' +
      (eta ? '<span class="eta">' + escapeHtml(eta) + "</span>" : "") +
      "</div>";
  }
  if (!suites.length && running) {
    html += '<p class="progress-message">Starting suites...</p>';
  }
  $("progress").innerHTML = html;
}

async function poll() {
  if (!state.activeRun) return;
  try {
    const run = await api("/api/runs/" + state.activeRun);
    renderProgress(run);
    if (run.status === "running" || run.status === "cancel_requested") return;
    clearInterval(state.timer);
    state.timer = null;
    $("cancel-run").disabled = true;
    await loadRuns();
    $("results-run").value = run.run_id;
    await showResults();
    show("results");
  } catch (err) {
    progressMessage(err.message);
  }
}

$("start-run").addEventListener("click", async () => {
  try {
    const run = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        connection_id: $("run-connection").value,
        preset: $("run-preset").value,
        dataset_id: $("run-dataset").value || "sample",
      }),
    });
    state.activeRun = run.run_id;
    $("cancel-run").disabled = false;
    renderProgress(run);
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(poll, 1000);
  } catch (err) {
    progressMessage(err.message);
  }
});

$("cancel-run").addEventListener("click", async () => {
  if (!state.activeRun) return;
  await api("/api/runs/" + state.activeRun + "/cancel", { method: "POST" });
});

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  if (data.pass_bar_percent != null) state.passBarPercent = Number(data.pass_bar_percent);
  for (const id of ["results-run", "compare-left", "compare-right"]) {
    const select = $(id);
    const previous = select.value;
    select.innerHTML = "";
    if (!state.runs.length) {
      select.innerHTML = '<option value="">No runs yet</option>';
      continue;
    }
    for (const run of state.runs) {
      const option = document.createElement("option");
      option.value = run.run_id;
      const bits = [run.model || "Unknown model"];
      if (run.overall_pass_percent != null) bits.push(run.overall_pass_percent + "%");
      bits.push(formatDate(run.created_at) || run.run_id);
      option.textContent = bits.join(", ") + " (" + (STATUS_TEXT[run.status] || run.status) + ")";
      option.title = run.run_id;
      select.appendChild(option);
    }
    if (previous) select.value = previous;
  }
}

/* ---------- Results dashboard ---------- */

function renderReadout(run, card) {
  const invalid = run.validity === "invalid" || run.status === "invalid";
  const verdict = invalid ? "Invalid run" : (card.verdict || "No live categories scored");
  let verdictClass = "none";
  if (invalid) verdictClass = "invalid";
  else if (card.meets_bar === true) verdictClass = "pass";
  else if (card.meets_bar === false) verdictClass = "fail";
  const detailBits = [];
  if (invalid) detailBits.push("The headline score is withheld");
  if (card.live_item_count != null) detailBits.push(card.live_item_count + " live prompts scored");
  if (card.failure_count != null) detailBits.push(card.failure_count + " failing");
  const detail = detailBits.length ? detailBits.join(". ") + "." : "";
  const connection = run.connection || {};
  const meta = [
    ["Model", connection.model || connection.name || "Unknown"],
    ["Preset", run.preset || "Custom"],
    ["Started", formatDate(run.created_at) || "Unknown"],
  ];
  let scoreHtml;
  if (invalid) {
    scoreHtml = '<div class="readout-score"><span class="withheld">Score withheld</span></div>';
  } else if (card.live_category_count) {
    const word = card.meets_bar ? "Pass" : "Below bar";
    scoreHtml = '<div class="readout-score is-empty"><span class="num">' + escapeHtml(word) + "</span></div>";
  } else {
    scoreHtml = '<div class="readout-score is-empty"><span class="num">Not measured</span></div>';
  }
  $("readout").innerHTML =
    scoreHtml +
    '<div><p class="verdict ' + verdictClass + '">' + escapeHtml(verdict) + "</p>" +
    '<p class="readout-detail">' + escapeHtml(detail) + "</p></div>" +
    '<dl class="run-meta">' +
    meta.map(([term, value]) => "<dt>" + escapeHtml(term) + '</dt><dd title="' + escapeHtml(value) + '">' + escapeHtml(value) + "</dd>").join("") +
    "<dt>Status</dt><dd>" + statusBadge(run.status) + "</dd></dl>";
}

function fixtureMeterText(percent) {
  if (percent == null || Number.isNaN(Number(percent))) return "Not measured";
  const value = Number(percent);
  const rounded = Math.round(value);
  const text = Math.abs(value - rounded) < 0.05 ? String(rounded) : value.toFixed(1);
  return text + "% fixture";
}

function garakMeterNote(run, row) {
  if (!run || (!run.garak_pass_rate_label && !run.garak_wording && !row)) return "";
  const invalid = run.validity === "invalid" || run.status === "invalid";
  const live = row && (row.status === "pass" || row.status === "fail") && row.pass_percent != null;
  if (invalid || !live) return "";
  const pass = Number(row.pass_percent).toFixed(1);
  const asr = (100 - Number(row.pass_percent)).toFixed(1);
  const label = run.garak_pass_rate_label || "Pass rate (1 - ASR)";
  const wording = run.garak_wording || "Pass rate is 1 minus garak's attack success rate (ASR).";
  return label + ": " + pass + "%. ASR: " + asr + "%. " + wording;
}

function meterRow(row, bar, note) {
  const status = row.status === "not_run" ? "not_run"
    : row.status === "withheld" ? "withheld"
    : row.source && row.source !== "live" && row.status === "fixture" ? "fixture"
      : row.status === "pass" ? "pass"
        : row.status === "fail" ? "fail"
          : "fixture";
  const pct = row.pass_percent == null ? 0 : Math.max(0, Math.min(100, Number(row.pass_percent)));
  let sub;
  if (status === "not_run") sub = "No prompts in this run";
  else {
    const count = row.sample_count == null ? "" : row.sample_count + " prompts, ";
    const source = row.source === "live" ? "live" : row.source === "mixed" ? "live and fixture" : "fixture or smoke";
    sub = count + source;
  }
  const value = status === "not_run"
    ? "Not run"
    : status === "withheld" ? "Score withheld"
    : status === "fixture" ? escapeHtml(fixtureMeterText(row.pass_percent))
      : row.pass_percent == null ? "Not measured" : Number(row.pass_percent).toFixed(1) + "<small>%</small>";
  const badge = status === "pass" ? '<span class="badge badge-pass">Pass</span>'
    : status === "fail" ? '<span class="badge badge-fail">Below bar</span>'
      : status === "fixture" ? '<span class="badge badge-fixture">Fixture</span>'
        : status === "withheld" ? '<span class="badge badge-neutral">Withheld</span>'
        : '<span class="badge badge-neutral">Not run</span>';
  const barText = bar == null || status === "withheld" ? "" : " Pass bar " + bar + "%.";
  const label = status === "not_run"
    ? row.label + ": not run"
    : status === "withheld"
      ? row.label + ": score withheld"
    : row.label + ": " + (status === "fixture"
      ? fixtureMeterText(row.pass_percent)
      : (row.pass_percent == null ? row.status : row.pass_percent + "% pass")) + ", " + sub + "." + barText;
  const barMark = bar == null || status === "withheld" ? "" : '<div class="meter-pass-bar" style="--bar:' + bar + '%"></div>';
  const noteHtml = note && status !== "withheld" ? '<span class="meter-note">' + escapeHtml(note) + "</span>" : "";
  const showFill = status !== "not_run" && status !== "withheld";
  return '<div class="meter-row" data-status="' + status + '">' +
    '<div><span class="meter-name">' + escapeHtml(row.label) + '</span><span class="meter-sub">' + escapeHtml(sub) + "</span>" +
    noteHtml + "</div>" +
    '<div class="meter" role="img" aria-label="' + escapeHtml(label) + '">' +
    (showFill ? '<div class="meter-fill" style="--v:' + pct + '%"></div>' : "") +
    barMark + "</div>" +
    '<div class="meter-value">' + value + "</div>" + badge + "</div>";
}

function tableHtml(headers, rows) {
  const head = "<thead><tr>" + headers.map((h) => "<th" + (h.num ? ' class="num"' : "") + ">" + escapeHtml(h.label) + "</th>").join("") + "</tr></thead>";
  const body = "<tbody>" + rows.map((cells) => "<tr>" + cells.join("") + "</tr>").join("") + "</tbody>";
  return "<table>" + head + body + "</table>";
}

function td(value, cls) {
  const text = value == null ? "" : String(value);
  if (cls === "clip") return '<td class="clip"><div title="' + escapeHtml(text.slice(0, 600)) + '">' + escapeHtml(text) + "</div></td>";
  return "<td" + (cls ? ' class="' + cls + '"' : "") + ">" + escapeHtml(text) + "</td>";
}

let resultsGeneration = 0;

async function showResults() {
  const runId = $("results-run").value;
  if (!runId) return;
  const generation = ++resultsGeneration;
  const run = await api("/api/runs/" + runId);
  const items = await api("/api/runs/" + runId + "/items");
  if (generation !== resultsGeneration || $("results-run").value !== runId) return;
  const banner = $("validity-banner");
  const invalid = run.validity === "invalid" || run.status === "invalid";
  banner.classList.toggle("hidden", !invalid);
  banner.textContent = invalid ? ("Invalid run. " + (run.validity_reason || "Too many empty generations.")) : "";

  const pathBits = [];
  if (run.garak_runs_dir) pathBits.push("Garak report folder: " + run.garak_runs_dir);
  if (run.log_path) pathBits.push("Run log: " + run.log_path);
  $("garak-path").innerHTML = pathBits.map((bit) => '<span class="path-line">' + escapeHtml(bit) + "</span>").join("");

  const card = run.scorecard || {};
  const bar = barPercent(card);
  renderReadout(run, card);
  const categories = (card.categories || []).map((row) => {
    if (!invalid || row.status === "not_run") return row;
    return {
      category: row.category,
      label: row.label,
      status: "withheld",
      pass_rate: null,
      pass_percent: null,
      sample_count: row.sample_count,
      source: row.source,
    };
  });
  $("scorecard").innerHTML = categories.map((row) => {
    const note = row.category === "security_jailbreak" ? garakMeterNote(run, row) : "";
    return meterRow(row, bar, note);
  }).join("");

  const suites = run.suites || [];
  $("suite-list").innerHTML = suites.length
    ? suites.map((suite) => '<div class="list-row suite-row"><div>' +
      '<div class="list-title">' + escapeHtml(suite.name) + " " + sourceBadge(suite.source, suite.label) + "</div>" +
      (suite.notes ? '<div class="list-sub">' + escapeHtml(suite.notes) + "</div>" : "") +
      "</div></div>").join("")
    : emptyNote("No suites recorded for this run.");

  const analysis = run.analysis || {};
  $("analysis-text").textContent = analysis.narrative || "No analysis yet. Write analysis asks the council judges, or uses the template summary when no local judge is available.";
  $("summary-source").textContent = analysis.source_label ? "Source: " + analysis.source_label : "";
  const cloud = (analysis.source_label || "").includes("cloud");
  $("cloud-judge-banner").classList.toggle("hidden", !cloud);

  const labels = {};
  for (const row of card.categories || []) labels[row.category] = row.label;
  const categoryLabel = (key) => labels[key] || key;
  const allItems = items.items || [];
  const liveRows = allItems.filter((item) => item.passed === false && item.source === "live" && item.counts_toward_score !== false);
  const fixtureRows = allItems.filter((item) => item.passed === false && item.source !== "live");
  const failureCount = card.failure_count == null ? liveRows.length : Number(card.failure_count);
  const failureHeaders = [{ label: "Category" }, { label: "Source" }, { label: "Prompt" }, { label: "Response" }, { label: "Expected" }, { label: "Score", num: true }, { label: "Matched span" }, { label: "Excerpt" }];
  const failureCells = (item) => {
    const evidence = item.evidence || {};
    return [td(categoryLabel(item.category), "strong"), td(item.source), td(item.prompt, "clip"), td(item.response, "clip"), td(item.expected, "clip"), td(formatScore(item.score), "num"), td(evidence.span), td(evidence.excerpt, "clip")];
  };
  const failures = $("failures");
  let failureHtml = "";
  if (!liveRows.length) {
    failureHtml += emptyNote("No failing live prompts stored for this run.");
  } else {
    failureHtml += tableHtml(failureHeaders, liveRows.map(failureCells));
  }
  const noun = failureCount === 1 ? "prompt" : "prompts";
  failureHtml += '<p class="table-note">' + failureCount + " failing live " + noun + ".</p>";
  if (fixtureRows.length) {
    failureHtml += '<h3>Fixture or smoke, not counted</h3>' +
      tableHtml(failureHeaders, fixtureRows.map(failureCells)) +
      '<p class="table-note">These rows are not included in the failing-prompt count.</p>';
  }
  failures.innerHTML = failureHtml;
  renderEvidence(allItems);
}

function detectorNote(item) {
  const notes = item.detector_notes || [];
  return notes
    .filter((row) => row.status === "skipped" || row.status === "not_applicable")
    .map((row) => row.reason || row.status)
    .join("; ");
}

function renderEvidence(items) {
  const root = $("evidence");
  const rows = items.filter((item) => item.suite === "factcheck" || item.suite === "garak" || item.evidence || detectorNote(item));
  if (!rows.length) {
    root.innerHTML = emptyNote("No scored prompts stored for this run.");
    return;
  }
  const shown = rows.slice(0, 80);
  root.innerHTML = tableHtml(
    [{ label: "Suite" }, { label: "Score", num: true }, { label: "Match" }, { label: "Matched span" }, { label: "Excerpt" }, { label: "Detector" }],
    shown.map((item) => {
      const evidence = item.evidence || {};
      return [td(item.suite, "strong"), td(formatScore(item.score), "num"), td(evidence.match || item.detector), td(evidence.span), td(evidence.excerpt || item.response, "clip"), td(detectorNote(item), "clip")];
    }),
  ) + (rows.length > shown.length
    ? '<p class="table-note">Showing the first ' + shown.length + " of " + rows.length + " scored prompts. Export the HTML report for the full list.</p>"
    : "");
}

$("refresh-results").addEventListener("click", () => showResults().catch((err) => alert(err.message)));
$("results-run").addEventListener("change", () => showResults().catch((err) => alert(err.message)));

$("analyze-run").addEventListener("click", async () => {
  const runId = $("results-run").value;
  if (!runId) return;
  $("analysis-text").textContent = "Writing analysis...";
  try {
    await api("/api/runs/" + runId + "/analyze", { method: "POST" });
    await showResults();
  } catch (err) {
    $("analysis-text").textContent = err.message;
  }
});

$("export-report").addEventListener("click", () => {
  const runId = $("results-run").value;
  if (!runId) return;
  window.open("/api/runs/" + runId + "/report", "_blank");
});

$("download-report").addEventListener("click", () => {
  const runId = $("results-run").value;
  if (!runId) return;
  const link = document.createElement("a");
  link.href = "/api/runs/" + encodeURIComponent(runId) + "/report?download=1";
  link.download = "eval-report-" + runId + ".html";
  document.body.appendChild(link);
  link.click();
  link.remove();
});

/* ---------- Compare ---------- */

function compareSlotLabel(name, createdAt, fallback) {
  const title = name || fallback;
  const when = formatDate(createdAt);
  return when ? title + ", " + when : title;
}

function compareBar(label, rate, later, invalid) {
  const withheld = !!invalid;
  const pct = rate == null ? 0 : Math.max(0, Math.min(100, Number(rate) * 100));
  const bar = withheld ? null : barPercent();
  const barMark = bar == null ? "" : '<div class="meter-pass-bar" style="--bar:' + bar + '%"></div>';
  const text = withheld ? "Score withheld" : (rate == null ? "Not run" : formatPercent(rate));
  return '<div class="compare-bar' + (later ? " later" : "") + '"><span>' + label + "</span>" +
    '<div class="meter">' + (withheld || rate == null ? "" : '<div class="meter-fill" style="--v:' + pct + '%"></div>') +
    barMark + "</div>" +
    '<span class="val">' + text + "</span></div>";
}

$("do-compare").addEventListener("click", async () => {
  const out = $("compare-out");
  try {
    const data = await api("/api/compare?left=" + encodeURIComponent($("compare-left").value) + "&right=" + encodeURIComponent($("compare-right").value));
    const invalidSide = !!(data.left_invalid || data.right_invalid);
    const leftLabel = compareSlotLabel(data.left_label, data.left_created_at, "Earlier");
    const rightLabel = compareSlotLabel(data.right_label, data.right_created_at, "Later");
    const cats = data.categories.map((row) => {
      const delta = invalidSide ? { text: "Not compared", cls: "delta-flat" } : formatDelta(row.delta, true);
      return '<div class="compare-row"><div class="meter-name">' + escapeHtml(row.label) + "</div>" +
        '<div class="compare-bars">' + compareBar(leftLabel, row.left_pass_rate, false, data.left_invalid) + compareBar(rightLabel, row.right_pass_rate, true, data.right_invalid) + "</div>" +
        '<div class="compare-delta ' + delta.cls + '">' + escapeHtml(delta.text) + "</div></div>";
    }).join("");
    const warning = invalidSide
      ? '<div class="banner banner-fail">An invalid run has no category score. Deltas against it are not shown.</div>'
      : "";
    let html = warning + '<div class="panel"><div class="panel-head"><h3>Pass rate by category</h3>' +
      '<p class="hint">Change is in percentage points, the later run minus the earlier run.</p></div>' +
      '<div class="compare-cats">' + (cats || emptyNote("Neither run has category scores.")) + "</div></div>";
    const unpairedNote = !invalidSide && data.unpaired_note
      ? '<p class="hint">' + escapeHtml(data.unpaired_note) + "</p>"
      : "";
    if (data.items && data.items.length) {
      const shown = data.items.slice(0, 30);
      html += '<div class="panel"><div class="panel-head"><h3>Prompts that changed most</h3>' +
        '<p class="hint">Largest changes first. Positive means the later run scored higher.</p>' +
        unpairedNote + '</div><div class="table-wrap">' +
        tableHtml(
          [{ label: "Prompt" }, { label: leftLabel, num: true }, { label: rightLabel, num: true }, { label: "Change", num: true }],
          shown.map((row) => {
            const delta = formatDelta(row.delta, false);
            return [td(row.prompt, "clip"), td(formatScore(row.left_score), "num"), td(formatScore(row.right_score), "num"), td(delta.text, "num " + delta.cls)];
          }),
        ) + "</div></div>";
    } else if (!invalidSide && Number(data.unpaired_prompts || 0) > 0) {
      html += '<div class="panel"><div class="panel-head"><h3>Prompts that changed most</h3>' +
        unpairedNote + "</div></div>";
    } else if (!invalidSide && data.unchanged_prompts) {
      html += '<div class="panel"><div class="panel-head"><h3>Prompts that changed most</h3>' +
        '<p class="hint">No prompt scores changed.</p></div></div>';
    }
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = '<div class="banner banner-fail">' + escapeHtml(err.message) + "</div>";
  }
});

/* ---------- Judges ---------- */

function judgeCloudBanner() {
  const cloud = state.judges.some((row) => row.cloud || (row.base_url && !/localhost|127\.0\.0\.1/.test(row.base_url)));
  $("judge-cloud-banner").classList.toggle("hidden", !cloud);
}

function renderJudges() {
  const root = $("judge-rows");
  root.innerHTML = "";
  if (!state.judges.length) {
    root.innerHTML = '<div class="panel panel-quiet">' + emptyNote("No judges yet. Add at least two for peer ranking.") + "</div>";
  }
  state.judges.forEach((judge, index) => {
    const wrap = document.createElement("div");
    wrap.className = "panel judge-card";
    const id = (field) => "judge-" + index + "-" + field;
    wrap.innerHTML = '<div class="judge-head"><h3>Judge ' + (index + 1) + "</h3>" +
      '<button type="button" class="link" data-remove="' + index + '">Remove</button></div>' +
      '<div class="fields">' +
      '<div class="field"><label for="' + id("model") + '">Model</label><input id="' + id("model") + '" data-field="model" data-index="' + index + '" value="' + escapeHtml(judge.model || "") + '" spellcheck="false"></div>' +
      '<div class="field"><label for="' + id("base_url") + '">Base URL</label><input id="' + id("base_url") + '" data-field="base_url" data-index="' + index + '" value="' + escapeHtml(judge.base_url || "") + '" spellcheck="false"></div>' +
      '<div class="field"><label for="' + id("type") + '">Type</label><input id="' + id("type") + '" data-field="type" data-index="' + index + '" value="' + escapeHtml(judge.type || "ollama") + '" spellcheck="false"></div>' +
      "</div>";
    root.appendChild(wrap);
  });
  root.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => {
      const row = state.judges[Number(input.dataset.index)];
      row[input.dataset.field] = input.value;
      judgeCloudBanner();
    });
  });
  root.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => {
      state.judges.splice(Number(button.dataset.remove), 1);
      renderJudges();
    });
  });
  judgeCloudBanner();
}

async function loadJudges() {
  const data = await api("/api/judges");
  state.judges = data.judges || [];
  $("chairman").value = data.chairman || "";
  if (data.timeout) $("judge-timeout").value = data.timeout;
  if (data.max_tokens) $("judge-max-tokens").value = data.max_tokens;
  renderJudges();
}

$("add-judge").addEventListener("click", () => {
  state.judges.push({ model: "", base_url: "http://127.0.0.1:11434", type: "ollama", cloud: false, mode: "chat" });
  renderJudges();
});

$("save-judges").addEventListener("click", async () => {
  try {
    const saved = await api("/api/judges", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chairman: $("chairman").value.trim(),
        judges: state.judges,
        timeout: Number($("judge-timeout").value) || 90,
        max_tokens: Number($("judge-max-tokens").value) || 1200,
      }),
    });
    state.judges = saved.judges;
    renderJudges();
  } catch (err) {
    alert(err.message);
  }
});

/* ---------- Boot ---------- */

async function boot() {
  await loadOffline();
  await loadConnections();
  await loadPresets();
  await loadRuns();
  await loadJudges();
  updateLocalWeights();
  const sample = await api("/api/datasets/sample");
  $("dataset-note").textContent = "Sample set has " + sample.row_count + " public facts.";
  if ($("results-run").value) await showResults();
}

boot().catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin", '<div class="banner banner-fail">' + escapeHtml(err.message) + "</div>");
});
