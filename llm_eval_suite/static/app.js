const state = {
  presets: [],
  connections: [],
  runs: [],
  judges: [],
  datasetId: "sample",
  activeRun: null,
  timer: null,
};

// Matches PASS_BAR in llm_eval_suite/scoring.py.
const PASS_BAR_PERCENT = 80;

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
  button.addEventListener("click", () => show(button.dataset.tab));
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
  const kind = source === "live" ? "badge-live" : "badge-fixture";
  const text = source === "live" ? (label || "Live") : (label || "Fixture or smoke, no live model call");
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
  };
}

function updateCloudBanner() {
  const profile = profileFromForm();
  const local = !profile.base_url || /localhost|127\.0\.0\.1|\[::1\]/.test(profile.base_url);
  $("cloud-banner").classList.toggle("hidden", local || profile.type === "hf" || profile.type === "nanogpt");
}

["conn-url", "conn-type"].forEach((id) => $(id).addEventListener("input", updateCloudBanner));
$("conn-type").addEventListener("change", updateCloudBanner);

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
  const data = await api("/api/presets");
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
  const estimate = preset.estimated_seconds == null ? "Not estimated" : formatSeconds(preset.estimated_seconds);
  const source = preset.estimate_source === "measured"
    ? "Estimate uses measured speed on this machine: " + preset.seconds_per_prompt + " s per prompt."
    : "Estimate assumes " + preset.seconds_per_prompt + " s per prompt until a run is measured.";
  const cell = (term, value) => "<div><dt>" + escapeHtml(term) + "</dt><dd>" + escapeHtml(value) + "</dd></div>";
  $("preset-estimate").innerHTML = '<dl class="estimate-grid">' +
    cell("Garak probes", probes) +
    cell("Prompt cap", prompts) +
    cell("Fact checks", facts) +
    cell("Estimated time", estimate) +
    '</dl><p class="estimate-source">' + escapeHtml(source) + "</p>";
}

$("run-preset").addEventListener("change", showPreset);

$("run-dataset").addEventListener("change", async () => {
  state.datasetId = $("run-dataset").value;
  const data = await api("/api/presets?dataset_id=" + encodeURIComponent(state.datasetId || "sample"));
  state.presets = data.presets;
  showPreset();
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
    const indeterminate = pct == null && running && suite.status !== "completed";
    const finished = suite.status === "completed" || (!running && pct == null);
    const width = pct == null ? (finished ? 100 : 0) : pct;
    const raw = suite.status || suite.source || "";
    let eta = raw ? raw.charAt(0).toUpperCase() + raw.slice(1).replace(/_/g, " ") : "";
    if (suite.eta_seconds != null) {
      eta = "About " + formatSeconds(suite.eta_seconds) + " left at " + suite.seconds_per_prompt + " s per prompt";
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
  const overall = card.overall_pass_percent;
  const invalid = run.validity === "invalid" || run.status === "invalid";
  let verdict;
  let verdictClass;
  if (invalid) {
    verdict = "Invalid run";
    verdictClass = "invalid";
  } else if (overall == null) {
    verdict = "No live categories scored";
    verdictClass = "none";
  } else if (overall >= PASS_BAR_PERCENT) {
    verdict = "Meets the " + PASS_BAR_PERCENT + "% pass bar";
    verdictClass = "pass";
  } else {
    verdict = "Below the " + PASS_BAR_PERCENT + "% pass bar";
    verdictClass = "fail";
  }
  const detailBits = ["Average pass rate across live categories."];
  if (card.live_item_count != null) detailBits.push(card.live_item_count + " live prompts scored");
  if (card.failure_count != null) detailBits.push(card.failure_count + " failing");
  const detail = detailBits.length > 1
    ? detailBits[0] + " " + detailBits.slice(1).join(", ") + "."
    : detailBits[0];
  const connection = run.connection || {};
  const meta = [
    ["Model", connection.model || connection.name || "Unknown"],
    ["Preset", run.preset || "Custom"],
    ["Started", formatDate(run.created_at) || "Unknown"],
  ];
  const num = overall == null ? "n/a" : Number(overall).toFixed(1);
  $("readout").innerHTML =
    '<div class="readout-score' + (overall == null ? " is-empty" : "") + '"><span class="num">' + num + "</span>" +
    (overall == null ? "" : '<span class="unit">%</span>') + "</div>" +
    '<div><p class="verdict ' + verdictClass + '">' + escapeHtml(verdict) + "</p>" +
    '<p class="readout-detail">' + escapeHtml(detail) + "</p></div>" +
    '<dl class="run-meta">' +
    meta.map(([term, value]) => "<dt>" + escapeHtml(term) + '</dt><dd title="' + escapeHtml(value) + '">' + escapeHtml(value) + "</dd>").join("") +
    "<dt>Status</dt><dd>" + statusBadge(run.status) + "</dd></dl>";
}

function meterRow(row) {
  const status = row.status === "not_run" ? "not_run"
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
    : row.pass_percent == null ? escapeHtml(row.status) : Number(row.pass_percent).toFixed(1) + "<small>%</small>";
  const badge = status === "pass" ? '<span class="badge badge-pass">Pass</span>'
    : status === "fail" ? '<span class="badge badge-fail">Below bar</span>'
      : status === "fixture" ? '<span class="badge badge-fixture">Fixture</span>'
        : '<span class="badge badge-neutral">Not run</span>';
  const label = status === "not_run"
    ? row.label + ": not run"
    : row.label + ": " + (row.pass_percent == null ? row.status : row.pass_percent + "% pass") + ", " + sub + ". Pass bar " + PASS_BAR_PERCENT + "%.";
  return '<div class="meter-row" data-status="' + status + '">' +
    '<div><span class="meter-name">' + escapeHtml(row.label) + '</span><span class="meter-sub">' + escapeHtml(sub) + "</span></div>" +
    '<div class="meter" role="img" aria-label="' + escapeHtml(label) + '">' +
    (status === "not_run" ? "" : '<div class="meter-fill" style="--v:' + pct + '%"></div>') +
    '<div class="meter-pass-bar" style="--bar:' + PASS_BAR_PERCENT + '%"></div></div>' +
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

async function showResults() {
  const runId = $("results-run").value;
  if (!runId) return;
  const run = await api("/api/runs/" + runId);
  const items = await api("/api/runs/" + runId + "/items");
  const banner = $("validity-banner");
  const invalid = run.validity === "invalid" || run.status === "invalid";
  banner.classList.toggle("hidden", !invalid);
  banner.textContent = invalid ? ("Invalid run. " + (run.validity_reason || "Too many empty generations.")) : "";

  const pathBits = [];
  if (run.garak_pass_rate_label) {
    const pass = run.garak_pass_rate == null ? "n/a" : Math.round(run.garak_pass_rate * 1000) / 10 + "%";
    const asr = run.garak_attack_success_rate == null ? "n/a" : Math.round(run.garak_attack_success_rate * 1000) / 10 + "%";
    pathBits.push(run.garak_pass_rate_label + ": " + pass + ". ASR: " + asr + ". " + (run.garak_wording || ""));
  }
  if (run.garak_runs_dir) pathBits.push("Garak report folder: " + run.garak_runs_dir);
  if (run.log_path) pathBits.push("Run log: " + run.log_path);
  $("garak-path").innerHTML = pathBits.map((bit) => "<span>" + escapeHtml(bit) + "</span>").join("");

  const card = run.scorecard || {};
  renderReadout(run, card);
  $("scorecard").innerHTML = (card.categories || []).map(meterRow).join("");

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
  const rows = allItems.filter((item) => item.passed === false);
  const failures = $("failures");
  if (!rows.length) {
    failures.innerHTML = emptyNote("No failing prompts stored for this run.");
  } else {
    failures.innerHTML = tableHtml(
      [{ label: "Category" }, { label: "Source" }, { label: "Prompt" }, { label: "Response" }, { label: "Expected" }, { label: "Score", num: true }, { label: "Matched span" }, { label: "Excerpt" }],
      rows.map((item) => {
        const evidence = item.evidence || {};
        return [td(categoryLabel(item.category), "strong"), td(item.source), td(item.prompt, "clip"), td(item.response, "clip"), td(item.expected, "clip"), td(formatScore(item.score), "num"), td(evidence.span), td(evidence.excerpt, "clip")];
      }),
    ) + '<p class="table-note">' + rows.length + " failing prompt" + (rows.length === 1 ? "" : "s") + ".</p>";
  }
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

/* ---------- Compare ---------- */

function compareBar(label, rate, later) {
  const pct = rate == null ? 0 : Math.max(0, Math.min(100, Number(rate) * 100));
  return '<div class="compare-bar' + (later ? " later" : "") + '"><span>' + label + "</span>" +
    '<div class="meter">' + (rate == null ? "" : '<div class="meter-fill" style="--v:' + pct + '%"></div>') +
    '<div class="meter-pass-bar" style="--bar:' + PASS_BAR_PERCENT + '%"></div></div>' +
    '<span class="val">' + (rate == null ? "Not run" : formatPercent(rate)) + "</span></div>";
}

$("do-compare").addEventListener("click", async () => {
  const out = $("compare-out");
  try {
    const data = await api("/api/compare?left=" + encodeURIComponent($("compare-left").value) + "&right=" + encodeURIComponent($("compare-right").value));
    const cats = data.categories.map((row) => {
      const delta = formatDelta(row.delta, true);
      return '<div class="compare-row"><div class="meter-name">' + escapeHtml(row.label) + "</div>" +
        '<div class="compare-bars">' + compareBar("Earlier", row.left_pass_rate, false) + compareBar("Later", row.right_pass_rate, true) + "</div>" +
        '<div class="compare-delta ' + delta.cls + '">' + escapeHtml(delta.text) + "</div></div>";
    }).join("");
    let html = '<div class="panel"><div class="panel-head"><h3>Pass rate by category</h3>' +
      '<p class="hint">Change is in percentage points, later minus earlier.</p></div>' +
      '<div class="compare-cats">' + (cats || emptyNote("Neither run has category scores.")) + "</div></div>";
    if (data.items && data.items.length) {
      const shown = data.items.slice(0, 30);
      html += '<div class="panel"><div class="panel-head"><h3>Prompts that changed most</h3>' +
        '<p class="hint">Largest drops first.</p></div><div class="table-wrap">' +
        tableHtml(
          [{ label: "Prompt" }, { label: "Earlier score", num: true }, { label: "Later score", num: true }, { label: "Change", num: true }],
          shown.map((row) => {
            const delta = formatDelta(row.delta, false);
            return [td(row.prompt, "clip"), td(formatScore(row.left_score), "num"), td(formatScore(row.right_score), "num"), td(delta.text, "num " + delta.cls)];
          }),
        ) + "</div></div>";
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
      body: JSON.stringify({ chairman: $("chairman").value.trim(), judges: state.judges }),
    });
    state.judges = saved.judges;
    renderJudges();
  } catch (err) {
    alert(err.message);
  }
});

/* ---------- Boot ---------- */

async function boot() {
  await loadPresets();
  await loadConnections();
  await loadRuns();
  await loadJudges();
  const sample = await api("/api/datasets/sample");
  $("dataset-note").textContent = "Sample set has " + sample.row_count + " public facts.";
}

boot().catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin", '<div class="banner banner-fail">' + escapeHtml(err.message) + "</div>");
});
