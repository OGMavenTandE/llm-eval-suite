const state = {
  presets: [],
  connections: [],
  runs: [],
  judges: [],
  datasetId: "sample",
  activeRun: null,
  timer: null,
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

function show(tab) {
  for (const name of ["connect", "run", "results", "compare", "judges"]) {
    $(name).classList.toggle("hidden", name !== tab);
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => show(button.dataset.tab));
});

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
  try {
    const data = await api("/api/ollama/tags?base_url=" + encodeURIComponent($("conn-url").value.trim() || "http://127.0.0.1:11434"));
    const select = $("ollama-models");
    select.innerHTML = "";
    for (const name of data.models) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (data.models[0]) $("conn-model").value = data.models[0];
  } catch (err) {
    alert(err.message);
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
  box.classList.remove("hidden");
  box.textContent = "Testing...";
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

async function loadConnections() {
  const data = await api("/api/connections");
  state.connections = data.connections;
  const list = $("connection-list");
  list.innerHTML = "";
  const runSelect = $("run-connection");
  const previous = runSelect.value;
  runSelect.innerHTML = "";
  if (!state.connections.length) {
    list.textContent = "No saved connections yet.";
  }
  for (const row of state.connections) {
    const card = document.createElement("div");
    card.className = "card";
    const cloud = row.cloud ? " Cloud endpoint." : " Local.";
    card.textContent = row.name + " · " + row.type + " · " + (row.model || row.folder || "") + "." + cloud;
    list.appendChild(card);
    const option = document.createElement("option");
    option.value = row.id;
    option.textContent = row.name + " (" + (row.model || row.type) + ")";
    runSelect.appendChild(option);
  }
  if (previous) runSelect.value = previous;
}

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

function formatSeconds(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "unknown";
  const total = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes <= 0) return rest + "s";
  return minutes + "m " + rest + "s";
}

function showPreset() {
  const preset = state.presets.find((row) => row.id === $("run-preset").value);
  if (!preset) return;
  $("preset-note").textContent = preset.description;
  const warning = $("preset-warning");
  warning.textContent = preset.warning || "";
  warning.classList.toggle("hidden", !preset.warning);
  const probes = preset.probe_count == null ? "unspecified" : preset.probe_count;
  const prompts = preset.prompt_count == null ? "unspecified" : preset.prompt_count;
  const facts = preset.factcheck_count == null ? 0 : preset.factcheck_count;
  const estimate = preset.estimated_seconds == null ? "not estimated" : formatSeconds(preset.estimated_seconds);
  const source = preset.estimate_source === "measured" ? "measured seconds per prompt" : "default 3.0 s per prompt until a run is measured";
  $("preset-estimate").textContent =
    "Probes: " + probes + ". Prompts (cap): " + prompts + ". Fact-check questions: " + facts +
    ". Estimate: " + estimate + " (" + source + ", " + preset.seconds_per_prompt + " s/prompt).";
}

$("run-preset").addEventListener("change", showPreset);

$("run-dataset").addEventListener("change", async () => {
  state.datasetId = $("run-dataset").value;
  const data = await api("/api/presets?dataset_id=" + encodeURIComponent(state.datasetId || "sample"));
  state.presets = data.presets;
  showPreset();
});

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
    $("progress").textContent = "Demo pair started on " + data.preset + ". Compare will open when both runs finish.";
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(pollDemo, 1000);
  } catch (err) {
    $("progress").textContent = err.message;
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
    $("progress").textContent = err.message;
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

function sourceBadge(source, label) {
  const kind = source === "live" ? "badge-live" : "badge-fixture";
  const text = source === "live" ? (label || "Live") : (label || "Fixture / smoke (no live model call)");
  return '<span class="badge ' + kind + '">' + escapeHtml(text) + "</span>";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function renderProgress(run) {
  const suites = (run.progress && run.progress.suites) || [];
  const bits = ["Status: " + run.status];
  if (run.validity === "invalid") bits.push("INVALID: " + (run.validity_reason || "empty generations"));
  for (const suite of suites) {
    const done = suite.done == null ? 0 : suite.done;
    const total = suite.total == null ? "?" : suite.total;
    let line = suite.name + ": " + done + " / " + total + " (" + (suite.status || suite.source || "") + ")";
    if (suite.eta_seconds != null) {
      line += ". ETA " + formatSeconds(suite.eta_seconds) + " at " + suite.seconds_per_prompt + " s/prompt";
    }
    bits.push(line);
  }
  $("progress").textContent = bits.join("\n");
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
    $("progress").textContent = err.message;
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
    $("progress").textContent = err.message;
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
    for (const run of state.runs) {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = run.run_id + " · " + (run.model || "model") + " · " + run.status;
      select.appendChild(option);
    }
    if (previous) select.value = previous;
  }
}

function categoryCard(row) {
  const card = document.createElement("div");
  card.className = "card";
  const title = document.createElement("div");
  title.textContent = row.label;
  const value = document.createElement("strong");
  if (row.status === "not_run") value.textContent = "Not run";
  else if (row.pass_percent == null) value.textContent = row.status;
  else value.textContent = row.pass_percent + "%";
  card.appendChild(title);
  card.appendChild(value);
  if (row.source && row.source !== "live") {
    const badge = document.createElement("div");
    badge.innerHTML = sourceBadge(row.source, "Fixture / smoke (no live model call)");
    card.appendChild(badge);
  } else if (row.status === "not_run") {
    const note = document.createElement("div");
    note.className = "muted";
    note.textContent = "Not run";
    card.appendChild(note);
  }
  return card;
}

async function showResults() {
  const runId = $("results-run").value;
  if (!runId) return;
  const run = await api("/api/runs/" + runId);
  const items = await api("/api/runs/" + runId + "/items");
  const banner = $("validity-banner");
  const invalid = run.validity === "invalid" || run.status === "invalid";
  banner.classList.toggle("hidden", !invalid);
  banner.textContent = invalid ? ("INVALID run. " + (run.validity_reason || "Too many empty generations.")) : "";
  const garakBits = [];
  if (run.garak_pass_rate_label) {
    const pass = run.garak_pass_rate == null ? "n/a" : Math.round(run.garak_pass_rate * 1000) / 10 + "%";
    const asr = run.garak_attack_success_rate == null ? "n/a" : Math.round(run.garak_attack_success_rate * 1000) / 10 + "%";
    garakBits.push(run.garak_pass_rate_label + ": " + pass + ". ASR: " + asr + ". " + (run.garak_wording || ""));
  }
  if (run.garak_runs_dir) garakBits.push("Garak report folder: " + run.garak_runs_dir);
  if (run.log_path) garakBits.push("Run log: " + run.log_path);
  $("garak-path").textContent = garakBits.join(" ");
  const board = $("scorecard");
  board.innerHTML = "";
  const overall = document.createElement("div");
  overall.className = "card";
  overall.innerHTML = "<div>Overall (live categories)</div><strong>" +
    (run.scorecard && run.scorecard.overall_pass_percent != null ? run.scorecard.overall_pass_percent + "%" : "n/a") +
    "</strong>";
  board.appendChild(overall);
  for (const row of (run.scorecard && run.scorecard.categories) || []) {
    board.appendChild(categoryCard(row));
  }
  const suiteList = $("suite-list");
  suiteList.innerHTML = "";
  for (const suite of run.suites || []) {
    const line = document.createElement("p");
    line.innerHTML = "<strong>" + escapeHtml(suite.name) + "</strong> " +
      sourceBadge(suite.source, suite.label) + " " + escapeHtml(suite.notes || "");
    suiteList.appendChild(line);
  }
  const analysis = run.analysis || {};
  $("analysis-text").textContent = analysis.narrative || "No analysis yet.";
  $("summary-source").textContent = analysis.source_label ? "Summary source: " + analysis.source_label : "";
  const cloud = (analysis.source_label || "").includes("cloud");
  $("cloud-judge-banner").classList.toggle("hidden", !cloud);
  const failures = $("failures");
  const rows = (items.items || []).filter((item) => item.passed === false);
  if (!rows.length) {
    failures.textContent = "No failing prompts stored for this run.";
    renderEvidence(items.items || []);
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = "<tr><th>Category</th><th>Source</th><th>Prompt</th><th>Response</th><th>Expected</th><th>Score</th><th>Matched span</th><th>Excerpt</th></tr>";
  for (const item of rows) {
    const evidence = item.evidence || {};
    const tr = document.createElement("tr");
    tr.innerHTML = [item.category, item.source, item.prompt, item.response, item.expected, item.score, evidence.span, evidence.excerpt]
      .map((value) => "<td>" + escapeHtml(value) + "</td>").join("");
    table.appendChild(tr);
  }
  failures.innerHTML = "";
  failures.appendChild(table);
  renderEvidence(items.items || []);
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
    root.textContent = "No scored prompts stored for this run.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = "<tr><th>Suite</th><th>Score</th><th>Match</th><th>Matched span</th><th>Excerpt</th><th>Detector</th></tr>";
  for (const item of rows.slice(0, 80)) {
    const evidence = item.evidence || {};
    const tr = document.createElement("tr");
    tr.innerHTML = [
      item.suite,
      item.score,
      evidence.match || item.detector,
      evidence.span,
      evidence.excerpt || item.response,
      detectorNote(item),
    ].map((value) => "<td>" + escapeHtml(value) + "</td>").join("");
    table.appendChild(tr);
  }
  root.innerHTML = "";
  root.appendChild(table);
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

$("do-compare").addEventListener("click", async () => {
  try {
    const data = await api("/api/compare?left=" + encodeURIComponent($("compare-left").value) + "&right=" + encodeURIComponent($("compare-right").value));
    const out = $("compare-out");
    const table = document.createElement("table");
    table.innerHTML = "<tr><th>Category</th><th>Earlier</th><th>Later</th><th>Delta</th></tr>";
    for (const row of data.categories) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + escapeHtml(row.label) + "</td><td>" + escapeHtml(row.left_pass_rate) +
        "</td><td>" + escapeHtml(row.right_pass_rate) + "</td><td>" + escapeHtml(row.delta) + "</td>";
      table.appendChild(tr);
    }
    out.innerHTML = "";
    out.appendChild(table);
    if (data.items && data.items.length) {
      const items = document.createElement("table");
      items.innerHTML = "<tr><th>Prompt</th><th>Earlier score</th><th>Later score</th><th>Delta</th></tr>";
      for (const row of data.items.slice(0, 30)) {
        const tr = document.createElement("tr");
        tr.innerHTML = "<td>" + escapeHtml(row.prompt) + "</td><td>" + row.left_score +
          "</td><td>" + row.right_score + "</td><td>" + row.delta + "</td>";
        items.appendChild(tr);
      }
      out.appendChild(items);
    }
  } catch (err) {
    $("compare-out").textContent = err.message;
  }
});

function judgeCloudBanner() {
  const cloud = state.judges.some((row) => row.cloud || (row.base_url && !/localhost|127\.0\.0\.1/.test(row.base_url)));
  $("judge-cloud-banner").classList.toggle("hidden", !cloud);
}

function renderJudges() {
  const root = $("judge-rows");
  root.innerHTML = "";
  state.judges.forEach((judge, index) => {
    const wrap = document.createElement("div");
    wrap.className = "card";
    wrap.innerHTML = ""
      + '<label>Model <input data-field="model" data-index="' + index + '" value="' + escapeHtml(judge.model || "") + '"></label>'
      + '<label>Base URL <input data-field="base_url" data-index="' + index + '" value="' + escapeHtml(judge.base_url || "") + '"></label>'
      + '<label>Type <input data-field="type" data-index="' + index + '" value="' + escapeHtml(judge.type || "ollama") + '"></label>';
    root.appendChild(wrap);
  });
  root.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => {
      const row = state.judges[Number(input.dataset.index)];
      row[input.dataset.field] = input.value;
      judgeCloudBanner();
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

async function boot() {
  await loadPresets();
  await loadConnections();
  await loadRuns();
  await loadJudges();
  const sample = await api("/api/datasets/sample");
  $("dataset-note").textContent = "Sample set has " + sample.row_count + " public facts.";
}

boot().catch((err) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin", "<p class='banner'>" + escapeHtml(err.message) + "</p>");
});
