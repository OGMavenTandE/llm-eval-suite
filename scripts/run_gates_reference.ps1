# Unattended run of every evaluation gate on the message-triage reference case.
#
# Run from a fresh clone (it never touches another checkout or .venv-eval):
#   powershell -ExecutionPolicy Bypass -File scripts\run_gates_reference.ps1
#
# What it does, logging everything to <OutRoot>\run.log:
#   1. Makes .venv-gates here with CUDA torch, transformers and peft.
#   2. Picks the baseline: the largest Ollama model at or under -MaxBaselineGB.
#   3. Uses -BaseFolder if given, else downloads ibm-granite/granite-4.0-micro (Apache-2.0, not gated).
#   4. Scores the baseline (Ollama) and the untuned base model (local folder) on the 600 test messages.
#   5. Gates 2, 3, 4.
#   6. Trains a LoRA adapter on the 900 training messages, scores it, Gate 5.
#   7. Gate 6: accepted model (base + adapter) against the base model with the adapter missing,
#      on the 400 shifted-month messages.
#   8. PACKET.md and summary.json.
# A NO-GO is a result, not a failure, so the script keeps going after one.

param(
    [string]$OutRoot = "$HOME\llm-eval-suite\runs\gates-triage-$(Get-Date -Format yyyyMMdd-HHmm)",
    [string]$BaseFolder = "",
    [string]$BaseRepo = "ibm-granite/granite-4.0-micro",
    [string]$Baseline = "",
    [double]$MaxBaselineGB = 20,
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [int]$Epochs = 2,
    [int]$Batch = 4
)

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$Packet = Join-Path $OutRoot "packet"
$Untuned = Join-Path $OutRoot "packet-untuned"
$Preds = Join-Path $OutRoot "predictions"
New-Item -ItemType Directory -Force -Path $Packet, $Untuned, $Preds | Out-Null
Start-Transcript -Path (Join-Path $OutRoot "run.log") -Append | Out-Null

# Keep the PC awake while this script runs. This is a request from this process only;
# it changes no power settings and ends when the script exits.
Add-Type -Namespace Gates -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Gates.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$Summary = [ordered]@{ started = (Get-Date).ToString("s"); repo_commit = (git rev-parse HEAD); steps = [ordered]@{} }
function Step($name, [scriptblock]$body) {
    Write-Host "`n=== $name  $(Get-Date -Format HH:mm:ss) ==="
    $t = Get-Date
    $code = 0
    try { & $body 2>&1 | Out-Host; $code = $LASTEXITCODE } catch { Write-Host "ERROR in ${name}: $_"; $code = 99 }
    $Summary.steps[$name] = [ordered]@{ exit = $code; seconds = [int]((Get-Date) - $t).TotalSeconds }
    $Summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $OutRoot "summary.json")
    return $code
}

$Spec = "examples\message_triage\spec.yaml"
$Items = "examples\message_triage\items.jsonl"
$Shift = "examples\message_triage\shift_items.jsonl"
$Audit = "examples\message_triage\audit.yaml"
$Py = Join-Path $Repo ".venv-gates\Scripts\python.exe"

# 1. Environment ------------------------------------------------------------
Step "setup" {
    if (-not (Test-Path $Py)) { python -m venv .venv-gates }
    & $Py -m pip install --upgrade pip -q
    & $Py -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu126 -q
    & $Py -m pip install -e ".[hf,lora]" "transformers==5.18.0" "torch==2.13.0" --extra-index-url https://download.pytorch.org/whl/cu126 -q
    & $Py -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('transformers', transformers.__version__, 'peft', peft.__version__)"
} | Out-Null

# 2. Baseline model from Ollama ------------------------------------------------
Step "pick-baseline" {
    $tags = Invoke-RestMethod "$OllamaUrl/api/tags"
    $models = $tags.models | Where-Object { $_.name -notmatch "embed" } | Sort-Object size -Descending
    $models | ForEach-Object { Write-Host ("  {0,-45} {1,6:N1} GB" -f $_.name, ($_.size / 1GB)) }
    if (-not $Baseline) {
        $pick = $models | Where-Object { ($_.size / 1GB) -le $MaxBaselineGB } | Select-Object -First 1
        if (-not $pick) { throw "No Ollama model at or under $MaxBaselineGB GB." }
        $script:Baseline = $pick.name
        if ($models[0].name -ne $pick.name) {
            Write-Host "  Largest model $($models[0].name) is over $MaxBaselineGB GB; using $($pick.name) so 600 calls finish."
        }
    }
    Write-Host "  Baseline: $script:Baseline"
    $Summary.baseline_model = $script:Baseline
} | Out-Null

# 3. Base model for training -------------------------------------------------
Step "base-model" {
    if (-not $BaseFolder) {
        $script:BaseFolder = Join-Path $HOME ("models\" + ($BaseRepo -replace "/", "__"))
        if (-not (Test-Path (Join-Path $script:BaseFolder "config.json"))) {
            & $Py -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'])" $BaseRepo $script:BaseFolder
        }
    }
    if (-not (Test-Path (Join-Path $script:BaseFolder "config.json"))) { throw "No config.json in $script:BaseFolder" }
    Write-Host "  Base model folder: $script:BaseFolder"
    $Summary.base_model = $BaseRepo
    $Summary.base_folder = $script:BaseFolder
} | Out-Null

# 4. Score the baseline and the untuned base model ------------------------------
Step "predict-baseline" {
    & $Py -m gates predict --spec $Spec --items $Items --provider ollama --model $Baseline --base-url $OllamaUrl --out "$Preds\baseline.jsonl"
} | Out-Null
# Free the GPU before loading local models.
try { Invoke-RestMethod -Method Post "$OllamaUrl/api/generate" -Body (@{ model = $Baseline; keep_alive = 0 } | ConvertTo-Json) -ContentType "application/json" | Out-Null } catch {}

Step "predict-untuned" {
    & $Py -m gates predict --spec $Spec --items $Items --provider hf --model "untuned base" --folder $BaseFolder --out "$Preds\untuned.jsonl"
} | Out-Null

# 5. Gates 2, 3, 4 ------------------------------------------------------------
Step "gate2" { & $Py -m gates baseline --spec $Spec --items $Items --predictions "$Preds\baseline.jsonl" --out $Packet } | Out-Null
Step "gate3" { & $Py -m gates audit --config $Audit --log examples\message_triage\traffic_log.jsonl --out $Packet } | Out-Null
Step "gate4" { & $Py -m gates router --config $Audit --cases examples\message_triage\router_cases.jsonl --out $Packet } | Out-Null

# 6. Train, score, Gate 5 -----------------------------------------------------
$Lora = Join-Path $Repo "runs\triage-lora"
Step "train-lora" {
    & $Py -m gates.train_lora --spec $Spec --items $Items --base $BaseFolder --out $Lora --epochs $Epochs --batch $Batch --merge
    Copy-Item "$Lora\train_record.json" $OutRoot -ErrorAction SilentlyContinue
} | Out-Null
Step "predict-tuned" {
    & $Py -m gates predict --spec $Spec --items $Items --provider hf --model "specialized small model" --folder "$Lora\merged" --out "$Preds\tuned.jsonl"
} | Out-Null
Step "gate5" {
    & $Py -m gates accept --spec $Spec --items $Items --candidate "$Preds\tuned.jsonl" --candidate-name "specialized small model" --baseline "$Preds\baseline.jsonl" --out $Packet
} | Out-Null
Step "gate5-untuned" {
    & $Py -m gates accept --spec $Spec --items $Items --candidate "$Preds\untuned.jsonl" --candidate-name "same small model, untuned" --baseline "$Preds\baseline.jsonl" --out $Untuned
} | Out-Null

# 7. Gate 6 -------------------------------------------------------------------
Step "predict-shift" {
    & $Py -m gates predict --spec $Spec --items $Shift --split monitor --provider hf --model "accepted model" --folder "$Lora\merged" --out "$Preds\shift_accepted.jsonl"
    & $Py -m gates predict --spec $Spec --items $Shift --split monitor --provider hf --model "adapter missing" --folder $BaseFolder --out "$Preds\shift_no_adapter.jsonl"
} | Out-Null
Step "gate6" {
    & $Py -m gates drift --spec $Spec --items $Shift --split monitor --reference "$Preds\shift_accepted.jsonl" --current "$Preds\shift_no_adapter.jsonl" --reference-name "accepted model (base + adapter)" --current-name "after update, adapter missing" --out $Packet
} | Out-Null

# 8. Packet -------------------------------------------------------------------
Step "packet" {
    & $Py -m gates packet --out $Packet --workload "Message precedence triage (synthetic reference case)"
} | Out-Null
$Summary.finished = (Get-Date).ToString("s")
foreach ($g in 2..6) {
    $f = Join-Path $Packet "gate$g.json"
    if (Test-Path $f) { $Summary["gate$g"] = (Get-Content $f -Raw | ConvertFrom-Json).verdict }
}
$Summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $OutRoot "summary.json")
Write-Host "`nDone. Reports: $Packet"
Stop-Transcript | Out-Null
