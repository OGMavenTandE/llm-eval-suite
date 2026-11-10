# Review

Security fixes and deletions in this PR are the ones that keep scores, public pages, and URLs the same. Everything else is listed under "Not changed".

No committed secret, token, password, or private key was found. Nothing to rotate. Git history was not rewritten.

## Security findings

### High — redirects skipped the base-URL check

`requests` follows redirects by default. `check_base_url` ran on the configured host, then a 301/302/303/307/308 could send the next request to a metadata address.

Fixed in `refuse_redirect` (`llm_eval/http_policy.py:69`). These calls pass `allow_redirects=False` and refuse a redirect: `llm_eval/models/openai_model.py:141` and `:206`, `llm_eval/models/ollama_model.py:52` and `:103`, `llm_eval/models/served.py:113` and `:138`, `llm_eval/edge/streaming.py:100` and `:144`, `llm_eval_suite/app.py:324`, `apps/api/services/discovery.py:77`. A health or model-list redirect becomes `{ok: false}` instead of a followed request. A normal 200 is unchanged.

### High — short IPv4 spellings and embedded metadata names

The old host check required four decimal octets. `inet_aton` also accepts `169.254.43518`, `169.16689662`, and `0xA9FEA9FE`, and a resolver turns `customer1.169.254.169.254.nip.io` into the metadata address. `100.100.100.200`, `fd00:ec2::254`, `metadata.tencentyun.com`, and `instance-data.ec2.internal` were not in the block list.

Fixed in `llm_eval/http_policy.py`: metadata hosts at `:22`, extra addresses at `:31`, label scan at `:116`, short IPv4 forms at `:169`. Localhost, `127.1`, and private LAN addresses such as `10.1.2.3` still connect. A hostname that contains a link-local spelling is rejected.

### Medium — run and audit ids could leave their directories

`Path / absolute` drops the left side, so a `resume_run_id` or an indexed `audit_path` could read or write outside the runs or audit directory. The report `Content-Disposition` filename used that id.

Fixed. `RunManager._run_dir` (`llm_eval_suite/runs.py:229`) accepts one safe segment and requires the resolved parent to be the runs directory. `audit_file_path` (`llm_eval/core/audit_service.py:33`) does the same for the audit directory and still returns the relative path that was stored before. `get_run_audit` (`llm_eval/core/run_service.py:198`) opens an indexed audit file only when `_path_inside` (`:22`) says it stays under `output_dir`.

### Medium — dataset ids were globbed

`dataset_id` was interpolated into a glob, so `*` or `..` could select a file outside the intended name.

Fixed in `_dataset_path` (`llm_eval_suite/app.py:571`). The id must match a single safe stem. The lookup lists that directory and keeps a `.jsonl` or `.csv` whose resolved parent is the datasets directory. `sample` is unchanged. An unknown or unsafe id on `POST /api/runs` is HTTP 400 (that path used to raise before the handler and surface as 500).

### Medium — API keys were world-readable and judge keys were returned

`connections.json`, `judges.json`, and `job.json` were created with the process umask and hold API keys. `GET /api/judges` returned the judge key. The Judges screen has no key field; it round-trips the judge object.

Fixed. `write_private_text` (`llm_eval_suite/connections.py:99`) writes mode `0600` and is used for those three files (`job.json` at `llm_eval_suite/runs.py:492`). Judge responses use the same `set` placeholder as connections (`_public_judge`, `connections.py:111`). Saving `set`, or omitting the field, keeps the stored key. An explicit empty string clears it. Renaming a judge model drops a key the screen never showed.

### Low — local dashboard inserted raw JSON into HTML

`dashboard.html` built sample rows and evaluation details with string concatenation. A local results file could break out of the markup via `sample_idx` or a detail value.

Fixed. Detail keys and values are escaped (`dashboard.html:1305`). Displayed indexes are escaped. Row ids use the loop index (`:1218`, `:1239`), not the raw index.

## Removals

- `openai_stream_generate`, `_post_stream`, and `_openai_blocking` were removed from `llm_eval/edge/streaming.py`. Nothing in the repo called them. Live OpenAI, vLLM, and llama.cpp streaming stays on `OpenAIModel._generate_stream`. `consume_openai_sse`, `consume_ollama_lines`, and `ollama_stream_generate` stay.
- `apps/api/routes/artifacts.py` was a one-line stub and is deleted, including its export. Artifact routes remain on `apps/api/routes/runs.py`. `llm_eval/storage/artifacts.py` stays.
- An unused `allPass` local in `dashboard.html` was removed. It was not read.

## Not changed

### High — no DNS lookup

`check_base_url` (`llm_eval/http_policy.py:9`) uses the host string only. A name that resolves to a metadata address is not blocked. Looking it up would make the check depend on the network and would fail closed for air-gapped names. Leave this as an optional follow-up.

### High — the code sandbox is not a security boundary

`standard_bench/sandbox.py:47` documents a subprocess, resource limits, and a socket monkeypatch. It does not create a network namespace or a hypervisor. Tests require the child to write in its temp directory. A tighter sandbox would change scored runs. Product decision.

### Medium — nanoGPT checkpoints are loaded with pickle

`torch.load(..., weights_only=False)` is at `llm_eval/models/nanogpt_convert.py:35`. `weights_only=True` can reject checkpoints that carry `model_args`, and the logits check must stay at `1e-4`. Change it only with a fixture that still passes that check.

### Medium — the local APIs have no token

The click-through app binds `127.0.0.1:8765` and the JSON API binds `127.0.0.1:8000`. A browser on another origin cannot make a JSON POST without a preflight. A client that can already reach loopback is unauthenticated. A token would change the documented local API and the UI.

### Medium — YAML paths are not confined

`config_service.py:40` opens `config_path` as given, and `:74` stores `output_dir` as given. Users pass absolute configs and result directories. Confining them would break that.

### Medium — dataset fetch follows redirects with the HF token

`standard_bench/fetch.py:69` uses `urllib.request.urlopen`, which follows redirects, and the request can carry `HF_TOKEN`. The URL is built from the catalog (`huggingface.co`), not from an eval prompt. An allowlist can break a CDN redirect. Recommend checking the host again before the token is sent on a redirected request.

### Medium — fact-check uploads have no size cap

`validate_factcheck_text` (`llm_eval_suite/suites.py:107`) writes the body, then validates it. A limit needs a chosen maximum.

### Low — OpenAPI docs stay on

`apps/api/main.py:9` leaves FastAPI's `/docs` and `/redoc` enabled on the local API. Turning them off changes a local developer URL.

### Low — duplicated helpers

`is_local_url` (`llm_eval_suite/connections.py:81`), `is_local_endpoint` (`llm_eval/offline.py:78`), and `_is_local_base_url` (`llm_eval/models/openai_model.py:14`) decide "local" separately. SSE parsing also exists in more than one place. Merging them can change which hosts count as local.

### Low — `requirements.txt` overlaps `pyproject.toml`

Both list `requests`, `pyyaml`, and `pydantic`. `docs/TUTORIAL.md:146` installs from the requirements file. Keep it.

### Low — frontend `npm audit`

`npm audit` reports 14 findings: 2 critical, 6 high, 6 moderate. The critical and high findings are in the Vite and Vitest dev tree. `react-router-dom` (a production dependency) is moderate. `npm audit fix --force` was not run.

### Low — plugin assets trust the local manifest

`plugin_asset` (`llm_eval_suite/app.py:403`) serves a file named by the on-disk plugin map. The path itself cannot contain a slash. Tightening this belongs with the plugin host.

### Low — version string mismatch

The click-through app reports `0.3.0` (`llm_eval_suite/app.py:139`). The package and the JSON API report `0.2.0`. Changing the string changes the OpenAPI document.

### Left as they are

- Private LAN ranges stay allowed. `10.1.2.3` is a valid model server.
- `trust_remote_code` stays off unless a connection sets it.
- `third_party/certifi` and `third_party/tqdm` stay. The air-gap bundle and the attribution docs require those license texts.
- No deploy script, DNS change, or secret was touched.

## What ran

There is no ruff, flake8, or mypy config, and CI does not run a linter. Python was 3.12.3 in a local venv. The same four tests stayed skipped (Playwright UI tests when Chromium is not installed).

| Command | Before | After |
| --- | --- | --- |
| `python -m pytest -q` | 369 passed, 4 skipped, 1 warning, 46.60s | 372 passed, 4 skipped, 1 warning, 46.85s |
| `python -m standard_bench verify-reference --fixtures` | reference ok | reference ok |
| `python -m standard_bench verify-reference --data tests/evalplus_prepared` | reference ok | reference ok |
| `cd frontend && npm test` | 17 passed | 17 passed |
| `cd frontend && npm run build` | vite build ok | vite build ok |

The three extra passing tests cover redirects, run and dataset ids, and private key files. Frontend sources were not modified. `standard_bench` sources were not modified.

## Counts

Tracked files at `HEAD`, compared with this branch. A file that contains a null byte is counted as a file and not as lines. Dependencies are the declared requirements, not the transitive install.

| | Before | After |
| --- | ---: | ---: |
| Files | 335 (323 text, 12 binary) | 335 (323 text, 12 binary) |
| Text lines | 59237 | 59553 |
| Python runtime deps | 3 (`pydantic`, `pyyaml`, `requests`) | 3 |
| Optional dev / api / hf | 3 / 3 / 2 | 3 / 3 / 2 |
| `requirements.txt` | 3 (same runtime set) | 3 |
| Frontend dependencies / devDependencies | 4 / 10 | 4 / 10 |

One stub file was removed and this review was added, so the file count is unchanged. No dependency was added or removed.
