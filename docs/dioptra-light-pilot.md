# Dioptra light pilot

NIST Dioptra is a testbed for the NIST AI Risk Management Framework Measure function. It tracks experiments, queues jobs, and stores metrics through a REST API, a web UI, and a Python client. A full install is Docker Compose plus worker containers. That stack is out of scope here.

This pilot maps Dioptra's vocabulary onto the eval suite and writes the same shape to a local JSON file. An engineer can read the map and run the smoke command in a few minutes. Nothing in this path pulls container images, uses a GPU, or calls a Dioptra server.

## What you can run

From the repo root, after `pip install -e .`:

```bash
python -m llm_eval.dioptra smoke
```

That writes `results/dioptra/smoke-experiment.json`. The file has one experiment, one job, and two metrics. `live_api` is false. `metrics_source` is `public-fixture`. The metric numbers are placeholders so the schema can be checked. They are not scores from a model.

The same command is available as `llm-eval-dioptra smoke` after install.

To map a run the suite already audited:

```bash
python -m llm_eval.dioptra export-audit results/audit/<run_id>.json \
  --output results/dioptra/<run_id>.json
```

If the audit points at a `results_summary.csv`, each evaluator row becomes two metrics, `{name}_mean_score` and `{name}_pass_rate`, at step 0. If the CSV is missing, the job is still written and the command prints a warning. Dry-runs often have no summary file. That is expected.

## Concept map

| Dioptra | Eval suite | This pilot |
| --- | --- | --- |
| Experiment | `run_name` plus the YAML config | `ExperimentRecord` |
| Group | Not modeled | `group_id` stays null. `group_label` is `local` |
| Entrypoint | YAML config: dataset, models, evaluators | Local name `yaml-eval`. Not registered |
| Job | One model execution, recorded in the audit JSON | `JobRecord`. `submitted` is false |
| Queue and worker | In-process runner | Not used |
| Metric | `mean_score` and `pass_rate` in the summary CSV | `name`, `value`, `step` |
| Artifact | Summary CSV, detailed JSON, audit JSON | Paths listed. Not uploaded |
| Snapshot | `config_hash` on the audit | Copied onto job parameters. Not a Dioptra snapshot |

Dioptra keys a metric by name plus step. The public client posts `{ "name", "value", "step" }` to `POST /api/v1/jobs/{id}/metrics`. NaN and infinities are the strings `nan`, `inf`, and `-inf`. The offline client uses that encoding.

Suite status, if a later step actually submitted the job:

| Suite status | Dioptra status |
| --- | --- |
| `validated` (dry run) | Not submitted |
| `completed` | `finished` |
| `failed_validation` | `failed` |
| `failed_runtime` | `failed` |

The JSON field is `dioptra_status_if_submitted`. The pilot never sets `submitted` to true.

## What the JSON is for

`rest_examples` holds request bodies shaped like the public API:

- `POST /api/v1/experiments` with `group_id`, `name`, `description`, and `entrypoints`
- `POST /api/v1/jobs/{job_id}/metrics` with `name`, `value`, and `step`

Every example includes a `blocked_reason`. `group_id`, entrypoint ids, and job ids are null or omitted because a live deployment assigns them. Do not treat the file as something Dioptra has stored.

## Optional full deployment

If a lab already runs Dioptra, use that deployment's own docs to start it. The upstream project is [usnistgov/dioptra](https://github.com/usnistgov/dioptra). The docs home is [pages.nist.gov/dioptra](https://pages.nist.gov/dioptra/). Architecture, experiments and jobs, metrics, and the Python client:

- [Architecture overview](https://pages.nist.gov/dioptra/explanation/architecture-overview.html)
- [Experiments and jobs](https://pages.nist.gov/dioptra/explanation/components/experiments-jobs-explanation.html)
- [Metrics](https://pages.nist.gov/dioptra/reference/dioptra-components/metrics-reference.html)
- [Python client setup](https://pages.nist.gov/dioptra/how-to/essential-workflows/setup-python-client.html)

The live client is `connect_json_dioptra_client()`. It reads `DIOPTRA_API` and defaults to `http://localhost`. Creating an experiment needs an integer `group_id`. Running a job needs a registered entrypoint, a queue, and a worker. This repo does not start that stack, and CI does not either.

A later integration could read `rest_examples` and post them after those ids exist. That step is not implemented.

## Tests

```bash
python3 -m pytest tests/unit/test_dioptra_pilot.py -v
```

The tests build the fixture, round-trip the JSON, check metric encoding, and map a small audit plus summary CSV. They do not need a model server or Docker.
