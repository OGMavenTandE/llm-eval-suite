# Evaluation gates for small-model specialization

The `gates` package takes one workload that a frontier model handles today and
walks it down the Enterprise AI Adoption Curve: seven stages, with an
evaluation gate between each pair of stages. Every gate ends in one of three
decisions:

- **GO**: every bar in the spec was met. The work moves to the next stage.
- **NO-GO**: a bar was missed. The work does not move until it is met.
- **INVALID**: the inputs cannot support a decision (too few items, too many
  replies that do not parse to a label). No score is reported as a pass or a fail.

| Gate | Command | Question it answers |
|---|---|---|
| 1 Mission metrics before seats | the spec file | What is the task, and what is good enough? |
| 2 Independent quality measurement | `gates baseline` | How well does the model in use today actually do? |
| 3 Right-sizing log audit | `gates audit` | Which calls and tokens could move to a small model or plain code? |
| 4 Workflow and agent harness | `gates router` | Do the routing rules send hard work to the large model and easy work off it? |
| 5 Acceptance | `gates accept` | Does the specialized model pass the same tests as the model it would replace? |
| 6 Continuous T&E | `gates drift` | After cutover, has anything drifted? |

Each gate writes `gateN.html` (standalone, prints cleanly, works offline),
`gateN.md`, and `gateN.json` with the SHA-256 of every input file.
`gates packet` writes `PACKET.md`, the index of the evidence for one workload.
The exit code is 0 for GO, 1 for NO-GO and 2 for INVALID.

## How the bars are judged

- An unparsed reply is scored wrong, never skipped. A reply that names two
  labels ("PRIORITY or IMMEDIATE") does not parse.
- Gate 5 compares the candidate and baseline **on the same held-out items**.
  Non-inferiority passes only when the lower end of the 95% paired bootstrap
  interval for (candidate accuracy minus baseline accuracy) is at or above the
  negative margin. A point estimate that looks equal is not enough when the
  test set is too small to tell; pick the test-set size and margin together.
- Labels with a `min_recall` in the spec are high-cost labels. The candidate
  must meet that recall and must not trail the baseline on it beyond the margin.
- Gate 6 checks the predicted-label mix (PSI) on any traffic, version agreement
  on shared items, and accuracy drop on a labeled monitoring set when one is given.

## The message-triage reference case

`examples/message_triage/` is a public, synthetic workload: assign one of four
precedences (FLASH, IMMEDIATE, PRIORITY, ROUTINE) to a message. Every message is
generated from templates with a fixed seed. No message appears in both the
train and test splits, and one phrasing per label is kept out of training
entirely (`heldout_phrasing`), so Gate 5 reports how each model does on wording
the specialized model never saw.

Regenerate the files with `python examples/message_triage/make_data.py`.

### Simulated demo

```bash
python -m gates demo --out runs/gates-demo
```

This writes a full packet with **simulated** predictions so the layout and
checks can be reviewed without a model. Every page carries a SIMULATED banner.
Do not quote its numbers.

### Real run (one workstation with a GPU)

Install the training extras once on a connected machine:

```bash
python -m pip install -e ".[hf,lora]"
```

1. **Gate 1.** Review `examples/message_triage/spec.yaml`. The bars are set
   before anything is scored; change them now or not at all.
2. **Gate 2.** Score the model in use today on the test split. Any connector
   the suite supports works, for example an OpenAI-compatible endpoint:

   ```bash
   python -m gates predict --spec examples/message_triage/spec.yaml \
     --items examples/message_triage/items.jsonl --provider openai \
     --model <frontier model> --base-url <endpoint> --out runs/triage/baseline.jsonl
   python -m gates baseline --spec examples/message_triage/spec.yaml \
     --items examples/message_triage/items.jsonl \
     --predictions runs/triage/baseline.jsonl --out runs/triage/packet
   ```

3. **Gate 3.** `python -m gates audit --config examples/message_triage/audit.yaml --log examples/message_triage/traffic_log.jsonl --out runs/triage/packet`.
   For a program's own traffic, export its logs as JSONL with `prompt`,
   `tokens_in` and `tokens_out`.
4. **Gate 4.** `python -m gates router --config examples/message_triage/audit.yaml --cases examples/message_triage/router_cases.jsonl --out runs/triage/packet`.
5. **Build.** Pick a US open base model from the DoW Bench leaderboard,
   download it once to a local folder, then train offline:

   ```bash
   python -m gates.train_lora --spec examples/message_triage/spec.yaml \
     --items examples/message_triage/items.jsonl --base <base model folder> \
     --out runs/triage/lora --merge
   ```

   Check the base model's license for the intended use before training.
6. **Gate 5.**

   ```bash
   python -m gates predict --spec examples/message_triage/spec.yaml \
     --items examples/message_triage/items.jsonl --provider hf \
     --model triage-lora --folder runs/triage/lora/merged --out runs/triage/candidate.jsonl
   python -m gates accept --spec examples/message_triage/spec.yaml \
     --items examples/message_triage/items.jsonl \
     --candidate runs/triage/candidate.jsonl --candidate-name "specialized small model" \
     --baseline runs/triage/baseline.jsonl --out runs/triage/packet
   ```

7. **Gate 6.** Score the accepted model on `shift_items.jsonl` as the
   reference, then score a changed setup (a new base-model version, a
   different quantization, or the same model next month) on the same items and
   compare:

   ```bash
   python -m gates drift --spec examples/message_triage/spec.yaml \
     --items examples/message_triage/shift_items.jsonl \
     --reference runs/triage/ref.jsonl --current runs/triage/cur.jsonl \
     --reference-name "accepted model" --current-name "after update" --out runs/triage/packet
   ```

8. `python -m gates packet --out runs/triage/packet --workload "Message precedence triage"`.

For an air-gapped run, build the offline bundle (see
[AIRGAP_INSTALL.md](AIRGAP_INSTALL.md)), copy the base-model folder and the
example files across, and run steps 5 to 8 with the network disconnected.

## Limits

- The reference case is synthetic and templated. A specialized model will find
  it easier than real traffic. It shows the method and the evidence trail; it
  is not a claim about any program's workload.
- Gate 3 sorts traffic with keyword rules from the audit config. Tune them to
  real logs, and read the workload map as a starting point for review.
- Token counts that are missing from a log are estimated at four characters
  per token, and the report says so.
