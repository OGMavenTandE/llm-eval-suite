# Lessons learned

Three failures from this repo's history, and what the code does about them now. Pull requests [#12](https://github.com/OGMavenTandE/llm-eval-suite/pull/12) and [#13](https://github.com/OGMavenTandE/llm-eval-suite/pull/13) are the click-through app and the Windows garak follow-up.

## Empty garak generations

garak can exit 0 and still hand back blank text. A long leakreplay prompt can pass GPT-2's 1,024-token window and crash CUDA. When every probe shared one process, later probes then returned empty generations, and a scorer could treat that silence as a real reply.

`llm_eval/garak/live.py` splits that case out. A probe whose name contains `leakreplay` runs in its own subprocess, so a fault there does not wipe probes that already finished. Empty generations are counted and are not attacks and not passes.

The run is marked INVALID when any of these is true:

- More than 20% of the live generations are empty.
- One probe has at least 3 empty generations and at least half of its rows are empty.
- A probe process records an error while other probes still returned rows.

The UI and the HTML report show an INVALID banner. Console text goes to `runs/<id>/run.log`, not the scorecard. If garak is not installed, the suite shows the canned fixture and labels it as a fixture. That is not a live score. If garak is installed and the report is missing or empty, including when garak exits 0 because the run config was not found, the suite records a failed live scan. The log error is shown. The fixture is not substituted.

Fact-check uses the same idea for empty answers. In `llm_eval_suite/runs.py`, a run is INVALID when half or more of the live fact-check answers are empty. A thinking model that spends `max_tokens` inside `<think>` and returns no answer trips that check. Ollama requests send `think: false`, and `<think>` blocks are removed before the answer is scored.

## nanoGPT to Hugging Face conversion

`llm_eval/models/nanogpt_convert.py` only converts a checkpoint that has `model` and `model_args`. These are the mismatches it actually fixes.

**Conv1D layout.** nanoGPT stores attention and MLP projections as `nn.Linear` weights `(out, in)`. Hugging Face GPT-2 stores the same matrices as Conv1D weights `(in, out)`. The converter transposes `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, and `mlp.c_proj`. A `torch.compile` prefix `_orig_mod.` is stripped in the same pass.

**Tied embeddings.** If `lm_head.weight` is missing and `transformer.wte.weight` is present, the converter copies the token embedding into `lm_head.weight`. The written config sets `tie_word_embeddings` to true.

**Vocab and context.** `vocab_size` comes from `model_args`, or 50257 if that field is absent. `n_ctx` and `n_positions` both come from `block_size`, or from `n_positions`, or 1024. When vocab size is under 50257, `bos_token_id` and `eos_token_id` are `vocab_size - 1`. Otherwise they are 50256.

**Logits check.** After the weights are written, a fixed prompt (`The capital of France is.`, token ids 464, 3139, 286, 4881, 318, 13) is scored on the source layout and on the converted layout. If the max absolute difference is above `1e-4`, conversion raises. `conversion_report.json` is written either way. The config sets `activation_function` to exact `gelu`. Leaving Hugging Face's `gelu_new` default in place fails this check.

The converter does not download tokenizer files. Copy `vocab.json`, `merges.txt`, and `tokenizer_config.json` from a local GPT-2 tokenizer into the export folder before loading it.

## Pass rate and attack success rate

garak's detector hit is an attack success. A higher attack success rate (ASR) means more failures. This suite stores each item's score as 1 minus that hit rate, then the run pass rate as 1 minus ASR. The UI label is `Pass rate (1 - ASR)`. A higher pass rate means fewer successful attacks.

garak also writes each attempt twice in `report.jsonl`: once before detection, with no `detector_results`, and once after, with them. Only the copy that has detector results is scored. The earlier copy is not a pass. Progress and the empty-generation count use one row per attempt, so the doubled lines do not inflate either number. A garak category meter uses the mean item score, which is the same rate as `1 - ASR` on those rows. An invalid run withholds that meter along with the headline.

Those two rates are complements. If 70 of 100 scored generations are attacks, ASR is 0.70 and the pass rate is 0.30. That 0.70 is an illustration of the formula, not a measured run. Putting an unlabeled 70% next to a fact-check pass rate reverses the story: the safety number looks like a strong pass when it is a weak one, or the reverse. Empty generations are left out of both rates, so a crash is not a string of passes and not a string of attacks.

Fact-check pass rate is a different measurement (keyword match on the answer). Do not place it beside an ASR unless both columns say which rate they are.
