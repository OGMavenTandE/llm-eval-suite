# GPQA Diamond, in-house only

This suite is not part of `standard_capability`, `standard_capability_quick`, or the customer zip. Do not put the questions in the customer kit, in logs, or in exports. The authors ask that the questions not be republished in plain text.

Staff with their own Hugging Face access can fetch it on a connected machine:

```bash
python -m standard_bench fetch --dest datasets/gpqa --suites gpqa_diamond
```

Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment for that command. The token is not written to the manifest or the dataset. Unset it before you copy the directory. Copy the directory only.

Run the in-house preset against that directory:

```bash
python -m standard_bench run --preset standard_capability_gpqa --data datasets/gpqa --model "OLMo 2 1B Instruct" --folder path/to/model --output runs/gpqa.csv
```

`--include-gpqa` adds the suite to a customer preset. The data directory then needs the six customer suites and the GPQA files. If the GPQA files are absent, the other suites still run and the scorecard records GPQA as not run.

Question text is not stored on items, in the log, or in the CSV.
