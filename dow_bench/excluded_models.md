# Excluded models

These models were excluded because, although marketed at a smaller effective size, their total weights exceed the 6B category. Gemma 4 E4B is 7.996B total versus its '4.5B effective' label, and Arcee Trinity Nano is 6.120B total with 1B active (MoE).

Stage 1 scores total weights. The 6B line is the category limit. A smaller effective or active count does not bring a model back in. The Hub rows are google/gemma-4-E4B-it (7,996,156,490 parameters) and arcee-ai/Trinity-Nano-Preview (6,120,003,328 parameters). Trinity has no GA repo.

| Model | Repo | Total weights | Marketed smaller figure | Why it is out |
|---|---|---|---|---|
| Gemma 4 E4B | google/gemma-4-E4B-it | 7.996B | 4.5B effective | Total weights are 7.996B once per-layer embeddings are counted. |
| Arcee Trinity Nano | arcee-ai/Trinity-Nano-Preview | 6.120B | 1.0B active | Mixture-of-experts. All experts load, so the total is the size that counts. There is no GA repo. |
