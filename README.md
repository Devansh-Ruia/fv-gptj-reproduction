# GPT-J function vector reproduction

This reproduces Todd et al., Function Vectors (arXiv 2310.15213, ICLR 2024), on GPT-J-6B with nnsight 0.7.0 on a Kaggle T4. The function vector is added to the layer 9 output at the last token position.

Setup facts come from the stored outputs. Notebook cell 2 ("Cell 1") prints `nnsight      : 0.7.0` and `Tesla T4`. Notebook cell 3 ("Cell 2a") loads `EleutherAI/gpt-j-6B`. Notebook cell 4 ("Cell 2b") prints the injection: `model.transformer.h[9].output[0][:, -1, :] += fv_vector`.

Files: `fv_gptj_kaggle.ipynb` is the Kaggle run with its stored outputs, cited below as `notebook cell N ("label")`, where N is the cell's 1-based position and the label is the header on its first line.

## Deviation

The head ranking is reused from Todd's published causal mediation results, not recomputed. The heads are hardcoded as `UNIVERSAL_TOP_INDICES` in notebook cell 5 ("Cell 3").

## Results

Five seeds, first-token accuracy. Each mean ± std is taken over the five per-seed values, with sample std (ddof=1). Notebook cell 12 ("Cell 9") prints `5-seed (ddof=1)`, and notebook cell 16 ("Cell 13") computes its summary with `np.std(fracs, ddof=1)`. Each seed has 40 held-out test items.

### Filtered first-token

FV-correct among the test items that 10-shot ICL answers correctly (10-shot ICL scores are in notebook cell 15 ("Cell 12")). Per-seed counts are the `first-token filtered n/d` lines and the means are the `per-task filtered summary` lines, both from notebook cell 16 ("Cell 13"). The paper column holds the paper's values, not outputs of this notebook.

| Task | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± std | Paper (Todd et al.) |
|---|---|---|---|---|---|---|---|
| country_capital | 33/39 | 30/39 | 37/39 | 32/37 | 35/39 | 0.865 ± 0.066 | 0.832 |
| antonym | 13/25 | 11/20 | 13/22 | 14/26 | 14/26 | 0.548 ± 0.027 | 0.482 |
| present_past | 1/38 | 10/39 | 10/38 | 8/39 | 7/38 | 0.187 ± 0.096 | 0.197 |

### Unfiltered first-token

FV-correct among all 40 test items. Per-seed counts are the `unfiltered n/40` lines in notebook cell 16 ("Cell 13"). They match the per-seed `ft=` values in notebook cell 12 ("Cell 9"), and the means are its `5-seed (ddof=1)` lines.

| Task | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± std |
|---|---|---|---|---|---|---|
| country_capital | 33/40 | 30/40 | 37/40 | 33/40 | 35/40 | 0.840 ± 0.065 |
| antonym | 14/40 | 12/40 | 15/40 | 16/40 | 15/40 | 0.360 ± 0.038 |
| present_past | 1/40 | 10/40 | 10/40 | 8/40 | 7/40 | 0.180 ± 0.093 |

## Hook check

baukit and nnsight give identical activations at the attention `out_proj` input, checked on all 28 `out_proj` modules. Notebook cell 5 ("Cell 3") prints `len(names): 28` and `baukit vs nnsight  max|out_proj.input diff| = 0.000e+00`.

## Open issues

- Antonym filtered first-token is 6.6 points above the paper: 0.548 (notebook cell 16 ("Cell 13")) minus the paper's 0.482. The cause is unresolved.
- present_past seed 0 scores 0.025 unfiltered first-token, against 0.250, 0.250, 0.200 and 0.175 on seeds 1 to 4 (notebook cell 12 ("Cell 9")). The cause is unresolved. A vector-by-item swap between seeds 0 and 1 (notebook cell 23 ("pp-swap-v1")) scored 1/40 for the seed-0 vector on seed-0 items, 2/40 for the seed-1 vector on seed-0 items, 6/40 for the seed-0 vector on seed-1 items, and 10/40 for the seed-1 vector on seed-1 items. The seed-0 items score low under either vector, though the seed-0 vector also scores lower on seed-1 items. This is one swap at n = 40. The two cross-seed cells are not strictly held out, because each vector's extraction pool includes most of the other seed's test items, so those two scores may be inflated.
- Query-only antonym extraction, restricted to 10-shot-eligible queries, ran on all five seeds (notebook cell 19 ("aqo-extract-v2")). Filtered first-token was 0.557566 ± 0.035544 against 0.547566 ± 0.026506 for the original (notebook cell 20 ("aqo-evaluate-v1")). The paired difference was +0.010000 ± 0.022361 (notebook cell 20 ("aqo-evaluate-v1")), from one item flip, and the vectors have cosine 0.998404, the five-seed mean (notebook cell 21 ("aqo-cosine-v1")). Notebook cell 22 ("aqo-diagnostics-v1") counts a (both right), b (original right, query-only wrong), c (the reverse) and d (both wrong). The flip is in seed 1, with a=11 b=0 c=1 d=8, and b=c=0 for every other seed. Restricting extraction queries to eligible ones does not explain the 6.6-point gap.
- A Run All, including a Kaggle Save Version, stops at notebook cell 22 ("aqo-diagnostics-v1") if its concordance check raises, so notebook cell 23 ("pp-swap-v1") does not run in that version. Notebook cell 23 ("pp-swap-v1") does not depend on any aqo cell and can be run on its own afterward.

## Not done

No cross-family transfer, no tool-selection experiment, no layer sweep, and no accuracy-versus-coefficient sweep; notebook cell 6 ("Cell 5") prints top-1 tokens at scales 1, 3, 5 and 10 on 10 country_capital items without scoring accuracy.

## Provenance

Notebook changes were made by separate maker and checker coding agents, under an append-only rule that keeps earlier cells byte-identical. The released file is the Kaggle run copy. It differs from that append-only notebook in three ways. A data-fetch cell, notebook cell 11, was inserted before notebook cell 12 ("Cell 9") because the Kaggle dataset mount failed; it downloads Todd's two JSON files from GitHub, pinned by sha256. The `_DATA` line in notebook cell 12 ("Cell 9") points at the fetched files. Kaggle reassigned the cell ids on import. No cell source was otherwise changed.
