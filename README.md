# GPT-J function vector reproduction

This reproduces Todd et al., Function Vectors (arXiv 2310.15213, ICLR 2024), on GPT-J-6B with nnsight 0.7.0 on a Kaggle T4. The function vector is added to the layer 9 output at the last token position.

Setup facts come from the stored outputs. Notebook cell 2 ("Cell 1") prints `nnsight      : 0.7.0`, `transformers : 5.13.0`, `torch        : 2.10.0+cu128` and `Tesla T4`. Notebook cell 3 ("Cell 2a") loads `EleutherAI/gpt-j-6B` and prints `dtype: torch.float16`. Notebook cell 4 ("Cell 2b") prints the injection: `model.transformer.h[9].output[0][:, -1, :] += fv_vector`.

Files: `fv_gptj_kaggle.ipynb` is the Kaggle run with its stored outputs, cited below as `notebook cell N ("label")`, where N is the cell's 1-based position and the label is the header on its first line. Its sha256 is `30c30b86538a99fbb8cbf22d4586211606eeb9afbb768788fbb4218b5a24bfb6`. The file records no Kaggle version number. Its papermill metadata records a run from 2026-10-19 17:18:42 to 18:18:05 UTC.

## Method

- Data. country_capital uses 197 pairs embedded in notebook cell 5 ("Cell 3"), which prints `197 country-capital pairs`. A comment there says they equal Todd's `country-capital.json`, but no cell checks this. antonym and present_past use Todd's `antonym.json` and `present-past.json`, which notebook cell 11 downloads and checks against sha256 `ba66e8e399944b557251b11a37b11f9ba7a50f0a57286f7021a8a7b26b469e8d` and `bd6f7d484a967646aa5854740c70803568225f47a5630e1555c39e827ba9c78b`. Each seed shuffles a task's pairs with `random.Random(seed)` and holds out 40 test items. The rest form the extraction pool: 157, 2358 and 253 items (`pool=` in notebook cell 12 ("Cell 9")).
- Extraction. Each seed draws 100 prompts from its pool, each with 10 demonstrations and one query (`N_ICL, N_TRIALS, N_TEST, SEED = 10, 100, 40, 0` in notebook cell 5 ("Cell 3")). The last-token input to every layer's `out_proj` is split by head (`Per-head split : out_proj.input [1,S,4096] -> [1,S,16,256]`, notebook cell 4 ("Cell 2b")) and averaged over the 100 prompts into `mean_act: (28, 16, 256)` (notebook cell 5 ("Cell 3")).
- Vector. For each of the 10 heads in `UNIVERSAL_TOP_INDICES`, the head's mean activation is placed in its 256-wide slice of a zero vector and passed through that layer's `out_proj`. The function vector is the sum of the 10 results, printed as `fv_vector: (1, 4096)` (notebook cell 5 ("Cell 3")). GPT-J's `out_proj` has no bias (`bias=False` in notebook cell 4 ("Cell 2b")).
- Evaluation. FV and no-FV scoring use the zero-example frame `Q: {x}\nA:` for every task. Notebook cell 5 ("Cell 3") prints the country_capital case, `'Q: Laos\nA:'`. Notebook cell 12 ("Cell 9") scores all three tasks with the same functions: `clean_pred` and `fv_pred` from notebook cell 5 ("Cell 3") and `exact_capital_match` from notebook cell 10 ("Cell 8"). Each builds its prompt with `framed()` from notebook cell 5 ("Cell 3").

## Deviations

- The head ranking is reused from Todd's published causal mediation results, not recomputed. The heads are hardcoded as `UNIVERSAL_TOP_INDICES` in notebook cell 5 ("Cell 3"). Todd's code computes per-task heads by default and uses a universal set only with `--universal_set` (`src/evaluate_function_vector.py`). This repo does not check which of the two the paper's Table 6 used.
- Each test set has 40 items. Todd's `split_icl_dataset` (`src/utils/prompt_utils.py`) holds out 30% and then keeps 70% of that as test, about 21% of each dataset, or roughly 42, 504 and 61 items for country_capital, antonym and present_past.

These match the paper (arXiv v2): the vector is added at layer 9 of GPT-J at the last token, and accuracy is reported over 5 seeds on test items that 10-shot ICL answers correctly (Section 3.1). Accuracy is top-1 on the answer's first token, and GPT-J uses 10 heads (Appendix C).

## Results

Five seeds, first-token accuracy. Each mean ± std is taken over the five per-seed values, with sample std (ddof=1). Notebook cell 12 ("Cell 9") prints `5-seed (ddof=1)`, and notebook cell 16 ("Cell 13") computes its summary with `np.std(fracs, ddof=1)`. Each seed has 40 held-out test items.

### Baselines

All 40 test items, five-seed mean ± std. No FV is the `base=` value in the `5-seed (ddof=1)` lines of notebook cell 12 ("Cell 9"). 10-shot ICL is the `first-token` mean in notebook cell 15 ("Cell 12"). The FV column repeats the unfiltered means below.

| Task | No FV | 10-shot ICL | FV |
|---|---|---|---|
| country_capital | 0.050 ± 0.025 | 0.965 ± 0.022 | 0.840 ± 0.065 |
| antonym | 0.025 ± 0.018 | 0.595 ± 0.067 | 0.360 ± 0.038 |
| present_past | 0.015 ± 0.022 | 0.960 ± 0.014 | 0.180 ± 0.093 |

### Filtered first-token

FV-correct among the test items that 10-shot ICL answers correctly (10-shot ICL scores are in notebook cell 15 ("Cell 12")). Per-seed counts are the `first-token filtered n/d` lines and the means are the `per-task filtered summary` lines, both from notebook cell 16 ("Cell 13"). That cell scores each seed's FV predictions once and takes the filtered count from the same list, restricted to items 10-shot ICL got right, so each denominator is that seed's number of ICL-correct items. The paper column is Table 6 of arXiv v2 (zero-shot, v_t). Table 6 does not name its model, seed count or filtering. GPT-J, ± as the std over 5 seeds and the ICL-filtered subset are read from Section 3.1 and Appendix C, which describe the paper's results in general. These are the paper's values, not outputs of this notebook.

| Task | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± std | Paper (Todd et al.) |
|---|---|---|---|---|---|---|---|
| country_capital | 33/39 | 30/39 | 37/39 | 32/37 | 35/39 | 0.865 ± 0.066 | 0.832 ± 0.027 |
| antonym | 13/25 | 11/20 | 13/22 | 14/26 | 14/26 | 0.548 ± 0.027 | 0.482 ± 0.020 |
| present_past | 1/38 | 10/39 | 10/38 | 8/39 | 7/38 | 0.187 ± 0.096 | 0.197 ± 0.059 |

Pooled over seeds, notebook cell 16 ("Cell 13") prints 167/193, 65/119 and 36/192. Their 95% Wilson intervals are 0.810 to 0.906, 0.457 to 0.633 and 0.139 to 0.249. The intervals were computed for this README and are not in the notebook.

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

- Antonym filtered first-token is 6.6 points above the paper: 0.548 (notebook cell 16 ("Cell 13")) minus the paper's 0.482. The paper's value falls inside the pooled 95% Wilson interval, 0.457 to 0.633. The cause is unresolved.
- present_past seed 0 scores 0.025 unfiltered first-token, against 0.250, 0.250, 0.200 and 0.175 on seeds 1 to 4 (notebook cell 12 ("Cell 9")). The cause is unresolved. A vector-by-item swap between seeds 0 and 1 (notebook cell 23 ("pp-swap-v1")) scored 1/40 for the seed-0 vector on seed-0 items, 2/40 for the seed-1 vector on seed-0 items, 6/40 for the seed-0 vector on seed-1 items, and 10/40 for the seed-1 vector on seed-1 items. The seed-0 items score low under either vector, though the seed-0 vector also scores lower on seed-1 items. This is one swap at n = 40. The two cross-seed cells are not strictly held out, because each vector's extraction pool includes most of the other seed's test items, so those two scores may be inflated. Exact match equals first-token in all four conditions, and the filtered counts are 1/38, 2/38, 6/39 and 10/39.
- Query-only antonym extraction, restricted to 10-shot-eligible queries, ran on all five seeds (notebook cell 19 ("aqo-extract-v2")). Filtered first-token was 0.557566 ± 0.035544 against 0.547566 ± 0.026506 for the original (notebook cell 20 ("aqo-evaluate-v1")). The paired difference was +0.010000 ± 0.022361 (notebook cell 20 ("aqo-evaluate-v1")), from one item flip, and the vectors have cosine 0.998404, the five-seed mean (notebook cell 21 ("aqo-cosine-v1")). Notebook cell 22 ("aqo-diagnostics-v1") counts a (both right), b (original right, query-only wrong), c (the reverse) and d (both wrong). The flip is in seed 1, with a=11 b=0 c=1 d=8, and b=c=0 for every other seed. Restricting extraction queries to eligible ones does not explain the 6.6-point gap. Details are under "Query-only extraction" below.
- A Run All, including a Kaggle Save Version, stops at notebook cell 22 ("aqo-diagnostics-v1") if its concordance check raises, so notebook cell 23 ("pp-swap-v1") does not run in that version. Notebook cell 23 ("pp-swap-v1") does not depend on any aqo cell and can be run on its own afterward.

## Query-only extraction

A pool query is eligible when 10-shot ICL, with demonstrations drawn from the rest of the pool, puts the gold first token at rank 0 (`"eligible": predicted == gold` in notebook cell 19 ("aqo-extract-v2")). 1496, 1491, 1468, 1483 and 1478 of 2358 pool queries were eligible on seeds 0 to 4, the same as the `rank0-kept` counts in notebook cell 17 ("Cell 14").

The query-only arm draws 10 demonstrations and a query from a fresh `random.Random(seed)`, the same way the original extraction does, and skips a draw whose query is not eligible. It took 153, 155, 150, 158 and 160 draws to accept 100 (`proposals=` in notebook cell 19 ("aqo-extract-v2")). Reading the code, an accepted draw numbered 100 or lower is the same prompt as the original trial with that number. The cell's closing line, `Accepted trials are not individually matched to the original trials`, means no cell compares the two sets trial by trial. It does not mean they are disjoint. The number of shared trials was not printed.

Per-seed cosine between the original and query-only vectors runs from 0.998041 to 0.998741 (notebook cell 21 ("aqo-cosine-v1")). For scale, original antonym vectors from different seeds have cosine min 0.995306, mean 0.996029 and max 0.996903, and query-only/original norm ratios run from 0.995579 to 1.010850 (notebook cell 22 ("aqo-diagnostics-v1")). The same cell gives a and d for the seeds with no flip: a=13 d=12 for seed 0, a=13 d=9 for seed 2, a=14 d=12 for seed 3 and a=14 d=12 for seed 4.

## Reading the stored outputs

Notebook cell 13 ("Cell 10") prints `norm  observed=40.938  expected=40.939  -> DIFFER`. That line compares the norm of the stored fp16 vector, formatted to three decimals, with the fp32 norm that notebook cell 5 ("Cell 3") printed, by exact string match. Notebook cell 14 ("Cell 11") checks the fp32 norm against 40.939 with a tolerance of 0.002 and prints `AGREE`. The DIFFER comes from fp16 rounding.

## Not done

No cross-family transfer, no tool-selection experiment, no layer sweep, and no accuracy-versus-coefficient sweep; notebook cell 6 ("Cell 5") prints top-1 tokens at scales 1, 3, 5 and 10 on 10 country_capital items without scoring accuracy.

## Provenance

Notebook changes were made by Devansh Ruia along with the help of coding agents, under an append-only rule that keeps earlier cells byte-identical. The released file is the Kaggle run copy. It differs from that append-only notebook in three ways. A data-fetch cell, notebook cell 11, was inserted before notebook cell 12 ("Cell 9") because the Kaggle dataset mount failed; it downloads Todd's two JSON files from GitHub, pinned by sha256. The `_DATA` line in notebook cell 12 ("Cell 9") points at the fetched files. Kaggle reassigned the cell ids on import. No cell source was otherwise changed.
