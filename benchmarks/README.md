# HSKM Benchmark Suite

A reproducible comparison of **HSKM (Hierarchical Sparse Kernel Memory)** against
suitable baselines from Hugging Face, covering both **quality** (perplexity +
standard downstream tasks) and **efficiency** (the project's central O(N) claim).

Everything here is designed to be re-run end to end and produce the same
numbers: fixed seeds, fixed model revisions, a single source of truth for the
model registry, and automated tests that verify the harness itself.

```
benchmarks/
├── README.md                  <- you are here
├── run_all.sh                 <- one-command reproduction
├── requirements.txt
├── src/
│   ├── model_registry.py      <- every baseline + the reason it was chosen
│   ├── hskm_lm.py             <- lm-eval-harness adapter for HSKM
│   ├── hskm_stable.py         <- numerically stable replacement for the EMA
│   └── data.py                <- shared, cached corpus pipeline
├── scripts/
│   ├── 01_download_models.py
│   ├── 10_train_controlled.py <- controlled from-scratch HSKM vs GPT-2
│   ├── 20_run_lm_eval.py
│   ├── 30_perplexity.py
│   ├── 40_scaling_benchmark.py
│   └── 50_make_report.py
├── tests/test_benchmark.py    <- correctness tests for the harness
└── results/
    ├── REPORT.md              <- generated tables
    ├── figures/               <- generated plots
    └── ...                    <- raw JSON outputs
```

---

## 1. What HSKM is, and what therefore needs benchmarking

From the upstream README and `model.py`, HSKM is a **decoder-only language
model** with three novel pieces:

| Component | Implementation | Claimed role |
|---|---|---|
| **Multi-Head Kernel Attention (MHKA)** | queries are matched against *learned kernel prototypes*; a **Top-K** sparse softmax selects the most relevant prototypes and their values are aggregated | short-term memory, O(N·K) |
| **Medium-Term Memory (MTM)** | causal EMA over the token axis, evaluated with a parallel cumulative-sum scan | medium-term "narrative flow" |
| **Long-Term Memory (LTM)** | gated read/write over a global bank of 4096 learned patterns | long-term concept retrieval |

The rest is a conventional modern decoder stack: pre-norm **RMSNorm**,
**SwiGLU** feed-forward, **RoPE**, weight-tied embeddings, and a **GPT-2 BPE**
vocabulary of 50257 (+2 special tokens).

Two facts drive the whole benchmark design:

* **The task is causal language modelling.** So every baseline must be a
  causal decoder LM, and every metric must be a language-modelling metric.
* **The headline claim is asymptotic** (O(N) instead of O(N²)). So the
  benchmark must measure *scaling behaviour*, not just a single number at one
  sequence length.

### A note on the architecture, found by reading the code

`MultiHeadKernelAttention` computes the similarity of each token's query to
the learned kernel prototypes and aggregates the prototypes themselves. The
kernels are global parameters and the aggregation is per-token, so **the
attention block does not mix information across token positions**; the
cross-token mixing in HSKM comes from the MTM EMA (causal) and, indirectly,
from the shared kernel bank acting as a learned global codebook.

This is not a criticism — it is a coherent design (and it is *why* the block
is O(N)) — but it means HSKM's "attention" is not directly comparable to
token-to-token attention, and it explains why the O(N) claim holds so
emphatically. The benchmark treats this as a property to measure rather than
an assumption to make.

---

## 2. Baseline selection

Baselines were chosen on two axes: **architectural class** (does it solve the
same problem?) and **scale** (is it big enough for the comparison to be
meaningful?). The full table with per-model reasoning lives in
[`src/model_registry.py`](src/model_registry.py); it is also dumped to JSON by
that module.

### Tier 1 — Linear-time sequence models (the true architectural peers)

HSKM's thesis is that you can get long-context language modelling *without*
quadratic attention. The reference implementations of that idea are
state-space models, and they are the only fair comparison for the asymptotic
claim.

| Model | Params | Why it is the right baseline |
|---|---|---|
| `state-spaces/mamba-130m` | 130M | The canonical selective-SSM linear-time LM and the reference point for "O(N) instead of O(N²)". Matched at ~130M so capacity is held constant. |
| `state-spaces/mamba2-130m` | 130M | Mamba-2 is the current state of the art for linear-time modelling (structured SSM). Same size, so it isolates whether HSKM's advantage survives against the newer variant. |
| `state-spaces/mamba2-1.3b` | 1.3B | Scale point for the linear-time family — tests whether HSKM is competitive once quality, not just efficiency, dominates. |
| `state-spaces/mamba-2.8b-hf` | 2.8B | Large-scale SSM reference, and the release in native `transformers` format, which validates our custom loader for the legacy 130M checkpoints. |

### Tier 2 — Attention-based decoder LMs at matched scale

These answer a different question: **does HSKM's new memory hierarchy buy
anything over a standard causal transformer of the same size?**

| Model | Params | Why it is the right baseline |
|---|---|---|
| `openai-community/gpt2` | 124M | HSKM's direct architectural ancestor (RMSNorm / SwiGLU / RoPE / tied embeddings) and the only baseline sharing HSKM's exact tokenizer, so token-level perplexities are directly comparable with no re-tokenisation. |
| `facebook/opt-125m` | 125M | Same size and GPT-2 BPE family, but *learned absolute* position embeddings instead of RoPE — isolates the value of HSKM's rotary kernel attention. |
| `EleutherAI/pythia-160m` | 160M | Modern open causal LM (Pile), rotary positions and parallel residual attention like HSKM — the closest *pretrained* relative at comparable size. |
| `HuggingFaceTB/SmolLM2-135M` | 135M | A 2024-era small decoder trained on 2T tokens: the current state of the art at ~135M, i.e. the bar HSKM's efficiency has to be weighed against. |
| `facebook/opt-350m` | 350M | Second OPT size point — a controlled 125M→350M scaling curve inside one architecture. |
| `HuggingFaceTB/SmolLM2-360M` | 360M | Modern scale point, pairing with SmolLM2-135M. |
| `EleutherAI/pythia-410m` | 410M | Scale point for the Pile-trained attention family. |
| `Qwen/Qwen2.5-0.5B` | 494M | Strongest small-scale open decoder in its class — the quality ceiling for "what a modern architecture delivers below 0.5B parameters". |

### Tier 3 — Controlled from-scratch models (ours)

HSKM ships **no pretrained checkpoint**, so comparing a from-scratch HSKM
against heavily pretrained baselines is inherently confounded. To remove that
confound we train **two** models ourselves under an *identical* budget —
identical corpus, token count, optimiser, schedule, batch size and sequence
length. The only difference is the architecture.

| Model | Why |
|---|---|
| `hskm-ours` | HSKM trained by us under the controlled budget. |
| `gpt2-ours` | A standard causal GPT-2 trained by us under the *identical* budget. Any gap in held-out perplexity between these two is attributable to the architecture alone. |

The GPT-2 control is matched to HSKM on **parameter count** (not depth),
because HSKM's long-term pattern bank carries a large parameter block that a
transformer of the same depth does not have. The parameter-matched depth is
computed automatically by `find_matched_gpt2()` and reported in the training
summary; a matched-*depth* variant can additionally be trained with
`--also-matched-depth` to separate the two effects.

### Considered and deliberately excluded

| Model | Reason for exclusion |
|---|---|
| RWKV-4/5/6 | The closest *conceptual* peer to HSKM (linear attention with a recurrent state). **All HF-format checkpoints (`EleutherAI/rwkv-4-0.1b`, `BlinkDL/rwkv-4-0.1b`) now return 401 / no longer exist**; only the raw `.pth` training-format files remain, which would require a separate inference stack and a custom lm-eval adapter. Documented as a gap rather than silently omitted. |
| `google/recurrentgemma-2b` | The best modern recurrent (local/global hybrid) peer, but the repository is **gated** and requires accepting the licence with a token that has the right scope. |
| `ai2/jamba-1.5-mini` | Hybrid Mamba+attention — an excellent peer — but the repository returns 401 and is unavailable to this token. |
| Encoder models (BERT, RoBERTa) | Different task (masked LM), so not comparable on the language-modelling metrics used here. |
| Models > 3B | Do not fit the 2×T4 (15 GB) budget for a fair, repeatable all-model comparison at matched context. |

---

## 3. Benchmarks and why

### 3.1 Quality — held-out perplexity

**Corpora:** WikiText-103 raw (validation), WikiText-2 raw (validation), and
TinyStories (validation). WikiText-103 is the standard language-modelling
benchmark corpus at this scale and is the corpus for which published reference
perplexities exist. TinyStories is HSKM's *native* training distribution
(upstream `dataset.py`), so it measures in-domain performance.

**Two metrics, for two different purposes:**

* **Token-level PPL** = `exp(total_nll / total_tokens)`. Directly comparable
  only between models sharing a tokenizer (GPT-2, OPT and HSKM).
* **Word-level PPL** = `exp(total_nll / n_words)` where `n_words` is the
  whitespace-word count of the evaluated text. This is **tokenizer
  independent** and is therefore the primary cross-model metric.

**The evaluation span.** The corpus is truncated to a fixed number of
**words** (`--max-words`, default 500k) and every model tokenises *that same
text prefix*. This matters for two reasons:

* All models are scored on identical text, so the word-level denominator is
  identical across tokenizers. Using a per-model token count for the
  numerator but a shared corpus word count for the denominator would make
  the comparison tokenizer-dependent and therefore meaningless.
* It bounds the cost. The full TinyStories validation split is 4.8M tokens;
  scoring that with a dozen models (including a 2.8B one) would take hours.

**Special-token conventions matter.** Each model is tokenised with its own
tokenizer and its own convention. OPT was trained with `</s>` prepended to
every sequence — scoring it without a BOS token puts it far out of
distribution and inflates its perplexity by orders of magnitude (OPT-125M
measured 8498 token-PPL on WikiText-103 without BOS, against 32.1 with it).
GPT-2 uses no BOS; HSKM prepends the BOS token its upstream training used
(`add_bos=True` in `dataset.py`).

**Controls:** every model is scored with non-overlapping windows of the same
length (`--context`, default 512) so the comparison isolates the model rather
than its context window. `--native-context` switches to each model's own
maximum instead.

**A note on the Mamba baselines' absolute speed.** `mamba_ssm` and
`causal_conv1d` (the fused CUDA kernels) are not installable as prebuilt
wheels for this PyTorch/CUDA combination, so the Mamba models run on
`transformers`' reference PyTorch path. This makes their *absolute* timings
and throughput pessimistic — the fused kernels are considerably faster. Two
things to note:

* The measured **scaling exponent**, which is what this benchmark is about,
  is unaffected: both implementations have the same asymptotic complexity.
* The error direction is conservative. If anything, Mamba's numbers here
  *understate* it, so the comparison does not overstate HSKM's advantage.

### 3.2 Quality — standard downstream tasks

Run with **lm-evaluation-harness 0.4.13**, 0-shot, identical task versions and
scoring rules for every model:

| Task | What it measures |
|---|---|
| `lambada_openai` | Last-word prediction (word-level acc + ppl) |
| `piqa` | Physical commonsense |
| `hellaswag` | Adversarially-filtered sentence completion |
| `arc_easy` | Grade-school science (easy) |
| `arc_challenge` | Grade-school science (hard) |
| `winogrande` | Pronoun resolution |
| `boolq` | Yes/no reading comprehension |
| `openbookqa` | Open-book science questions |

This is the standard panel used for small decoder LMs across the literature,
so our numbers can be compared with published results.

HSKM is evaluated through a **custom adapter** (`src/hskm_lm.py`) because it is
not a `transformers` model. The adapter implements the same loglikelihood
contract as `HFLM`, and `tests/test_benchmark.py::test_hskm_adapter_matches_gpt2`
asserts the two agree to within 1e-3 on the same underlying model — so HSKM's
numbers are on the same scale as the baselines'.

### 3.3 Efficiency — the O(N) claim

This is the experiment the project actually needs, so it is run carefully.

For each model and each sequence length we measure:

* forward+backward wall-clock time per step (the **full step**),
* forward time for the **body only** (excluding the LM head),
* peak GPU memory,
* training throughput (tokens/s),

and fit a least-squares slope of `log(metric)` against `log(seq_len)`. The
slope `b` is the **empirical scaling exponent**: `b ≈ 1` for linear, `b ≈ 2`
for quadratic attention.

**Why the body-only measurement matters.** The LM head produces a
`[B, T, vocab]` tensor whose cost is `O(N)` for *every* model. Including it
in the measurement adds a linear term that dilutes the measured exponent of a
quadratic model towards 1 — which would *flatter* the attention baselines.
Measuring the body alone (the sequence-mixing stack, with no LM head) gives
the exponent the O(N) claim is actually about. Both numbers are reported.

**Architecture, not weights.** The attention baselines (GPT-2, Pythia) are
built from config with random weights and a position table widened to cover
the sweep. A scaling study measures the architecture; loading a pretrained
checkpoint would instead measure the *training context* its learned absolute
position embeddings were sized for (2048 for Pythia), capping the sweep for
a reason unrelated to complexity. HSKM and Mamba are
position-independent by construction. Note that GPT-2-style absolute
positions *are* a genuine handicap at long context — which is exactly why
the field moved to RoPE, and one more reason HSKM's RoPE-in-kernel-attention
design is interesting.

We also measure **generation throughput versus context length**, which
separates the asymptotic claim from the practical one (see §5).

**Methodological corrections relative to upstream `benchmark.py`:**

| Upstream | Here | Why |
|---|---|---|
| Baseline is `nn.TransformerEncoderLayer` — a **bidirectional encoder** layer | Baseline is the real causal `GPT2LMHeadModel` (plus Pythia and Mamba) | An encoder layer is not a valid decoder baseline; the comparison was not measuring the right thing. |
| `results` list is never appended to, so the script writes an **empty JSON** | Results are recorded and saved | The upstream script produced no machine-readable output. |
| Long sequence lengths measured with a **numerically broken** model | Numerically stable EMA (see §5) | Upstream HSKM returns NaN at ≥1024 tokens, so the long-context rows were meaningless. |
| Single GPU, single model at a time | Same dtype for all models, OOM recorded rather than crashing | Fairness and completeness of the sweep. |

**Training precision.** The controlled training uses fp16 autocast with a
`GradScaler` (as the upstream `train.py` also does), which is ~2.6× faster on
a T4: HSKM reaches 16.5k tok/s and the GPT-2 control 15.0k tok/s. The stable
EMA forces float32 internally, because under autocast the preceding `Linear`
emits float16 and the factor `decay ** -C` (4.9e11) overflows float16's
65504 range — silently producing inf/NaN. A test asserts that mixed
precision and full precision agree.

### 3.4 Model and configuration details

`benchmarks/scripts/05_dump_configs.py` writes
[`results/configs/model_configs.md`](results/configs/model_configs.md) — a
table of the resolved hyperparameters of every model in the registry
(architecture family, hidden size, depth, heads, FFN width, vocabulary,
context window, normalisation and positional scheme), plus the exact
configuration and training budget of the two from-scratch checkpoints. This
lets a reader confirm that the comparison is between the models claimed
without re-running anything.

---

## 4. Reproducing

### Requirements

```bash
pip install -r benchmarks/requirements.txt
```

The suite was developed and run on 2× NVIDIA Tesla T4 (15 GB each),
PyTorch 2.11 + CUDA 13.0, `transformers` 5.20.2, `lm-eval` 0.4.13.

### One command

```bash
bash benchmarks/run_all.sh              # full run
QUICK=1 bash benchmarks/run_all.sh      # fast smoke run (minutes)
```

### Step by step

```bash
# 0. verify the harness itself
python3 -m pytest benchmarks/tests -q

# 1. fetch checkpoints
python3 benchmarks/scripts/01_download_models.py

# 2. controlled from-scratch training (HSKM vs GPT-2, identical budget)
python3 benchmarks/scripts/10_train_controlled.py --model both --tokens 100000000

# 3. standard downstream tasks
python3 benchmarks/scripts/20_run_lm_eval.py --all

# 4. held-out perplexity
python3 benchmarks/scripts/30_perplexity.py --models all --context 512

# 5. scaling behaviour
python3 benchmarks/scripts/40_scaling_benchmark.py

# 6. tables + figures
python3 benchmarks/scripts/50_make_report.py
```

Outputs land in `benchmarks/results/`; the generated report is
[`benchmarks/results/REPORT.md`](results/REPORT.md).

### Configuration

| Knob | Where | Default |
|---|---|---|
| Training corpus | `10_train_controlled.py --dataset` | `wikitext103` |
| Token budget | `--tokens` | 100,000,000 |
| Sequence length | `--seq-len` | 512 |
| Batch size | `--batch-size` | 16 |
| Learning rate | `--lr` | 6e-4 |
| HSKM width/depth | `--d-model`, `--n-layers` | 512, 8 |
| Eval context | `30_perplexity.py --context` | 512 |
| Sweep lengths | `40_scaling_benchmark.py --seq-lens` | 512…16384 |
| lm-eval batch size | `20_run_lm_eval.py --batch-size` | 8 |

---

## 5. Findings

### 5.1 Upstream HSKM is numerically unstable at ≥1024 tokens

`MediumTermMemory` evaluates the EMA recurrence with the "geometric power"
parallel-scan trick:

```python
h_scaled = h * (self.decay ** -torch.arange(T))   # <-- overflows
h_sum    = torch.cumsum(h_scaled, dim=1)
h_ema    = h_sum * (self.decay ** torch.arange(T)) * (1 - self.decay)
```

The factor `decay ** -t` grows without bound: `0.9 ** -1024 ≈ 4.6e46`, far
outside float32's ~3.4e38 range. Empirically, upstream HSKM emits **NaN
logits at sequence length ≥ 1024**:

| seq len | 256 | 512 | 768 | **1024** | **2048** |
|---|---|---|---|---|---|
| upstream HSKM | finite | finite | finite | **NaN** | **NaN** |

This directly contradicts the project's long-context claim, so it had to be
fixed before any long-context measurement could mean anything.

`src/hskm_stable.py` provides a **numerically stable, algebraically
identical** implementation: the recurrence is evaluated with a segmented
scan, so the largest intermediate factor is `O(decay ** -C)` for a chunk
length `C = 256` instead of `O(decay ** -T)`. Verified in
`tests/test_benchmark.py`:

* matches upstream **exactly** (0.0 difference) for `T ≤ 256`;
* matches upstream to ~1.4e-6 (float32 round-off) for `T = 512, 768`;
* stays finite at `T = 1024, 2048, 4096, 8192` where upstream does not.

The stable EMA is used for all training and evaluation; the upstream
implementation is left untouched in `model.py` and can still be selected with
`--unstable-hskm`.

### 5.2 HSKM's `generate()` has no KV cache

`HSKM.generate` re-runs a full forward pass for every generated token and
truncates to `max_seq_len`. So while a *forward pass* is O(N), *generating M
tokens* costs O(N·M) — the same order as a transformer **without** a KV
cache. A transformer with a cache pays O(1) per token. The scaling benchmark
measures generation throughput against context length specifically to make
this visible rather than to hide it behind the forward-pass numbers.

### 5.3 RoPE positions restart on truncation

`generate()` truncates the context to `input_ids[:, -max_seq_len:]` and the
RoPE frequencies are always indexed from position 0. When the context exceeds
`max_seq_len`, absolute positions shift and the model sees a discontinuous
position encoding. This does not affect the forward-pass benchmarks (which
never truncate) but it is a real limitation for long-context generation.

### 5.4 Measured results

See [`benchmarks/results/REPORT.md`](results/REPORT.md) for the generated
tables and figures. The headline numbers are summarised there.

---

## 6. Fairness notes

* **HSKM has no public checkpoint.** Every "HSKM vs pretrained model"
  comparison therefore mixes *architecture* with *pretraining data and
  compute*. That is why the controlled from-scratch tier exists: it is the
  only comparison in which the architecture is the sole variable. Read the
  two result tables separately; they answer different questions.
* **Tokenizers differ.** Word-level perplexity is the tokenizer-independent
  metric and should be used for cross-vocabulary comparisons.
* **Context length is equalised** in the perplexity benchmark by default.
* **HSKM is evaluated with the stable EMA**, and its controlled checkpoint is
  trained at a 512-token context (upstream's own default training length).
* **The legacy Mamba checkpoints** (`state-spaces/mamba-130m`,
  `mamba2-130m`) predate the `transformers` hub format: their `config.json`
  has no `model_type` and their weights use the original Mamba naming
  (`backbone.embedding.weight` vs `backbone.embeddings.weight`). Our loader
  rebuilds the config and applies that single rename, then asserts that
  **every** parameter received a value (no missing or unexpected keys), so a
  silent mismatch cannot occur.
* **Causality is verified, not assumed**: `tests/test_benchmark.py` checks
  that changing a future token leaves all earlier logits bit-identical.
