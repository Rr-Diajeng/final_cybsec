# Personal Implementation Tracking
## LLM-Powered Phishing Email Detection System

**Course:** Cybersecurity Final Project  
**Author:** Fadhil Rasyid Pratama  
**Date:** June 2026  

---

## 1. System Overview

This project implements an end-to-end pipeline for detecting phishing emails using a fine-tuned Large Language Model. The system takes a raw email (sender, subject, body) as input and returns a structured classification: **PHISHING**, **SAFE**, **PROMPT_INJECTION**, or **REVIEW**.

The full pipeline consists of five stages:

```
Raw Emails (7 datasets)
        ↓
Dataset Curation & Preprocessing   (data/preprocess.py)
        ↓
QLoRA Fine-Tuning                  (fine_tuning/qlora_config.yaml)
        ↓
GGUF Export & Ollama Registration  (fine_tuning/export_to_gguf.py)
        ↓
Inference API + Guardrails         (inference_api/main.py + guardrails/)
```

**Base model:** `meta-llama/Llama-3.2-3B-Instruct`  
**Fine-tuning method:** QLoRA (Quantized Low-Rank Adaptation)  
**Serving:** Ollama (GGUF Q4_K_M) + FastAPI  
**Guardrails:** Regex-based prompt injection detection with NeMo Guardrails fallback

---

## 2. Dataset Curation

### 2.1 Source Datasets

Seven publicly available email datasets were obtained from the Kaggle dataset `naserabdullahalam/phishing-email-dataset` via `kagglehub`. Each dataset has a distinct origin and schema:

| Dataset | Source | Schema | Speciality |
|---|---|---|---|
| `phishing_email.csv` | General phishing corpus | `text_combined`, `label` | Pre-combined email text |
| `Enron.csv` | Enron email corpus | `subject`, `body`, `label` | Real corporate email |
| `Ling.csv` | Ling spam corpus | `subject`, `body`, `label` | Linguistic analysis dataset |
| `CEAS_08.csv` | CEAS 2008 spam challenge | `sender`, `subject`, `body`, `label`, `urls` | Contains URL features |
| `Nazario.csv` | Nazario phishing archive | `sender`, `subject`, `body`, `urls`, `label` | Targeted phishing |
| `Nigerian_Fraud.csv` | 419 fraud archive | `sender`, `subject`, `body`, `urls`, `label` | Advance-fee fraud |
| `SpamAssasin.csv` | SpamAssassin public corpus | `sender`, `subject`, `body`, `label`, `urls` | Widely-used benchmark |

**Label convention across all datasets:** `1 → PHISHING`, `0 → SAFE`

### 2.2 Preprocessing Pipeline

Preprocessing is implemented in `data/preprocess.py` and runs fully automatically including dataset download. The pipeline follows six steps:

**Step 1 — Download:** `kagglehub.dataset_download()` fetches all CSVs if not already present in `data/raw/`. This makes the pipeline reproducible on any machine with a Kaggle API token.

**Step 2 — Schema normalization:** Each dataset has a slightly different column layout. A `build_input_field()` function adapts to each schema and produces a unified structured string:
```
Sender: <sender> | Subject: <subject> | Body: <body>
```
Datasets without sender information (Enron, Ling, phishing_email) use a shorter format: `Subject: ... | Body: ...`

**Step 3 — Text cleaning:** A `clean_text()` function removes non-printable characters (`\x00–\x1f`, `\x7f`) and collapses excessive whitespace using regex. This is important because several datasets contain malformed or HTML-heavy email bodies.

**Step 4 — Truncation:** Email bodies are truncated to 400 words before constructing the input string. This acts as a conservative proxy for token count and ensures inputs stay within the 1024-token context window configured for training.

**Step 5 — Deduplication:** After concatenating all seven datasets, exact duplicates on the `input` column are removed. This prevents the same email appearing in both train and validation splits.

**Step 6 — Stratified split and capping:** A stratified 80/20 train/validation split is performed using `sklearn.model_selection.train_test_split` with `random_state=42`. The training set is capped at 15,000 samples and the validation set at 3,000 samples, both stratified to preserve class balance.

### 2.3 Final Dataset Statistics

| Split | Total Samples | PHISHING | SAFE | PHISHING % |
|---|---|---|---|---|
| Train | 15,000 | 7,831 | 7,169 | 52.2% |
| Val | 3,000 | 1,566 | 1,434 | 52.2% |

The class distribution is near-balanced (~52% PHISHING, ~48% SAFE), which is a healthy ratio for binary classification. No additional oversampling or undersampling was required.

### 2.4 Output Format (Alpaca SFT)

Each sample is serialized to LLaMA-Factory's Alpaca format:

```json
{
  "instruction": "Classify the following email as PHISHING or SAFE. Reply with exactly one word: PHISHING or SAFE.",
  "input": "Sender: newscientist <rssfeeds@...> | Subject: Cosmic crash of speeding jets tracked | Body: ...",
  "output": "SAFE"
}
```

The `instruction` field is kept constant across all samples. The `input` varies per email. The `output` is always exactly one word: `PHISHING` or `SAFE`. This strict single-token output design is intentional — it simplifies parsing at inference time and forces the model to commit to a hard decision.

---

## 3. Fine-Tuning Script

### 3.1 Method: QLoRA

QLoRA (Quantized Low-Rank Adaptation) was selected over full fine-tuning for two reasons:
1. The base model (`Llama-3.2-3B-Instruct`) has 3 billion parameters — full fine-tuning would require approximately 24 GB of VRAM.
2. For binary classification from a strong instruction-tuned base, only the attention projections need updating. The LoRA adapter adds far fewer trainable parameters.

The training framework is **LLaMA-Factory**, which handles tokenization, chat template application, and gradient checkpointing automatically.

### 3.2 LoRA Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `lora_rank` | 16 | Standard rank for instruction-following tasks; higher ranks risk overfitting on small datasets |
| `lora_alpha` | 32 | `alpha = 2 × rank` is the conventional scaling choice |
| `lora_dropout` | 0.05 | Light dropout; binary classification task with balanced data is low-risk |
| `lora_target` | `q_proj, v_proj` | Targeting only query and value projections keeps the adapter small (~18 MB) while capturing the most impactful attention changes |

### 3.3 Quantization Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `quantization_bit` | 4 | 4-bit NF4 quantization; reduces base model VRAM footprint to ~2 GB |
| `quantization_type` | `nf4` | Normal Float 4 outperforms FP4 for LLM weights due to its normal distribution assumption |
| `double_quantization` | `true` | Quantizes the quantization constants themselves, saving an additional ~0.4 GB |

### 3.4 Training Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Training samples | 15,000 |
| Epochs configured | 1 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 16 (effective batch = 16) |
| Learning rate | 2.0 × 10⁻⁴ |
| LR scheduler | Cosine decay |
| Warmup ratio | 0.05 (5% of steps) |
| Context length | 1,024 tokens |
| Optimizer | `adamw_torch_fused` |
| Precision | BFloat16 (Ampere GPU) |
| Seed | 42 |

The cosine learning rate schedule with 5% warmup is a standard choice for LoRA fine-tuning: it avoids large gradient updates at the beginning (when the adapter weights are random) and gradually decays toward the end to prevent overshooting the loss minimum.

### 3.5 Training Run

Training was executed from the LLaMA-Factory root directory:

```bash
llamafactory-cli train fine_tuning/qlora_config.yaml
```

The training run produced checkpoints at steps 1900, 2000, and 2100 (configurable via `save_steps: 500`). The best checkpoint was selected based on `eval_loss`, as configured with `load_best_model_at_end: true` and `metric_for_best_model: eval_loss`.

---

## 4. Evaluation Metrics

### 4.1 Training Loss Curve

The following values are extracted from `trainer_log.jsonl`:

| Step | Train Loss | Eval Loss | LR | Epoch % |
|---|---|---|---|---|
| 10 | 0.6223 | — | 1.46 × 10⁻⁶ | 0.12% |
| 50 | 0.5286 | — | 7.97 × 10⁻⁶ | 0.61% |
| 100 | 0.2365 | — | 1.61 × 10⁻⁵ | 1.22% |
| 500 | 0.0247 | 0.0624 | 8.11 × 10⁻⁵ | 2.03% |
| 1000 | 0.0191 | 0.0203 | 1.62 × 10⁻⁴ | 4.07% |
| 2050 | 0.0083 | — | 1.79 × 10⁻⁴ | 25.0% |
| 2100 | 0.0310 | 0.0099 | 1.77 × 10⁻⁴ | 25.6% |

Training loss dropped from **0.6223 at step 10** to **0.0083 at step 2050**. Eval loss similarly dropped from **0.0624 at step 500** to **0.0099 at step 2100**, confirming the model generalised and did not overfit on the training set.

### 4.2 Validation Loss

Validation evaluation was run at step 2100 (25.6% of one epoch, at which point training was concluded):

| Metric | Value |
|---|---|
| `eval_loss` | **0.009339** |
| `eval_runtime` | 195.68 seconds |
| `eval_samples_per_second` | 15.33 |
| Total epoch reached | ~25.6% of epoch 1 |

An eval loss of **0.0093** is extremely low for a classification task. This reflects both the quality of the base model (Llama-3.2-3B-Instruct already has strong language understanding) and the simplicity of the single-word output target.

### 4.3 End-to-End Inference Testing

After exporting to GGUF and registering with Ollama, the system was tested via the FastAPI `/classify` endpoint. Results from live API calls:

| Sender | Subject | Returned Label | Confidence | Guardrail | Latency | Reasoning |
|---|---|---|---|---|---|---|
| security@paypa1-alert.com | URGENT: Verify your account | **PHISHING** | 0.95 | SAFE_TO_INFER | 1967.78 ms | `(PHISHING)` |
| newsletter@github.com | Your weekly digest | **SAFE** | 0.95 | SAFE_TO_INFER | 2894.37 ms | `SAFE` |
| hr@company.com | Action required: update your details | **SAFE** | 0.95 | SAFE_TO_INFER | 176.19 ms | `SAFE` |
| test@example.com | Hello | **PROMPT_INJECTION** | 1.00 | BLOCKED | 0.08 ms | `Prompt injection attempt detected: 'Ignore all previous instructions'` |

The prompt injection case was caught by the guardrail layer before reaching the LLM — latency of **0.08 ms** confirms no Ollama round-trip occurred. The ambiguous HR email was classified as SAFE by the model, reflecting the model's judgment that an internal HR request phrased with "action required" does not meet phishing criteria.

**Debugging note:** An initial incorrect result (`label: REVIEW`, `reasoning: "("`) was traced to Ollama `stop` parameters in the Modelfile being set to `"PHISHING"` and `"SAFE"`. These caused Ollama to strip the label tokens from the model's output before returning the response. Removing the stop parameters resolved the issue and produced correct classifications.

---

## 5. Prompt Engineering Pipeline

### 5.1 Instruction Design

The instruction used for both training and inference is:

> *"Classify the following email as PHISHING or SAFE. Reply with exactly one word: PHISHING or SAFE."*

This instruction was designed with three deliberate constraints:

1. **Binary framing** — explicitly naming both valid outputs eliminates ambiguity and prevents the model from outputting synonyms ("spam", "legitimate", "harmful").
2. **Repetition of the output format** — "Reply with exactly one word: PHISHING or SAFE" appears both at the start and is implied by the training labels, creating strong reinforcement.
3. **No chain-of-thought** — the model is not asked to explain its reasoning during training. This produces a direct, low-latency single-token output.

### 5.2 Input Structure

The input field follows a consistent structured template:

```
Sender: <sender_address> | Subject: <subject_line> | Body: <truncated_body>
```

This pipe-delimited key-value format was chosen because:
- It maps naturally to how email headers appear in practice
- The delimiter `|` is unambiguous and rarely appears in normal email text
- LLMs handle structured key-value inputs well due to their prevalence in pre-training data

Datasets without sender information use a shorter format (`Subject: ... | Body: ...`) to avoid injecting the literal string `"Unknown"` as a sender, which could bias the model.

### 5.3 System Prompt at Inference

At inference time (Ollama + FastAPI), a system prompt is applied in addition to the instruction:

> *"You are an expert email security classifier. When given an email, respond with exactly one word: PHISHING or SAFE."*

This system prompt was kept in the Ollama Modelfile so it applies at the model level, independently of the API layer. The FastAPI layer also injects it per-request via the `/api/chat` payload, creating a double reinforcement of the output format constraint.

### 5.4 Prompt Pipeline Flow

```
User Input (JSON)
    ↓
build_input_text()          # formats sender/subject/body into structured string
    ↓
check_for_injection()       # guardrail — returns BLOCKED or SAFE_TO_INFER
    ↓  [if SAFE_TO_INFER]
CLASSIFICATION_PROMPT       # wraps input: "Classify ... PHISHING or SAFE.\n\n{input}"
    ↓
Ollama /api/chat             # system prompt + user prompt → model output
    ↓
regex parse                  # \bPHISHING\b or \bSAFE\b
    ↓
ClassificationResult         # label, confidence, reasoning, latency_ms
```

The regex parser at the end uses word-boundary anchors (`\b`) to correctly extract the label even if the model wraps its output in punctuation (e.g., the observed output `(PHISHING)` is correctly matched by `\bPHISHING\b`).

---

## 6. Export Pipeline and Deployment

### 6.1 LoRA Merge (Step 1)

After training, the LoRA adapter weights are merged back into the base model weights using LLaMA-Factory's export command:

```bash
llamafactory-cli export \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --adapter_name_or_path saves/llama3-phishing-qlora \
  --finetuning_type lora \
  --export_dir saves/llama3-phishing-merged \
  --export_legacy_format false
```

This produces a full merged HuggingFace model (~6.1 GB in SafeTensors format). The merge is mathematically exact: `W_merged = W_base + (B × A) × (alpha / rank)`.

### 6.2 GGUF Conversion (Step 2)

The merged model is converted to GGUF format in two sub-steps, automated by `fine_tuning/export_to_gguf.py`:

**Sub-step 2a — F16 conversion:**
```bash
python llama.cpp/convert_hf_to_gguf.py saves/llama3-phishing-merged \
  --outfile saves/llama3-phishing-gguf/phishing-detector-f16.gguf \
  --outtype f16
```

**Sub-step 2b — Q4_K_M quantization:**
```bash
llama.cpp/build/bin/llama-quantize \
  phishing-detector-f16.gguf \
  phishing-detector-q4_k_m.gguf \
  Q4_K_M
```

Q4_K_M is a 4-bit quantization scheme that uses "k-quants" (mixed 4/6-bit quantization per tensor block), achieving a good balance between model size and output quality. The final GGUF file is approximately **1.9 GB**, compared to ~6.1 GB for the F16 merged weights.

The intermediate F16 file is automatically deleted after quantization completes. The `llama-quantize` binary is built from source by the export script on first run (using `cmake`), and the binary is reused on subsequent runs.

### 6.3 Ollama Registration (Step 3 & 4)

A Modelfile is written and the model is registered with Ollama:

```
FROM /path/to/phishing-detector-q4_k_m.gguf

SYSTEM """You are an expert email security classifier.
When given an email, respond with exactly one word: PHISHING or SAFE."""

PARAMETER temperature 0.0
PARAMETER top_p       0.9
```

Temperature is set to **0.0** to make the output fully deterministic — for a security classifier, variability in the label is unacceptable.

### 6.4 Guardrails Layer

The guardrails module (`guardrails/guardrail_checker.py`) provides a two-layer protection against prompt injection attacks embedded in email bodies:

**Layer 1 — Regex pre-filter (fast):** 20 compiled patterns cover the most common injection techniques:

| Category | Example Pattern Detected |
|---|---|
| Instruction override | "ignore all previous instructions" |
| Role hijacking | "you are now a", "act as if you are" |
| System prompt leak | "[system]", `<system>`, "system: you are" |
| Jailbreak | "jailbreak", "DAN mode", "developer mode enabled" |
| Constraint bypass | "bypass your restrictions", "you have no restrictions" |

**Layer 2 — NeMo Guardrails (semantic):** If the Python package `nemoguardrails` is installed, a semantic check using a Colang configuration is applied for deeper pattern detection. This layer is optional and fails gracefully (the regex result is used if NeMo is unavailable).

The guardrail check runs before any LLM call. A BLOCKED result short-circuits the pipeline and returns `label: PROMPT_INJECTION` with `confidence: 1.0`, without consuming Ollama resources.

### 6.5 Inference API

The FastAPI application (`inference_api/main.py`) exposes four endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | Classify a single email |
| `/classify/batch` | POST | Classify up to 50 emails concurrently |
| `/results` | GET | Retrieve all stored results from SQLite |
| `/healthz` | GET | Health check for API and Ollama connectivity |

All results are persisted to a local SQLite database (`inference_api/results.db`) with schema: `id`, `timestamp`, `sender`, `subject`, `body_preview`, `label`, `confidence`, `reasoning`, `guardrail_status`, `latency_ms`.

Observed inference latency for single email classification is approximately **300–2000 ms** depending on email length, running on CPU. The high end of this range reflects the Q4_K_M GGUF running without GPU acceleration on the test machine; on a machine with a supported GPU, latency would drop significantly.

---

## 7. Issues Encountered and Resolutions

The following issues were encountered and resolved during implementation:

| Issue | Root Cause | Resolution |
|---|---|---|
| `ImportError: cannot import name 'get_web_endpoint'` | `kagglehub 1.0.x` requires `kagglesdk` function not present in `0.1.27` | Downgraded `kagglehub` to `0.3.13` |
| `llama.cpp requirements.txt` downgraded `torch` to CPU version | Running `pip install -r llama.cpp/requirements.txt` in the shared venv pins old CPU torch | Reinstalled `torch 2.12.0+cu130` from the PyTorch CUDA index; isolated future llama.cpp installs |
| `'llamafactory.train' is a package and cannot be directly executed` | Script used `-m llamafactory.train` but the correct entry point is `llamafactory-cli export` | Updated export script to use `llamafactory-cli export` |
| `convert_hf_to_gguf.py: invalid choice: 'q4_k_m'` | New llama.cpp split conversion and quantization into separate steps; `q4_k_m` removed from `--outtype` | Added two-step pipeline: `f16` conversion then `llama-quantize Q4_K_M` |
| `label: REVIEW`, `reasoning: "("` | Ollama `PARAMETER stop "PHISHING"` stripped the label from the model output before the API could read it | Removed stop parameters from Modelfile; model output `(PHISHING)` now returned intact and matched by `\bPHISHING\b` regex |
| `torchvision::nms does not exist` | `torch` was showing as `2.12.0` in `pip show` but the installed dist-info was `2.11.0+cpu` (stale metadata) | Reinstalled `torch 2.12.0+cu130` from the official PyTorch wheel index |

---

*All code, training configs, and scripts are versioned in the project repository. The LoRA adapter (`adapter_model.safetensors`, 18 MB) is tracked via Git LFS. The GGUF file (1.9 GB) is excluded from git and transferred separately.*
