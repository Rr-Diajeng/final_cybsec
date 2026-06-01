# Phishing Email Detection System
**Group 19 · M11401854 · Rr. Diajeng Alfisyahrinnisa Anandha**

End-to-end LLM-powered phishing detection using Llama 3.2, QLoRA, NeMo Guardrails,
FastAPI, and a Vite/React dashboard.

---

## Project Structure

```
phishing_system/
├── data/
│   ├── preprocess.py          ← Step 1: convert CSVs → alpaca JSON
│   └── raw/                   ← Put all 7 CSVs here
│       ├── phishing_email.csv
│       ├── Enron.csv
│       ├── Ling.csv
│       ├── CEAS_08.csv
│       ├── Nazario.csv
│       ├── Nigerian_Fraud.csv
│       └── SpamAssasin.csv
├── fine_tuning/
│   ├── qlora_config.yaml      ← Step 2: QLoRA training config
│   ├── dataset_info.json      ← Register datasets in LLaMA-Factory
│   └── export_to_gguf.py      ← Step 3: merge + quantize + Ollama
├── guardrails/
│   ├── config/
│   │   ├── config.yml         ← NeMo config
│   │   └── rails.co           ← Colang injection patterns
│   └── guardrail_checker.py   ← Python guardrail module
├── inference_api/
│   └── main.py                ← FastAPI server (port 8000)
├── dashboard/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       └── App.jsx            ← React dashboard
├── scripts/
│   └── run_all.py             ← Start all services
└── requirements.txt
```

---

## Dataset Label Convention

| Value | Meaning   |
|-------|-----------|
| `1`   | PHISHING  |
| `0`   | SAFE      |

All 7 CSVs use integer labels. The preprocessing script handles both schema variants:
- **With sender/receiver/date**: CEAS_08, Nazario, Nigerian_Fraud, SpamAssasin
- **Without sender**: Enron, Ling
- **Pre-combined text**: phishing_email (column `text_combined`)

---

## Setup Guide

### 0. Prerequisites

```bash
# Python 3.10+
python3 --version

# CUDA-enabled GPU (required for QLoRA)
nvidia-smi

# Node.js 18+ (for dashboard)
node --version

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 1. Install Python dependencies

```bash
cd phishing_system
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare raw data

Copy all 7 CSV files into `data/raw/`:

```bash
mkdir -p data/raw
cp /path/to/your/csvs/*.csv data/raw/
```

### 3. Run preprocessing

```bash
python data/preprocess.py
```

This produces:
- `data/phishing_train.json`  (~80% of data, stratified)
- `data/phishing_val.json`    (~20% of data)

### 4. Set up LLaMA-Factory

```bash
# In the PARENT directory of phishing_system:
python3 -m venv llamafactory-env && source llamafactory-env/bin/activate
git clone https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes]"

# Copy the generated JSON files into LLaMA-Factory's data/ directory
cp ../phishing_system/data/phishing_train.json data/
cp ../phishing_system/data/phishing_val.json   data/

# Register the datasets (merge into existing dataset_info.json)
# Copy the contents of phishing_system/fine_tuning/dataset_info.json
# into LLaMA-Factory/data/dataset_info.json

# Request the base model from HuggingFace (requires HF token):
huggingface-cli login
```

### 5. Run QLoRA fine-tuning

```bash
# From inside LLaMA-Factory/ with llamafactory-env active:
llamafactory-cli train ../phishing_system/fine_tuning/qlora_config.yaml
```

Training checkpoints saved to `saves/llama3-phishing-qlora/`.

### 6. Export to GGUF and register with Ollama

```bash
# Clone llama.cpp alongside LLaMA-Factory:
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

# Run export:
python phishing_system/fine_tuning/export_to_gguf.py
```

This will:
1. Merge LoRA weights into the base model
2. Convert to GGUF Q4_K_M format
3. Write an Ollama Modelfile
4. Register `phishing-detector` with Ollama

### 7. Start the inference API

```bash
# From phishing_system/ with venv active:
uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "email": {
      "sender": "security@paypa1-alert.com",
      "subject": "URGENT: Verify your account",
      "body": "Click here immediately to verify your credentials or your account will be closed."
    }
  }'
```

### 8. Start the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open: http://localhost:5173

---

## API Reference

| Endpoint             | Method | Description                      |
|----------------------|--------|----------------------------------|
| `/classify`          | POST   | Classify one email               |
| `/classify/batch`    | POST   | Classify up to 50 emails         |
| `/results`           | GET    | Get recent results (dashboard)   |
| `/healthz`           | GET    | Health check                     |
| `/docs`              | GET    | Interactive Swagger UI           |

### Example request

```json
POST /classify
{
  "email": {
    "sender":  "noreply@bank-security.net",
    "subject": "Account verification required",
    "body":    "Dear customer, your account has been flagged..."
  }
}
```

### Example response

```json
{
  "id":               "3f2a1b9c-...",
  "timestamp":        "2026-05-25T10:30:00Z",
  "sender":           "noreply@bank-security.net",
  "subject":          "Account verification required",
  "label":            "PHISHING",
  "confidence":       0.95,
  "reasoning":        "PHISHING",
  "guardrail_status": "SAFE_TO_INFER",
  "latency_ms":       312.5
}
```

Label values:
- `PHISHING` — email is a phishing attempt
- `SAFE` — email is legitimate
- `PROMPT_INJECTION` — injection attack detected in email body (blocked before LLM)
- `REVIEW` — ambiguous; route to human review

---

## n8n Workflow Configuration

1. Add a **Schedule Trigger** node (interval: 1 minute)
2. Add a **Gmail** node — configure OAuth2, set operation to "Get Many", filter `is:unread`
3. Add a **HTTP Request** node:
   - Method: `POST`
   - URL: `http://localhost:8000/classify`
   - Body (JSON):
     ```json
     {
       "email": {
         "sender":  "={{ $json.from }}",
         "subject": "={{ $json.subject }}",
         "body":    "={{ $json.text }}"
       }
     }
     ```
4. Add an **If** node to branch on `{{ $json.label }}` being `PHISHING` or `PROMPT_INJECTION`
5. (Optional) Add a **Gmail** node to label/move flagged emails

---

## Model Hyperparameters Summary

| Hyperparameter             | Value                        |
|----------------------------|------------------------------|
| Base model                 | meta-llama/Llama-3.2-3B-Instruct |
| Fine-tuning method         | QLoRA (SFT)                  |
| LoRA rank                  | 16                           |
| LoRA alpha                 | 32                           |
| LoRA target modules        | q_proj, v_proj               |
| Quantization               | 4-bit NF4                    |
| Learning rate              | 2e-4                         |
| Batch size (effective)     | 16 (4 × 4 grad accum)        |
| Epochs                     | 3                            |
| Max sequence length        | 2048 tokens                  |
| Optimizer                  | AdamW + cosine schedule      |
| Export format              | GGUF Q4_K_M                  |
