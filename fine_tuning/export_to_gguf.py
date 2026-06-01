"""
export_to_gguf.py
=================
After QLoRA training completes in LLaMA-Factory:

  Step 1 — Merge LoRA adapter back into base weights (LLaMA-Factory export).
  Step 2 — Convert merged HuggingFace model to GGUF Q4_K_M (llama.cpp).
  Step 3 — Write Ollama Modelfile.
  Step 4 — Register model with Ollama.

Usage (run from the LLaMA-Factory project root):
  python phishing_system/fine_tuning/export_to_gguf.py

Requirements:
  pip install transformers
  git clone https://github.com/ggerganov/llama.cpp  (sibling directory)
  pip install -r llama.cpp/requirements.txt
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Paths — adjust if your directory layout differs ───────────────────────────
LLAMAFACTORY_ROOT = Path(__file__).parent.parent.parent   # LLaMA-Factory root
ADAPTER_DIR       = LLAMAFACTORY_ROOT / "saves" / "llama3-phishing-qlora"
MERGED_DIR        = LLAMAFACTORY_ROOT / "saves" / "llama3-phishing-merged"
GGUF_DIR          = LLAMAFACTORY_ROOT / "saves" / "llama3-phishing-gguf"
LLAMA_CPP_DIR     = LLAMAFACTORY_ROOT.parent / "llama.cpp"
MODELFILE_PATH    = GGUF_DIR / "Modelfile"
OLLAMA_MODEL_NAME = "phishing-detector"

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert email security classifier. "
    "When given an email, respond with exactly one word: PHISHING or SAFE."
)
# ──────────────────────────────────────────────────────────────────────────────


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n▶  {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def step1_merge_lora() -> None:
    """Merge LoRA adapter into base weights using LLaMA-Factory's export CLI."""
    print("\n══ Step 1: Merge LoRA adapter ════════════════════════════════")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "llamafactory.train",
        "--stage",               "sft",
        "--model_name_or_path",  BASE_MODEL,
        "--adapter_name_or_path", str(ADAPTER_DIR),
        "--finetuning_type",     "lora",
        "--export_dir",          str(MERGED_DIR),
        "--export_legacy_format", "false",
    ], cwd=LLAMAFACTORY_ROOT)
    print(f"✅ Merged model saved to: {MERGED_DIR}")


def step2_convert_to_gguf() -> None:
    """Use llama.cpp convert script to produce Q4_K_M GGUF."""
    print("\n══ Step 2: Convert to GGUF Q4_K_M ═══════════════════════════")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    gguf_out = GGUF_DIR / "phishing-detector-q4_k_m.gguf"
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        sys.exit(
            f"llama.cpp not found at {LLAMA_CPP_DIR}.\n"
            "Clone it with:  git clone https://github.com/ggerganov/llama.cpp"
        )

    run([
        sys.executable, str(convert_script),
        str(MERGED_DIR),
        "--outfile",   str(gguf_out),
        "--outtype",   "q4_k_m",
    ])
    print(f"✅ GGUF model saved to: {gguf_out}")


def step3_write_modelfile() -> None:
    """Create Ollama Modelfile that mirrors the fine-tuning chat template."""
    print("\n══ Step 3: Write Ollama Modelfile ═══════════════════════════")
    gguf_path = GGUF_DIR / "phishing-detector-q4_k_m.gguf"
    modelfile_content = f"""\
FROM {gguf_path}

# Match the system prompt used during fine-tuning exactly
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

# Inference parameters
PARAMETER temperature 0.0
PARAMETER top_p       0.9
PARAMETER stop        "PHISHING"
PARAMETER stop        "SAFE"
"""
    MODELFILE_PATH.write_text(modelfile_content)
    print(f"✅ Modelfile written to: {MODELFILE_PATH}")


def step4_register_ollama() -> None:
    """Register and test the model with Ollama."""
    print("\n══ Step 4: Register with Ollama ══════════════════════════════")
    run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)])
    print(f"\n✅ Model registered as '{OLLAMA_MODEL_NAME}'")
    print("\nRunning smoke test …")
    run([
        "ollama", "run", OLLAMA_MODEL_NAME,
        (
            "Classify the following email as PHISHING or SAFE.\n\n"
            "Sender: security@paypa1-alert.com | "
            "Subject: URGENT: Your account will be suspended | "
            "Body: Dear customer, we have detected suspicious activity. "
            "Click here immediately to verify your credentials or your account will be closed."
        ),
    ])


def main():
    step1_merge_lora()
    step2_convert_to_gguf()
    step3_write_modelfile()
    step4_register_ollama()
    print("\n🎉 Export pipeline complete. Model is ready for inference.")


if __name__ == "__main__":
    main()
