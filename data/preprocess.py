"""
preprocess.py
=============
Converts the 7 raw CSV files (CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud,
SpamAssasin, phishing_email) into LLaMA-Factory alpaca-format JSON files for
Supervised Fine-Tuning (SFT).

Label convention in every CSV:
  1  → PHISHING / spam
  0  → SAFE    / ham / legitimate

Dataset schemas confirmed by EDA:
  phishing_email.csv   : text_combined, label
  Enron.csv            : subject, body, label
  Ling.csv             : subject, body, label
  CEAS_08.csv          : sender, receiver, date, subject, body, label, urls
  Nazario.csv          : sender, receiver, date, subject, body, urls, label
  Nigerian_Fraud.csv   : sender, receiver, date, subject, body, urls, label
  SpamAssasin.csv      : sender, receiver, date, subject, body, label, urls

Output:
  data/phishing_train.json  (80 %)
  data/phishing_val.json    (20 %)
"""

import json
import logging
import re
import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR   = Path(__file__).parent / "raw"          # CSVs are downloaded here
OUT_DIR   = Path(__file__).parent                  # output JSON goes here
MAX_TOKENS = 400          # approx. word-level truncation (conservative)
RANDOM_SEED = 42
TEST_SIZE   = 0.20
# ──────────────────────────────────────────────────────────────────────────────

INSTRUCTION = (
    "Classify the following email as PHISHING or SAFE. "
    "Reply with exactly one word: PHISHING or SAFE."
)


def truncate(text: str, max_words: int = MAX_TOKENS) -> str:
    """Truncate to max_words words (fast proxy for token count)."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def clean_text(text: str) -> str:
    """Strip excessive whitespace and non-printable characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_field(row: pd.Series, source: str) -> str:
    """
    Build a structured input string from a row, adapting to each dataset schema.
    Always returns:  Sender: ... | Subject: ... | Body: ...
    """
    sender  = clean_text(str(row.get("sender",  "Unknown")))
    subject = clean_text(str(row.get("subject", "No Subject")))

    if source == "phishing_email":
        # This dataset stores pre-combined text in text_combined
        body = clean_text(str(row.get("text_combined", "")))
    else:
        body = clean_text(str(row.get("body", "")))

    # Truncate only the body portion to respect context window
    body = truncate(body, MAX_TOKENS)

    if source in ("phishing_email",):
        # No sender info available; omit sender field
        return f"Subject: {subject} | Body: {body}"
    elif source in ("enron", "ling"):
        return f"Subject: {subject} | Body: {body}"
    else:
        return f"Sender: {sender} | Subject: {subject} | Body: {body}"


def load_dataset(path: Path, source: str) -> pd.DataFrame:
    """Load one CSV and return a normalised DataFrame with columns [input, label_str]."""
    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw rows: {len(df)} | columns: {df.columns.tolist()}")

    # ── normalise label ───────────────────────────────────────────────────────
    if "label" not in df.columns:
        raise ValueError(f"{path.name}: no 'label' column found")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Keep only 0 and 1
    df = df[df["label"].isin([0, 1])].copy()

    # ── drop rows with empty body / text ─────────────────────────────────────
    if source == "phishing_email":
        df = df.dropna(subset=["text_combined"])
        df = df[df["text_combined"].astype(str).str.strip() != ""]
    else:
        df = df.dropna(subset=["body"])
        df = df[df["body"].astype(str).str.strip() != ""]

    # ── build structured input ────────────────────────────────────────────────
    df["input"]     = df.apply(lambda r: build_input_field(r, source), axis=1)
    df["label_str"] = df["label"].map({1: "PHISHING", 0: "SAFE"})

    log.info(f"  After cleaning: {len(df)} rows | "
             f"PHISHING={df['label'].sum()} | SAFE={(df['label']==0).sum()}")
    return df[["input", "label_str"]].copy()


def to_alpaca(row: pd.Series) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input":       row["input"],
        "output":      row["label_str"],
    }


def save_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(records)} records → {path}")


def download_dataset() -> None:
    """
    Download the Phishing Email Dataset from Kaggle via kagglehub and copy
    all CSV files into RAW_DIR so the rest of the pipeline can find them.

    Requires either:
      • A ~/.kaggle/kaggle.json API token, OR
      • Environment variables KAGGLE_USERNAME and KAGGLE_KEY set.

    Install: pip install kagglehub
    """
    log.info("Downloading dataset from Kaggle: naserabdullahalam/phishing-email-dataset …")
    download_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    log.info(f"Path to dataset files: {download_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(download_path)

    csv_files = list(src.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in the downloaded dataset at {download_path}."
        )

    for csv_file in csv_files:
        dest = RAW_DIR / csv_file.name
        if dest.exists():
            log.info(f"  Already exists, skipping copy: {csv_file.name}")
        else:
            shutil.copy2(csv_file, dest)
            log.info(f"  Copied: {csv_file.name} → {dest}")

    log.info(f"✅ All CSVs ready in {RAW_DIR}")


def main():
    # ── Step 0: Download dataset from Kaggle (skips if CSVs already present) ──
    existing_csvs = list(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
    if len(existing_csvs) < 7:
        log.info("Raw CSV files not found (or incomplete). Downloading from Kaggle …")
        download_dataset()
    else:
        log.info(f"Found {len(existing_csvs)} CSV(s) in {RAW_DIR}, skipping download.")

    # ── Dataset catalogue ─────────────────────────────────────────────────────
    datasets = [
        ("phishing_email.csv", "phishing_email"),
        ("Enron.csv",          "enron"),
        ("Ling.csv",           "ling"),
        ("CEAS_08.csv",        "ceas"),
        ("Nazario.csv",        "nazario"),
        ("Nigerian_Fraud.csv", "nigerian_fraud"),
        ("SpamAssasin.csv",    "spamassasin"),
    ]

    frames = []
    for filename, source in datasets:
        path = RAW_DIR / filename
        if not path.exists():
            log.warning(f"File not found, skipping: {path}")
            continue
        frames.append(load_dataset(path, source))

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}. "
                                 "Place the 7 CSV files into data/raw/")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"\nCombined dataset: {len(combined)} rows | "
             f"PHISHING={combined['label_str'].eq('PHISHING').sum()} | "
             f"SAFE={combined['label_str'].eq('SAFE').sum()}")

    # ── Deduplication ─────────────────────────────────────────────────────────
    before = len(combined)
    combined = combined.drop_duplicates(subset=["input"])
    log.info(f"Deduplication: removed {before - len(combined)} duplicates")

    # ── Stratified train/val split ────────────────────────────────────────────
    train_df, val_df = train_test_split(
        combined,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=combined["label_str"],
    )
    log.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Convert to alpaca format and save ─────────────────────────────────────
    train_records = [to_alpaca(r) for _, r in train_df.iterrows()]
    val_records   = [to_alpaca(r) for _, r in val_df.iterrows()]

    save_json(train_records, OUT_DIR / "phishing_train.json")
    save_json(val_records,   OUT_DIR / "phishing_val.json")

    log.info("\n✅ Preprocessing complete.")
    log.info(f"   Train: {OUT_DIR / 'phishing_train.json'}")
    log.info(f"   Val:   {OUT_DIR / 'phishing_val.json'}")


if __name__ == "__main__":
    main()