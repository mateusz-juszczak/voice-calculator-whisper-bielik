import os
import re
import torch
import whisper
import pandas as pd
from pathlib import Path
import ollama
from tabulate import tabulate

# ======================================================
# TEXT NORMALIZATION
# ======================================================

def remove_trailing_dot(text: str) -> str:
    """Usuwa kropkę na końcu zdania"""
    return text.rstrip().rstrip(".")

# ======================================================
# WHISPER CONFIGURATION
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

def whisper_transcribe(audio_path: str) -> str:
    return model.transcribe(audio_path, language="pl")["text"].strip()


# ======================================================
# BIELIK CONFIGURATION
# ======================================================

def bielik_calculation(prompt: str):
    response = ollama.generate(
        model="mwiewior/bielik",
        system="Rozwiąż równanie i jako odpowiedź zwróć wyłącznie końcową liczbę.",
        prompt=prompt
    )["response"]

    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)

    if not numbers:
        return None

    value = numbers[-1]  # OSTATNIA liczba
    return float(value) if "." in value else int(value)


# ======================================================
# DATA LOADING
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
REC_BASE = BASE_DIR / "recordings"
CSV_PATH = REC_BASE / "recordings_data.csv"

df = pd.read_csv(
    CSV_PATH,
    sep=",",
    usecols=["path", "correct_transcription", "correct_solution"],
    low_memory=False
)


# ======================================================
# PROCESSING LOOP
# ======================================================

results = []

for _, row in df.iterrows():
    audio_path = REC_BASE / row["path"]

    whisper_text = whisper_transcribe(str(audio_path))
    whisper_text = remove_trailing_dot(whisper_text)

    bielik_result = bielik_calculation(whisper_text)

    results.append({
        "whisper_transcription": whisper_text,
        "correct_transcription": row["correct_transcription"],
        "bielik_solution": bielik_result,
        "correct_solution": row["correct_solution"]
    })


# ======================================================
# RESULTS AS DATAFRAME
# ======================================================

results_df = pd.DataFrame(results)

print(tabulate(
    results_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
))


# ======================================================
# OPTIONAL: SAVE TO CSV
# ======================================================

# results_df.to_csv(BASE_DIR / "results.csv", index=False)
