import os
import re
import torch
import whisper
import pandas as pd
from pathlib import Path
import ollama
from tabulate import tabulate
import spacy
from jiwer import wer

# ======================================================
nlp = spacy.load("pl_core_news_lg")


# ======================================================
# TEXT NORMALIZATION
# ======================================================

def remove_trailing_dot(text: str) -> str:
    """Usuwa kropkę na końcu zdania"""
    return text.rstrip().rstrip(".")


def polish_text_to_math_expression(text: str) -> str:
    """
    Zamienia polski tekst matematyczny na poprawne wyrażenie matematyczne.
    Obsługuje:
    - dowolną długość równania
    - duże liczby (tysiące, miliony)
    - odmianę przez przypadki
    - wiele operatorów
    - liczby ujemne i dziesiętne
    """

    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc]

    numbers = {
        "zero": 0, "jeden": 1, "dwa": 2, "trzy": 3, "cztery": 4,
        "pięć": 5, "sześć": 6, "siedem": 7, "osiem": 8, "dziewięć": 9,
        "dziesięć": 10, "jedenaście": 11, "dwanaście": 12,
        "trzynaście": 13, "czternaście": 14, "piętnaście": 15,
        "szesnaście": 16, "siedemnaście": 17,
        "osiemnaście": 18, "dziewiętnaście": 19,
        "dwadzieścia": 20, "trzydzieści": 30,
        "czterdzieści": 40, "pięćdziesiąt": 50,
        "sześćdziesiąt": 60, "siedemdziesiąt": 70,
        "osiemdziesiąt": 80, "dziewięćdziesiąt": 90,
        "sto": 100, "dwieście": 200, "trzysta": 300,
        "czterysta": 400, "pięćset": 500, "sześćset": 600,
        "siedemset": 700, "osiemset": 800, "dziewięćset": 900,
    }

    multipliers = {
        "tysiąc": 1_000,
        "milion": 1_000_000
    }

    operators = {
        "plus": "+", "dodać": "+", "dodawać": "+",
        "minus": "-", "odjąć": "-",
        "razy": "*", "pomnożyć": "*",
        "podzielić": "/", "przez": "/"
    }

    decimal_words = {"przecinek", "kropka"}
    negative_words = {"minus"}

    expression = []
    current = 0
    total = 0
    decimal = ""
    is_decimal = False
    negative = False

    def flush_number():
        nonlocal current, total, decimal, is_decimal, negative
        if current == 0 and total == 0 and not decimal:
            return
        value = total + current
        if decimal:
            value = float(f"{value}.{decimal}")
        if negative:
            value = -value
            negative = False
        expression.append(str(value))
        current = total = 0
        decimal = ""
        is_decimal = False

    for lemma in lemmas:
        if lemma in negative_words and not current:
            negative = True

        elif lemma in numbers:
            current += numbers[lemma]

        elif lemma in multipliers:
            if current == 0:
                current = 1
            total += current * multipliers[lemma]
            current = 0

        elif lemma in decimal_words:
            is_decimal = True

        elif is_decimal and lemma in numbers:
            decimal += str(numbers[lemma])

        elif lemma in operators:
            flush_number()
            expression.append(operators[lemma])

    flush_number()
    return " ".join(expression)


# df["calculated_solution"] = df["math_expression"].apply(eval_calculation)

# ======================================================
# WHISPER CONFIGURATION
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

def whisper_transcribe(audio_path: str) -> str:
    transcription = model.transcribe(audio_path, language="pl")["text"].strip()
    return transcription

def wer_calculation(expression:str, transcription: str):
    return wer(expression, transcription)
    
# ======================================================
# BIELIK CONFIGURATION
# ======================================================

def bielik_calculation(prompt: str):
    response = ollama.generate(
        model="mwiewior/bielik",
        system="Rozwiąż równanie i jako odpowiedź zwróć wyłącznie końcową liczbę.",
        prompt=prompt
    )["response"]

    #jako, że bielik jest gadatliwy, to bierzemy liczbę, która pojawia się na końcu jego wypocin - czyli nasz wynik
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
        "whisper_transcription_math": polish_text_to_math_expression(whisper_text),
        "correct_transcription": row["correct_transcription"],
        "correct_transcritpion_math": polish_text_to_math_expression(row["correct_transcription"]),
        "bielik_solution": bielik_calculation(whisper_text),
        "bielik_ground_truth_solution": bielik_calculation(row["correct_transcription"]),
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
