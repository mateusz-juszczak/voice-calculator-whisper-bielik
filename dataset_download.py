import requests
import os
import tarfile
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

load_dotenv()
API_KEY = os.getenv("COMMON_VOICE_API_KEY")
if API_KEY is None:
    raise RuntimeError("Brak zmiennej środowiskowej COMMON_VOICE_API_KEY")

DATASET_ID = "cmflnuzw6srrr8hlbizqk5115"
OUTPUT_FILE = "Common_Voice_PL.tar.gz"
EXTRACT_DIR = "common_voice_pl"


def download_common_voice_if_needed():
    # 1. Sprawdzenie, czy plik już istnieje
    if os.path.exists(OUTPUT_FILE):
        print(f"[INFO] Plik {OUTPUT_FILE} już istnieje – pomijam pobieranie.")
        return

    print("[INFO] Tworzę sesję pobierania...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 2. Create Download Session
    session_url = (
        f"https://datacollective.mozillafoundation.org/api/datasets/"
        f"{DATASET_ID}/download"
    )

    response = requests.post(session_url, headers=headers)
    response.raise_for_status()

    download_token = response.json()["downloadToken"]

    print("[INFO] Pobieram dataset...")

    # 3. Download file
    download_url = (
        f"https://datacollective.mozillafoundation.org/api/datasets/"
        f"{DATASET_ID}/download/{download_token}"
    )

    with requests.get(download_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(OUTPUT_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("[INFO] Pobieranie zakończone.")


def extract_if_needed():
    if os.path.exists(EXTRACT_DIR):
        print(f"[INFO] Folder {EXTRACT_DIR} już istnieje - pomijam rozpakowywanie.")
        return

    print("[INFO] Rozpakowuję archiwum...")

    with tarfile.open(OUTPUT_FILE, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)

    print("[INFO] Rozpakowywanie zakończone.")





def validate_data():
    BASE_DIR = Path(__file__).resolve().parent
    CV_BASE = BASE_DIR / "common_voice_pl"

    cv_dir = next(CV_BASE.glob("cv-corpus-*"))
    validated_path = cv_dir / "pl" / "validated.tsv"

    df = pd.read_csv(
        validated_path,
        sep="\t",
        usecols=["path", "sentence"],
        low_memory=False
    )

    # ===== NORMALIZACJA =====
    def normalize(s: str) -> str:
        return (
            str(s)
            .split("\t")[0]
            .replace("\n", " ")
            .replace("\r", " ")
            .lower()
            .strip()
        )

    df["sentence_norm"] = df["sentence"].apply(normalize)

    # ===== OPERATORY MATEMATYCZNE =====
    MATH_OPERATORS = [
        "plus", "minus", "razy", "podziel", "dzielone",
        "równa się", "równe", "pierwiastek"
    ]

    # ===== LICZBY (SŁOWNIE + RDZENIE) =====
    NUMBER_WORDS = [
        "zero", "jeden", "dwa", "trzy", "cztery", "pięć",
        "sześć", "siedem", "osiem", "dziewięć",
        "dziesięć", "naście", "dzieścia", "dzieści",
        "sto", "tysiąc"
    ]

    # ===== DETEKCJA + POWÓD =====
    def detect_math_reasons(sentence: str):
        reasons = []

        for op in MATH_OPERATORS:
            if op in sentence:
                reasons.append(f"operator:{op}")

        for num in NUMBER_WORDS:
            if num in sentence:
                reasons.append(f"number:{num}")

        return reasons

    # ===== FILTR =====
    df["math_reasons"] = df["sentence_norm"].apply(detect_math_reasons)
    filtered_df = df[df["math_reasons"].apply(len) > 0].copy()
    filtered_df.reset_index(drop=True, inplace=True)

    # ===== KOLUMNY WYJŚCIOWE =====
    final_df = filtered_df[["path", "sentence", "math_reasons"]]
    final_df["is_math"] = ""

    # zamień listę powodów na czytelny string
    final_df["notes"] = final_df["math_reasons"].apply(
        lambda reasons: ",".join(reasons)
    )

    final_df = final_df.drop(columns=["math_reasons"])

    # ===== ZAPIS =====
    out_path = BASE_DIR / "math_test_candidates.csv"
    final_df.to_csv(out_path, index=False, sep=",")

    print("Zapisano tylko nagrania z elementami matematycznymi:")
    print(out_path)
    print(f"   Liczba rekordów: {len(final_df)}")

download_common_voice_if_needed()
extract_if_needed()
validated_data = validate_data()

