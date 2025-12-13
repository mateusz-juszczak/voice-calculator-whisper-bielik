import requests
import os
import tarfile
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

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

    validated_path = (
        BASE_DIR
        / "common_voice_pl"
        / "cv-corpus-23.0-2025-09-05"
        / "pl"
        / "validated.tsv"
    )

    df = pd.read_csv(validated_path, sep="\t")

    math_keywords = [
        "plus", "minus", "razy", "równa",
        "podziel", "pierwiastek", "procent"
    ]

    def is_math(sentence):
        s = str(sentence).lower()
        return any(k in s for k in math_keywords)

    math_df = df[df["sentence"].apply(is_math)]

    print(f"Znaleziono {len(math_df)} nagrań matematycznych")


download_common_voice_if_needed()
extract_if_needed()
validate_data()

