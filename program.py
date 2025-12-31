import os
import re
import math
import time
import tempfile
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import keyboard
import ollama
from pathlib import Path
from tabulate import tabulate
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# KONFIGURACJA
# ======================================================
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

WHISPER_MODEL = "base"
OLLAMA_MODEL = "mwiewior/bielik"

print("[INFO] Ładowanie Whispera (CPU/int8)...")
whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

# ======================================================
# NARZĘDZIA
# ======================================================

def remove_trailing_dot(text: str) -> str:
    return text.rstrip().rstrip(".")

def calculate_safe(expression: str):
    if not expression:
        return "BŁĄD"
    allowed = {"sqrt": math.sqrt, "pow": math.pow, "abs": abs}
    try:
        expr = expression.replace("<s>", "").replace("</s>", "").strip()
        if "=" in expr:
            expr = expr.split("=")[-1]
        expr = expr.replace("wynik", "").replace("Result", "").strip()
        result = eval(expr, {"__builtins__": {}}, allowed)
        return round(result, 2) if isinstance(result, float) else result
    except Exception:
        return "BŁĄD"

# ======================================================
# WHISPER
# ======================================================

def whisper_transcribe(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path, language="pl")
    return " ".join(s.text for s in segments).strip()

# ======================================================
# BIELIK (ORYGINALNY PROMPT)
# ======================================================

def bielik_parse_to_math(text: str):
    system_prompt = (
        "Jesteś matematycznym parserem. Zamień polski tekst na wyrażenie matematyczne w Pythonie. "
        "Zasady:\n"
        "1. Zwróć WYŁĄCZNIE liczby i operatory (np. '2 + 2').\n"
        "2. NIE używaj słów 'Result', 'wynik', 'równa się'.\n"
        "3. NIE używaj znaku '='.\n"
        "4. Jeśli słyszysz liczebniki porządkowe (np. piątych), traktuj je jak ułamki lub liczby."
    )
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            system=system_prompt,
            prompt=f"Tekst: {text}\nWyrażenie:",
            options={"temperature": 0.0, "num_predict": 50}
        )["response"]

        match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()
    except Exception as e:
        print(f"[ERROR] Ollama błąd: {e}")
        return ""

# ======================================================
# WYBÓR AKTYWNEGO MIKROFONU
# ======================================================

def choose_active_input_device():
    while True:
        devices = sd.query_devices()
        default_index = sd.default.device[0]
        input_devices = {}
        for idx, d in enumerate(devices):
            if d['max_input_channels'] > 0 and d['default_samplerate'] > 0:
                if d['name'] not in input_devices:
                    input_devices[d['name']] = {'index': idx, 'default': (idx == default_index)}

        if not input_devices:
            print("Nie wykryto aktywnego mikrofonu. Podłącz mikrofon i spróbuj ponownie.")
            input("Naciśnij Enter, aby spróbować ponownie...")
            continue

        print("\nDostępne aktywne mikrofony:")
        for i, (name, info) in enumerate(input_devices.items()):
            mark = " (domyślny)" if info['default'] else ""
            print(f"{i}: {name}{mark}")

        try:
            choice = int(input("Wybierz numer mikrofonu: "))
            if 0 <= choice < len(input_devices):
                selected_name = list(input_devices.keys())[choice]
                selected_index = input_devices[selected_name]['index']
                try:
                    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, device=selected_index):
                        pass
                    print(f"Wybrano mikrofon: {selected_name}")
                    return selected_index
                except Exception as e:
                    print(f"Nie można użyć mikrofonu: {e}")
                    continue
            else:
                print("Niepoprawny numer, spróbuj ponownie.")
        except ValueError:
            print("Wpisz liczbę.")

# ======================================================
# PIPELINE AUDIO
# ======================================================

def process_audio(audio_path):
    try:
        text = whisper_transcribe(audio_path)
        text = remove_trailing_dot(text)
        print(f"[WHISPER] {text}")

        expr = bielik_parse_to_math(text)
        expr = expr.replace("<s>", "").replace("</s>", "").strip()
        print(f"[BIELIK] {expr}")

        result = calculate_safe(expr)
        print(f"[WYNIK] {result}\n")
    finally:
        os.remove(audio_path)

# ======================================================
# TRYB CSV
# ======================================================

def process_csv():
    BASE_DIR = Path(__file__).resolve().parent
    REC_BASE = BASE_DIR / "recordings"
    CSV_PATH = REC_BASE / "recordings_data.csv"

    if not CSV_PATH.exists():
        print("Nie znaleziono recordings_data.csv")
        return

    df = pd.read_csv(CSV_PATH)
    results = []

    for i, row in df.iterrows():
        audio_path = REC_BASE / row["path"]
        if not audio_path.exists():
            continue

        print(f"[{i+1}/{len(df)}] {row['path']}")

        text = whisper_transcribe(str(audio_path))
        text = remove_trailing_dot(text)

        expr = bielik_parse_to_math(text)
        expr = expr.replace("<s>", "").replace("</s>", "").strip()
        calc = calculate_safe(expr)

        results.append({
            "whisper_text": text,
            "bielik_expression": expr,
            "calculated_result": calc,
            "ground_truth_solution": row.get("correct_solution", "")
        })

    results_df = pd.DataFrame(results)

    print("\n=== WYNIKI ===")
    print(tabulate(
        results_df,
        headers="keys",
        tablefmt="grid",
        showindex=False
    ))

    out = BASE_DIR / "final_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nZapisano: {out}")

# ======================================================
# STABILNA SESJA AUDIO
# ======================================================

def audio_session(device_index):
    """
    Sesja nagrywania: jeden InputStream na całą pętlę.
    Każde nagranie czyste, ESC kończy sesję.
    """
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index
        ) as stream:

            while True:
                print(f"\nNagrywanie (max {RECORD_SECONDS} s, wydaj dźwięk lub ESC kończy)")

                frames = []
                start = time.time()

                while time.time() - start < RECORD_SECONDS:
                    data, _ = stream.read(1024)
                    if np.any(data):
                        frames.append(data)
                    if keyboard.is_pressed("esc"):
                        print("Zakończono nagrywanie.")
                        return
                    time.sleep(0.01)

                if not frames:
                    print("Nic się nie nagrało, spróbuj ponownie...")
                    continue

                audio = np.concatenate(frames, axis=0)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    sf.write(f.name, audio, SAMPLE_RATE)
                    audio_path = f.name

                process_audio(audio_path)

                print("Naciśnij dowolny klawisz, aby nagrać kolejne zadanie lub ESC aby zakończyć.")
                key = keyboard.read_key()
                if key == "esc":
                    print("Zakończono nagrywanie.")
                    break

    except Exception as e:
        print(f"Błąd mikrofonu: {e}")
        return

# ======================================================
# MAIN
# ======================================================

def main():
    print("VOICE CALCULATOR – audio krokowy / CSV")

    while True:
        mode = input("Tryb (audio / csv): ").strip().lower()
        if mode in ["audio", "csv"]:
            break
        print("Nieznany tryb, wpisz 'audio' lub 'csv'.")

    if mode == "audio":
        device_index = choose_active_input_device()
        if device_index is None:
            print("Koniec programu, brak działającego mikrofonu.")
            return
        audio_session(device_index)

    elif mode == "csv":
        process_csv()

if __name__ == "__main__":
    main()
