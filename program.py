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
import config

load_dotenv()

# ======================================================
# KONFIGURACJA
# ======================================================
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

OLLAMA_MODEL = "mwiewior/bielik"

print("[INFO] Ładowanie Whispera...")
whisper_model = WhisperModel(
    config.WHISPER_CONFIG["model_size"], 
    device = config.WHISPER_CONFIG["device"], 
    compute_type = config.WHISPER_CONFIG["compute_type"],
    cpu_threads = config.WHISPER_CONFIG["cpu_threads"]
)

# ======================================================
# NARZĘDZIA
# ======================================================

def remove_trailing_dot(text: str) -> str:
    return text.rstrip().rstrip(".")

def calculate_safe(expression: str):
    # Zabezpieczenie przed pustym wejściem (NaN z pandasa lub pusty string)
    if not expression or (isinstance(expression, float) and math.isnan(expression)):
        return "BŁĄD"

    # 1. Czyszczenie wstępne
    expr = str(expression).lower().strip()
    
    # Usuwamy typowe "gadanie" modelu
    expr = expr.replace("wynik:", "").replace("wzór:", "").replace("odpowiedź:", "")
    expr = expr.replace("expression:", "").replace("code:", "")
    
    # Bierzemy tylko pierwszą linię (jeśli model dodał coś po enterze)
    expr = expr.split("\n")[0]
    
    # Usuwamy znak równości (np. "2+2=")
    expr = expr.replace("=", "").strip()

    # 2. Tłumaczenie symboli matematycznych na Python
    # Zamiana potęgowania (^ -> **)
    expr = expr.replace("^", "**")
    
    # Zamiana przecinków na kropki (kluczowe dla ułamków w Polsce)
    expr = expr.replace(",", ".")
    
    # Obsługa pierwiastków
    # Zamiana symbolu √ na funkcję sqrt(
    expr = expr.replace("√", "sqrt(")
    # Zamiana math.sqrt na samo sqrt (żeby ujednolicić)
    expr = expr.replace("math.sqrt", "sqrt")

    # 3. Zamykanie nawiasów (Bielik czasem o tym zapomina)
    open_count = expr.count("(")
    close_count = expr.count(")")
    if open_count > close_count:
        expr += ")" * (open_count - close_count)

    # 4. Definicja dozwolonych funkcji
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": math.pow,
        "abs": abs,
        "round": round
    }

    try:
        # Próbujemy obliczyć
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        
        # Formatowanie wyniku (zaokrąglamy floaty do 2 miejsc)
        if isinstance(result, (int, float)):
            return round(result, 2)
        return result
    except Exception:
        # Jeśli mimo naprawy coś jest nie tak, zwracamy BŁĄD
        return "BŁĄD"

# ======================================================
# WHISPER
# ======================================================

def whisper_transcribe(audio_path: str) -> str:
    # Dodajemy parametr initial_prompt z configu
    segments, _ = whisper_model.transcribe(
        audio_path, 
        language="pl",
        initial_prompt=config.WHISPER_PROMPT
    )
    return " ".join(s.text for s in segments).strip()

# ======================================================
# BIELIK (ORYGINALNY PROMPT)
# ======================================================

def bielik_parse_to_math(text: str):
    
    user_msg = config.USER_PROMPT_TEMPLATE.format(text=text)

    try:
        response = ollama.generate(
            model=config.BIELIK_CONFIG["model_name"], 
            system=config.SYSTEM_PROMPT,  # <-- Prompt z configu
            prompt=user_msg,
            options=config.BIELIK_CONFIG["options"] # <-- Opcje (temperatura 0.0) z configu
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
    DATA_DIR = BASE_DIR / "eval"

    # Szukamy konkretnie tego pliku
    TARGET_FILE = "reference_data_bezpl_utf8.csv"
    CSV_PATH = DATA_DIR / TARGET_FILE

    if not CSV_PATH.exists():
        print(f"[BŁĄD] Nie znaleziono pliku: {CSV_PATH}")
        # Awaryjnie szukamy innego csv
        others = list(DATA_DIR.glob("*.csv"))
        if others:
            CSV_PATH = others[0]
            print(f"[INFO] Używam innego znalezionego pliku: {CSV_PATH.name}")
        else:
            return

    print(f"[INFO] Przetwarzam: {CSV_PATH.name}")

    try:
        # WCZYTYWANIE: Wymuszamy średnik jako separator
        df = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8', dtype=str)
        
        # 1. Normalizacja nagłówków (zamiana na małe litery i usunięcie spacji)
        # Dzięki temu 'Path' zamieni się na 'path', a 'Reference_solution' na 'reference_solution'
        df.columns = df.columns.str.lower().str.strip()
        
    except Exception as e:
        print(f"[BŁĄD KRYTYCZNY] Nie udało się otworzyć CSV: {e}")
        return

    # Sprawdzenie czy mamy kluczową kolumnę 'path'
    if "path" not in df.columns:
        print("[BŁĄD] Plik CSV nie ma kolumny 'Path' (lub 'path').")
        print(f"Znalezione kolumny: {list(df.columns)}")
        return

    results = []
    print(f"[INFO] Analiza {len(df)} wierszy...")

    for i, row in df.iterrows():
        filename = str(row["path"]).strip()
        
        audio_path = DATA_DIR / filename
        
        if not audio_path.exists():
            # Opcjonalnie wyświetl błąd, jeśli pliku brakuje
            # print(f"[MISSING] Brak pliku audio: {filename}")
            continue

        print(f"[{i+1}/{len(df)}] {filename}")

        # 1. Transkrypcja Whisper
        whisper_text = whisper_transcribe(str(audio_path))
        whisper_text = remove_trailing_dot(whisper_text)

        # 2. Parsowanie Bielik
        bielik_expr = bielik_parse_to_math(whisper_text)
        bielik_expr = bielik_expr.replace("<s>", "").replace("</s>", "").strip()
        
        # 3. Obliczenie wyniku
        calc_result = calculate_safe(bielik_expr)

        # Pobieranie poprawnego rozwiązania z CSV
        # Szukamy kolumny 'reference_solution' (bo tak nazywa się w Twoim pliku)
        ground_truth = row.get("reference_solution", "")
        if pd.isna(ground_truth): 
            ground_truth = ""
        
        # Opcjonalnie: Referencyjna transkrypcja też się przyda do porównań
        ref_transcription = row.get("reference_transcription", "")

        results.append({
            "File": filename,
            "Whisper_Text": whisper_text,
            "Bielik_Expr": bielik_expr,
            "Calculated": calc_result,
            "Target_Result": ground_truth,
            "Target_Text": ref_transcription
        })

    if not results:
        print("[OSTRZEŻENIE] Brak wyników (sprawdź czy nazwy plików w CSV zgadzają się z plikami w folderze eval)")
        return

    results_df = pd.DataFrame(results)

    # Wyświetlanie tabeli (wybrane kolumny dla czytelności)
    print("\n=== WYNIKI ===")
    print(tabulate(
        results_df[["Whisper_Text", "Bielik_Expr", "Calculated", "Target_Result"]],
        headers="keys",
        tablefmt="grid",
        showindex=False
    ))

    out_path = BASE_DIR / "final_results.csv"
    results_df.to_csv(out_path, index=False, sep=';') # Zapisujemy też ze średnikiem, żeby Excel łatwo otworzył
    print(f"\n[SUKCES] Zapisano raport: {out_path}")

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
