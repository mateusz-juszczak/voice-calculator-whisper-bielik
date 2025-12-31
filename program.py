import os
import re
import io
import math
import pandas as pd
from pathlib import Path
from tabulate import tabulate
import ollama
from faster_whisper import WhisperModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

# ======================================================
# KONFIGURACJA MODELI
# ======================================================

print("[INFO] Ładowanie modelu Whisper (CPU/int8)...")
# Używamy faster-whisper na CPU.
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Nazwa modelu w Ollama (musi być zgodna z 'ollama list')
OLLAMA_MODEL = "mwiewior/bielik" 

# ======================================================
# NARZĘDZIA POMOCNICZE
# ======================================================

def remove_trailing_dot(text: str) -> str:
    """Usuwa kropkę na końcu zdania."""
    if not text: return ""
    return text.rstrip().rstrip(".")

def calculate_safe(expression: str):
    """
    Inteligentne obliczanie, które radzi sobie z 'śmieciami' od LLMa.
    """
    if not expression:
        return None
        
    allowed_names = {"math": math, "sqrt": math.sqrt, "pow": math.pow, "abs": abs}
    
    try:
        # 1. Usuwamy znaki specjalne modelu (<s>, </s>)
        clean_expr = expression.replace("<s>", "").replace("</s>", "")
        
        # 2. Jeśli model zwrócił "x = 2+2", bierzemy tylko to co po prawej stronie "="
        if "=" in clean_expr:
            clean_expr = clean_expr.split("=")[-1]
            
        # 3. Usuwamy wszelkie litery (np. "Result"), zostawiamy tylko matematykę
        # To Regex: zostaw tylko cyfry, kropki, nawiasy i operatory
        # Uwaga: usuwa też "math.sqrt", więc musimy być ostrożni.
        # Zamiast usuwać litery, po prostu usuńmy popularne "śmieci":
        clean_expr = clean_expr.replace("Result", "").replace("wynik", "").strip()

        # 4. Obliczenie
        result = eval(clean_expr, {"__builtins__": {}}, allowed_names)
        
        if isinstance(result, float):
            return round(result, 2)
        return result
        
    except Exception as e:
        # Pokaż błąd w terminalu, żebyśmy wiedzieli co poprawić
        # print(f"[DEBUG] Nie udało się obliczyć '{expression}' -> '{clean_expr}': {e}")
        return "BŁĄD" # Zwracamy string, żeby tabela się nie sypała

# ======================================================
# GŁÓWNE FUNKCJE LOGICZNE
# ======================================================

def whisper_transcribe_faster(audio_path: str) -> str:
    """Transkrypcja audio -> tekst"""
    if not os.path.exists(audio_path):
        return ""
    try:
        segments, _ = whisper_model.transcribe(audio_path, language="pl")
        text = " ".join([s.text for s in segments]).strip()
        return text
    except Exception as e:
        print(f"[ERROR] Błąd Whispera: {e}")
        return ""

def bielik_parse_to_math(text: str):
    """
    Zamienia tekst na kod, z wymuszeniem precyzji (temperature=0).
    """
    system_prompt = (
        "Jesteś matematycznym parserem. Zamień polski tekst na wyrażenie matematyczne w Pythonie. "
        "Zasady:\n"
        "1. Zwróć WYŁĄCZNIE liczby i operatory (np. '2 + 2').\n"
        "2. NIE używaj słów 'Result', 'wynik', 'równa się'.\n"
        "3. NIE używaj znaku '='.\n"
        "4. Jeśli słyszysz liczebniki porządkowe (np. piątych), traktuj je jak ułamki lub liczby."
    )
    
    user_prompt = f"Tekst: {text}\nWyrażenie:"

    try:
        response = ollama.generate(
            model=OLLAMA_MODEL, 
            system=system_prompt,
            prompt=user_prompt,
            options={
                "temperature": 0.0, # <--- KLUCZOWA ZMIANA: Model staje się deterministyczny
                "num_predict": 50   # Ograniczamy długość wypowiedzi
            }
        )["response"]

        # Wyciąganie kodu z ewentualnych bloków markdown
        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        if code_match:
            expression = code_match.group(1).strip()
        else:
            expression = response.strip()
            
        return expression
        
    except Exception as e:
        print(f"[ERROR] Błąd Ollama: {e}")
        return ""

# ======================================================
# GŁÓWNA PĘTLA PROGRAMU
# ======================================================

def main():
    BASE_DIR = Path(__file__).resolve().parent
    REC_BASE = BASE_DIR / "recordings"
    CSV_PATH = REC_BASE / "recordings_data.csv"

    if not CSV_PATH.exists():
        print(f"[BLAD] Nie znaleziono pliku: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, sep=",", low_memory=False)
    
    # Opcjonalnie: test na pierwszych 5 rekordach
    # df = df.head(5)

    results = []
    print(f"[INFO] Przetwarzanie {len(df)} nagrań...")

    for i, row in df.iterrows():
        audio_path = REC_BASE / row["path"]
        
        if not audio_path.exists():
            continue

        print(f"[{i+1}/{len(df)}] Analiza: {row['path']}")

        # 1. Transkrypcja
        transcription = whisper_transcribe_faster(str(audio_path))
        transcription = remove_trailing_dot(transcription)

        # 2. Parsowanie (Bielik)
        math_expr = bielik_parse_to_math(transcription)
        
        # 3. OBLICZENIE (To o co prosiłeś)
        # Funkcja bierze string "2+2" i robi z niego liczbę 4
        calc_result = calculate_safe(math_expr)

        results.append({
            "original_path": row["path"],
            "whisper_text": transcription,
            "bielik_expression": math_expr,
            "calculated_result": calc_result,  # <-- Tu będzie liczba
            "ground_truth_solution": row.get("correct_solution", "")
        })

    # Tworzenie tabeli wyników
    results_df = pd.DataFrame(results)
    
    # Wyświetlenie w terminalu (tylko najważniejsze kolumny)
    print("\n=== WYNIKI ===")
    print(tabulate(
        results_df[["whisper_text", "bielik_expression", "calculated_result"]],
        headers="keys",
        tablefmt="grid",
        showindex=False,
        missingval="BŁĄD" # Wyświetli "BŁĄD" jeśli calc_result to None
    ))
    
    # Zapis do CSV
    output_path = BASE_DIR / "final_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[INFO] Zapisano wyniki w: {output_path}")

if __name__ == "__main__":
    main()
    