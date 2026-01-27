import os
import math
import ollama
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import config
import re
import time
import csv

# Ładowanie zmiennych środowiskowych
load_dotenv()

# ======================================================
# INICJALIZACJA MODELI (Singleton)
# ======================================================
print("[BACKEND] Inicjalizacja modelu Whisper...")
try:
    whisper_model = WhisperModel(
        config.WHISPER_CONFIG["model_size"], 
        device=config.WHISPER_CONFIG["device"], 
        compute_type=config.WHISPER_CONFIG["compute_type"],
        cpu_threads=config.WHISPER_CONFIG["cpu_threads"]
    )
    print("[BACKEND] Whisper gotowy.")
except Exception as e:
    print(f"[ERROR] Błąd ładowania Whispera: {e}")
    whisper_model = None

# ======================================================
# LOGIKA MATEMATYCZNA
# ======================================================

def remove_trailing_dot(text: str) -> str:
    """Usuwa kropkę kończącą zdanie, która może mylić parser."""
    return text.rstrip().rstrip(".")

def clean_math_string(text: str) -> str:
    """Czyści surową odpowiedź z modelu Bielik (usuwa tagi markdown, zbędne opisy)."""
    # Usuwanie popularnych "śmieci" z LLM
    text = text.lower()
    text = re.sub(r'<(.*?)>', '', text)
    text = text.replace("`", "").replace("=", "")
    text = text.replace("wynik", "").replace("Result", "").replace("wyrażenie", "")
    text = text.replace("Oblicz", "").replace("to", "")

    
    # Bierzemy tylko pierwszą linię (często LLM dopisuje coś po wzorze)
    return text.strip().split('\n')[0]

def normalize_roots(text: str) -> str:
    text = text.lower()
    text = re.sub(r'√\s*(\d+(\.\d+)?)', r'sqrt(\1)', text)
    text = re.sub(r'pierwiastek z (\d+(\.\d+)?)', r'sqrt(\1)', text)
    return text

def calculate_safe(expression: str):
    """Bezpieczne obliczanie wyrażenia matematycznego (eval)."""
    if not expression:
        return "BŁĄD"

    # Dozwolone funkcje matematyczne
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({"abs": abs, "round": round})

    try:
        # 1. Czyszczenie stringa
        expr_clean = clean_math_string(expression)
        
        # 2. Zamiana operatorów "ludzkich" na Pythonowe (na wszelki wypadek)
        expr_clean = expr_clean.replace("^", "**").replace(",", ".")
        
        expr_clean = re.sub(r'√\s*(\d+(\.\d+)?)', r'sqrt(\1)', expr_clean)
        expr_clean = re.sub(r'pierwiastek z (\d+(\.\d+)?)', r'sqrt(\1)', expr_clean)
        
        # 3. Ewaluacja
        code = compile(expr_clean, "<string>", "eval")
        
        # Sprawdzamy czy kod nie używa niedozwolonych nazw (zabezpieczenie)
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Niedozwolona funkcja: {name}")

        # POPRAWIONO LITERÓWKĘ PONIŻEJ (__builtins__)
        result = eval(code, {"__builtins__": {}}, allowed_names)
        
        # Formatowanie wyniku
        if isinstance(result, float):
            return round(result, 2)
        return result

    except Exception as e:
        print(f"[CALC ERROR] Nie udało się obliczyć '{expression}': {e}")
        return "BŁĄD"

# ======================================================
# GŁÓWNY PIPELINE PRZETWARZANIA
# ======================================================

def process_audio_pipeline(audio_path: str):
    """
    Główna funkcja logiki.
    Wejście: ścieżka do pliku audio.
    Wyjście: krotka (transkrypcja, wzór_matematyczny, wynik_obliczenia)
    """
    if not whisper_model:
        return "Błąd modelu Whisper", "---", "ERROR"

    try:
        # KROK 1: Transkrypcja (Whisper)
        segments, _ = whisper_model.transcribe(
            audio_path, 
            language="pl",
            initial_prompt=config.WHISPER_PROMPT
        )
        # Łączenie segmentów w jeden tekst
        transcribed_text = " ".join([s.text for s in segments]).strip()
        transcribed_text = remove_trailing_dot(transcribed_text)
        
        if not transcribed_text:
            return "(Cisza)", "---", "---"

        # KROK 2: Interpretacja (Ollama / Bielik)
        response = ollama.generate(
            model=config.BIELIK_CONFIG["model_name"],
            system=config.SYSTEM_PROMPT,
            prompt=f"Tekst: {transcribed_text}\nWzór:",
            options=config.BIELIK_CONFIG["options"]
        )
        math_expr = response['response'].strip()
        math_expr_clean = clean_math_string(math_expr)

        # KROK 3: Obliczenia
        result = calculate_safe(math_expr_clean)

        return transcribed_text, math_expr_clean, str(result)

    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")
        return f"Błąd przetwarzania: {str(e)}", "ERROR", "ERROR"
    
def process_with_timing(audio_path, label):
    metrics = {
        "label": label,
        "whisper_time": 0,
        "ollama_time": 0,
        "total_time": 0
    }
    
    start_total = time.perf_counter()
    
    # Whisper
    t0 = time.perf_counter()
    segments, _ = whisper_model.transcribe(audio_path, language="pl")
    list(segments) 
    metrics["whisper_time"] = round(time.perf_counter() - t0, 4)
    
    # Ollama
    t0 = time.perf_counter()
    ollama.generate(
        model=config.BIELIK_CONFIG["model_name"],
        prompt="Tekst: 2+2\nWzór:",
        options=config.BIELIK_CONFIG["options"]
    )
    metrics["ollama_time"] = round(time.perf_counter() - t0, 4)
    
    metrics["total_time"] = round(time.perf_counter() - start_total, 4)
    return metrics

def warmup(n_iterations=3):
    print(f"[START] Rozgrzewanie systemu ({n_iterations} iteracje)...")
    for i in range(n_iterations):
        # 1. Rozgrzewka Whispera
        segments, _ = whisper_model.transcribe("warmup.wav")
        list(segments)
        
        # 2. Rozgrzewka Ollamy
        ollama.generate(model=config.BIELIK_CONFIG["model_name"], prompt="2+2")
        
        print(f"[WARMUP] Ukończono przebieg {i+1}/{n_iterations}")
    print("[READY] System gotowy do pracy.")

def run_benchmark(test_audio, output_csv="benchmark_results.csv"):
    results = []

    results.append(process_with_timing(test_audio, "COLD"))

    if os.path.exists("warmup.wav"):
        list(whisper_model.transcribe("warmup.wav")[0])
    ollama.generate(model=config.BIELIK_CONFIG["model_name"], prompt="2+2")

    for i in range(1, 6):
        results.append(process_with_timing(test_audio, f"WARM_{i}"))

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Benchmark zapisany w {output_csv}")

# if __name__ == "__main__":
#     AUDIO_FILE = "warmup.wav" 
#     if os.path.exists(AUDIO_FILE):
#         run_benchmark(AUDIO_FILE)
#     else:
#         print("Brak pliku audio do testów.")