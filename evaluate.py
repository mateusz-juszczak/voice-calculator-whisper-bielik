import pandas as pd
import os
import datetime
import math
import re
from tabulate import tabulate

# Importujemy konfigurację
import config

# ======================================================
# USTAWIENIA PLIKÓW
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "final_results.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "experiments_history.csv")

# ======================================================
# FUNKCJE POMOCNICZE
# ======================================================

def normalize_text(text):
    """Czyści tekst do porównania."""
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compare_math(calc, target):
    """Porównuje liczby z tolerancją."""
    try:
        c_val = float(str(calc).replace(',', '.'))
        t_val = float(str(target).replace(',', '.'))
        return abs(c_val - t_val) < 0.01
    except:
        return False

def clean_for_csv(text):
    """
    Usuwa entery i znaki nowej linii, aby nie zepsuć pliku CSV.
    """
    if not text: return "BRAK"
    clean = str(text).replace("\n", " ").replace("\r", " ")
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

# ======================================================
# GŁÓWNA LOGIKA
# ======================================================

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"[BŁĄD] Nie znaleziono pliku wyników: {RESULTS_FILE}")
        return

    try:
        df = pd.read_csv(RESULTS_FILE, sep=';', encoding='utf-8-sig')
    except:
        df = pd.read_csv(RESULTS_FILE, sep=';', encoding='utf-8')

    total_samples = len(df)
    math_correct = 0
    text_exact_match = 0
    
    for _, row in df.iterrows():
        if compare_math(row.get('Calculated'), row.get('Target_Result')):
            math_correct += 1
            
        whisper_txt = normalize_text(row.get('Whisper_Text'))
        target_txt = normalize_text(row.get('Target_Text'))
        
        if whisper_txt == target_txt and target_txt != "":
            text_exact_match += 1

    math_accuracy = (math_correct / total_samples * 100) if total_samples > 0 else 0
    text_accuracy = (text_exact_match / total_samples * 100) if total_samples > 0 else 0

    # --- ZBIERANIE DANYCH ---
    
    whisper_prompt_raw = getattr(config, 'WHISPER_PROMPT', "")
    bielik_prompt_raw = getattr(config, 'SYSTEM_PROMPT', "")

    experiment_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        
        # WYNIKI
        "Total_Samples": total_samples,
        "Math_Accuracy": round(math_accuracy, 2),
        "Text_Accuracy": round(text_accuracy, 2),
        
        # WHISPER KONFIGURACJA
        "Whisper_Model": config.WHISPER_CONFIG.get("model_size"),
        "Whisper_Compute": config.WHISPER_CONFIG.get("compute_type"),
        "Whisper_Prompt_Content": clean_for_csv(whisper_prompt_raw),
        
        # BIELIK KONFIGURACJA
        "Bielik_Model": config.BIELIK_CONFIG.get("model_name"),
        "Bielik_Temp": config.BIELIK_CONFIG["options"].get("temperature"),
        "Bielik_System_Prompt": clean_for_csv(bielik_prompt_raw)
    }

    # --- ZABEZPIECZENIE PRZED DUPLIKATAMI ---
    
    should_save = True
    if os.path.exists(HISTORY_FILE):
        try:
            old_df = pd.read_csv(HISTORY_FILE, sep=';', encoding='utf-8-sig')
            if not old_df.empty:
                # Pobieramy ostatni wiersz jako słownik
                last_row = old_df.iloc[-1].to_dict()
                
                # Sprawdzamy czy kluczowe pola są identyczne
                # Ignorujemy Timestamp
                is_duplicate = True
                for key, value in experiment_data.items():
                    if key == "Timestamp":
                        continue
                        
                    # Pobieramy starą wartość, konwertujemy na string dla bezpiecznego porównania
                    # (bo pandas może wczytać liczby jako int/float, a my mamy np. float)
                    old_val = str(last_row.get(key, ""))
                    new_val = str(value)
                    
                    if old_val != new_val:
                        is_duplicate = False
                        break
                
                if is_duplicate:
                    print("\n" + "!"*50)
                    print("[INFO] Wykryto duplikat! Ostatni wpis w historii ma identyczną")
                    print("       konfigurację i wyniki. Pomijam zapis do pliku.")
                    print("!"*50)
                    should_save = False

        except Exception as e:
            print(f"[WARN] Nie udało się sprawdzić duplikatów (błąd odczytu historii): {e}")

    # --- RAPORT TERMINALOWY (Zawsze wyświetlamy) ---
    print("\n" + "="*50)
    print(" RAPORT EWALUACJI")
    print("="*50)
    print(f"Data: {experiment_data['Timestamp']}")
    print("-" * 50)
    print(f"MATEMATYKA:  {math_accuracy:.2f}%")
    print(f"TRANSKRYPCJA: {text_accuracy:.2f}%")
    print("-" * 50)
    print(f"Whisper Prompt: {experiment_data['Whisper_Prompt_Content'][:60]}...")
    print(f"Bielik Prompt:  {experiment_data['Bielik_System_Prompt'][:60]}...")
    print("="*50)

    # --- ZAPIS DO PLIKU (Tylko jeśli nie duplikat) ---
    if should_save:
        new_row_df = pd.DataFrame([experiment_data])
        
        write_header = True
        mode = 'w'
        
        if os.path.exists(HISTORY_FILE):
            try:
                old_df_check = pd.read_csv(HISTORY_FILE, sep=';', encoding='utf-8-sig')
                if list(new_row_df.columns) == list(old_df_check.columns):
                    mode = 'a'
                    write_header = False
            except:
                pass

        new_row_df.to_csv(HISTORY_FILE, mode=mode, header=write_header, index=False, sep=';', encoding='utf-8-sig')
        print(f"[SUKCES] Zapisano nowy wpis w: {HISTORY_FILE}")

if __name__ == "__main__":
    main()