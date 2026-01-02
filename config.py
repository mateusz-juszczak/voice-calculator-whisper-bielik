import os
import torch

DATA_SET_ID = "1urJYRBjo3TvXQlKnrZLXm-HglOlAUbpr"

WHISPER_CONFIG = {
    "model_size": "tiny",    
    "device": "cpu",    
    "compute_type": "int8",    
    "cpu_threads": 0
}


BIELIK_CONFIG = {
    "model_name": "mwiewior/bielik",    
    "options": {
        "temperature": 0.0,  # 0.0 = Model jest deterministyczny (nie zmyśla)
        "num_predict": 100,  # Maksymalna długość odpowiedzi (tokens)
        "top_k": 10,         # Ograniczenie losowości
        "top_p": 0.9
    }
}

SYSTEM_PROMPT = (
    "Jesteś matematycznym parserem. Twoim celem jest zamiana polskiego tekstu "
    "na ścisłe wyrażenie matematyczne w języku Python.\n"
    "ZASADY:\n"
    "1. Zwracaj TYLKO kod (liczby i operatory).\n"
    "2. Ignoruj słowa 'oblicz', 'wynik', 'równa się'.\n"
    "3. Zamieniaj liczebniki słowne na cyfry (np. 'jeden' -> '1').\n"
    "4. Obsługuj ułamki (np. 'pół' -> '0.5', 'jedna druga' -> '1/2').\n"
    "5. Nie dodawaj żadnych komentarzy ani znaczników markdown."
)

WHISPER_PROMPT = (
    "To jest transkrypcja matematyczna. Przykłady: "
    "jeden dodać dwa, trzy razy cztery, pięć odjąć sześć, "
    "siedem podzielić przez osiem, dziewięć do kwadratu, "
    "dziesięć do sześcianu, pierwiastek z liczby sto, jeden milion. "
    "Zero przecinek pięć. Nawias otwórz dwa plus dwa nawias zamknij. "
    "Wypisuj liczby cyframi lub słownie, ale bez zbędnych znaków interpunkcyjnych."
)

USER_PROMPT_TEMPLATE = "Tekst wejściowy: {text}\nWyrażenie Python:"

