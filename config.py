import os
import torch


DATA_SET_ID = "1urJYRBjo3TvXQlKnrZLXm-HglOlAUbpr"

WHISPER_CONFIG = {
    "model_size": "base",    
    "device": "cpu",    
    "compute_type": "int8",    
    "cpu_threads": 0
}


BIELIK_CONFIG = {
    "model_name": "mwiewior/bielik",    
    "options": {
        "temperature": 0.0,  # 0.0 = Model jest deterministyczny (nie zmyśla)
        "num_predict": 50,  # Maksymalna długość odpowiedzi (tokens)
        "stop": ["Tekst:", "Zadanie:", "Przykład"]    }
    }

SYSTEM_PROMPT = (
    "Jesteś matematycznym tłumaczem. Twoim zadaniem jest zamiana zaszumionego tekstu na wzór matematyczny. "
    "Możesz używać symboli: +, -, *, /, ^ (potęga), √ (pierwiastek). "
    "Nie musisz pisać kodu Python. Pisz po prostu wzór."
)

WHISPER_PROMPT = (
    "Przykłady liczb: 1234, 567, 1999, 2024, 10000, 1500, 100, 123456. "
    "Wzory: 1200 + 34, 5000 / 10. "
    "Zapisuj liczby cyframi: 200 (nie dwieście), 1000 (nie tysiąc)."
    )

USER_PROMPT_TEMPLATE = """Zamień tekst na wzór.

SŁOWNIK NAPRAWCZY:
'krzy', 'krzyj' -> 3
'pisztery', 'tery' -> 4
'usmyj', 'usmej' -> 8
'sześć sześć sześć', 'sześć ci', 'XVI' -> 16
'siedem naście' -> 17
'dwie ściu' -> 200
'videon' -> 1000000
'audiencie dm', 'o dźwięci', 'minu' -> -
'dosyś cianu' -> ^3
'dotać' -> +
'jedną piątom' -> (1/5)
'i pół' -> + 0.5

PRZYKŁADY:
Tekst: dwa dodać dwa
Wzór: 2 + 2

Tekst: tysiąc dwieście trzydzieści trzy plus jeden
Wzór: 1233 + 1

Tekst: cztery i pół razy sześć
Wzór: 4.5 * 6

Tekst: pierwiastek z szesnastu
Wzór: √16

Tekst: dwa do potęgi usmej
Wzór: 2 ^ 8

Tekst: trzy dosyś cianu
Wzór: 3 ^ 3

Tekst: dwadzieścia dzielone na jedną piątom
Wzór: 20 / (1/5)

Tekst: pięć audiencie dm
Wzór: 5 - 7

ZADANIE:
Tekst: {text}
Wzór:"""