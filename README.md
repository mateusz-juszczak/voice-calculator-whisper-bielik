Bielik Voice Calculator
=======================

Profesjonalny kalkulator głosowy w Pythonie:
- faster-whisper – rozpoznawanie mowy po polsku
- Bielik (mwiewior/bielik) przez Ollama – parser matematyczny
- Tkinter – GUI z miernikiem VU

Wymagania
---------
- Python 3.10 – 3.12
- Windows / Linux / macOS
- Mikrofon
- Minimum 8 GB RAM

Instalacja
----------
1. Sklonuj repozytorium lub pobierz ZIP:
   git clone https://github.com/mateusz-juszczak/voice-calculator-whisper-bielik.git
   cd bielik-voice-calculator

2. Zainstaluj potrzebne biblioteki:
   pip install -r requirements.txt

Instalacja modelu Bielik
------------------------
1. Zainstaluj Ollama: https://ollama.com
   Sprawdź:
   ollama --version

2. Uruchom serwer:
   ollama serve

3. Pobierz model Bielik:
   ollama pull mwiewior/bielik

4. Test:
   ollama run mwiewior/bielik
   Jeśli model odpowiada po polsku – gotowe.

Uruchomienie aplikacji
----------------------
python program.py

Przykład użycia
---------------
1. Wybierz mikrofon z listy i kliknij 'Zatwierdź urządzenie
2. Naciśnij zielony przycisk i wypowiedz działanie matematyczne, np. "dwa do potęgi trzeciej plus pierwiastek z dziewięciu"
3. Kliknij STOP, aby obliczyć wynik", przykładowy wynik 11

Najczęstsze problemy
-------------------
- Connection refused: sprawdź, czy działa `ollama serve`
- Model nie znaleziony: wykonaj `ollama pull mwiewior/bielik`
- Brak modułu ollama: pip install ollama
- Brak dźwięku: sprawdź mikrofon w ustawieniach aplikacji

Licencja
--------
Model Bielik: osobna licencja autora (pobierany lokalnie przez Ollama)
