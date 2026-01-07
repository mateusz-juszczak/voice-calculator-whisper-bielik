Bielik Voice Calculator
=======================

Profesjonalny kalkulator głosowy w Pythonie z wykorzystaniem:
- faster-whisper (rozpoznawanie mowy po polsku)
- Bielik (mwiewior/bielik) przez Ollama – parser matematyczny
- Tkinter – GUI z miernikiem VU

Wymagania
---------
- Python 3.10 – 3.12
- Windows / Linux / macOS
- Minimum 8 GB RAM
- Mikrofon

Instalacja
----------
1. Sklonuj repozytorium:
   git clone https://github.com/TWOJ_LOGIN/bielik-voice-calculator.git
   cd bielik-voice-calculator

2. Utwórz wirtualne środowisko:
   python -m venv venv

   Aktywacja:
   Windows: venv\Scripts\activate
   Linux/Mac: source venv/bin/activate

3. Zainstaluj wymagane biblioteki:
   pip install -r requirements.txt

Instalacja modelu Bielik
------------------------
Model nie jest dołączony do repozytorium. Pobierz go lokalnie przez Ollama.

1. Zainstaluj Ollama: https://ollama.com
   Sprawdź w terminalu:
   ollama --version

2. Uruchom serwer Ollama:
   ollama serve
   (okno terminala musi być cały czas włączone)

3. Pobierz model Bielik:
   ollama pull mwiewior/bielik

4. Test:
   ollama run mwiewior/bielik
   Jeśli model odpowiada po polsku – instalacja zakończona.

Uruchomienie aplikacji
----------------------
python program.py

Przykład użycia
---------------
Powiedz do mikrofonu:
"dwa do potęgi trzeciej plus pierwiastek z dziewięciu"
Wynik powinien być: 11

Najczęstsze problemy
-------------------
- Connection refused: sprawdź, czy działa `ollama serve`
- Model nie znaleziony: wykonaj `ollama pull mwiewior/bielik`
- Brak modułu ollama: `pip install ollama`
- Brak dźwięku: sprawdź mikrofon w ustawieniach aplikacji

Licencja
--------
Model Bielik: osobna licencja autora (pobierany lokalnie przez Ollama)
