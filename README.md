# Bielik Voice Calculator

Kalkulator głosowy w Pythonie, wykorzystujący nowoczesne narzędzia AI. Posiada przejrzysty interfejs webowy zbudowany w Gradio.

-   **Rozpoznawanie mowy:** `faster-whisper` (model "base") do precyzyjnej transkrypcji języka polskiego.
-   **Parser matematyczny:** Model `mwiewior/bielik` uruchamiany przez Ollama do konwersji tekstu na wyrażenie matematyczne.
-   **Interfejs użytkownika:** `Gradio` do stworzenia nowoczesnego interfejsu webowego dostępnego w przeglądarce.

## Wymagania

-   Python 3.10 – 3.12
-   Windows / Linux / macOS
-   Mikrofon
-   Nowoczesna przeglądarka internetowa (np. Chrome, Firefox)
-   Minimum 8 GB RAM (zalecane do płynnego działania modeli AI)

## Instalacja

1.  Sklonuj repozytorium lub pobierz pliki ZIP:
    ```bash
    git clone https://github.com/mateusz-juszczak/voice-calculator-whisper-bielik.git
    cd voice-calculator-whisper-bielik
    ```

2.  Zainstaluj wymagane biblioteki Python. Zaleca się utworzenie wirtualnego środowiska.
    ```bash
    pip install -r requirements.txt
    ```
    *Jeśli plik `requirements.txt` nie istnieje, zainstalaj pakiety ręcznie:*
    ```bash
    pip install gradio ollama faster-whisper numpy
    ```
    *(Uwaga: `faster-whisper` może wymagać dodatkowych zależności jak `torch` i `ctranslate2`, które `pip` powinien zainstalować automatycznie).*

## Instalacja modelu Bielik (przez Ollama)

1.  Zainstaluj **Ollama** ze strony: [https://ollama.com](https://ollama.com)
    Po instalacji sprawdź, czy polecenie działa w terminalu:
    ```bash
    ollama --version
    ```

2.  Uruchom serwer Ollama. W nowym oknie terminala wpisz:
    ```bash
    ollama serve
    ```
    **Nie zamykaj tego okna**, serwer musi działać w tle.

3.  Pobierz model `bielik`:
    ```bash
    ollama pull mwiewior/bielik
    ```

4.  Sprawdź, czy model działa poprawnie:
    ```bash
    ollama run mwiewior/bielik
    ```
    Zadaj pytanie po polsku. Jeśli model odpowiada, wszystko jest gotowe.

## Uruchomienie aplikacji

W głównym folderze projektu uruchom skrypt:

```bash
python gui gradio.py
```

Żeby zobaczyć panel GUI, należy użyć linka do strony, który pojawi się w konsoli Python po uruchomieniu skryptu
