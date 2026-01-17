import os
import re
import math
import time
import numpy as np
import gradio as gr
import ollama
from faster_whisper import WhisperModel
import program

def render_result(result_text):
    """
    Funkcja generująca HTML.
    Używamy zmiennych CSS (var(--...)), aby tło i tekst
    same dostosowywały się do trybu Light/Dark.
    """
    # Kolor ramki i tekstu wyniku (zielony/czerwony/szary)
    color = "var(--neutral-400)" # Domyślny szary (z palety Gradio)
    
    if result_text not in ["---", "BŁĄD", "ERROR"]:
        # Sukces - używamy intensywnej zieleni
        color = "#2ecc71"
    elif result_text in ["BŁĄD", "ERROR"]:
        # Błąd - czerwień
        color = "#e74c3c"

    # Zauważ użycie var(--background-fill-secondary) - to klucz do poprawnego działania w obu trybach
    return f"""
        <div style='
            font-size: 60px; 
            font-weight: 900; 
            color: {color}; 
            background: var(--background-fill-secondary); 
            border: 4px solid {color}; 
            padding: 30px; 
            border-radius: 20px; 
            text-align: center;
            transition: all 0.3s ease;
        '>{result_text}</div>
    """

def process_voice(audio_path, history):
    if audio_path is None:
        return "Brak nagrania", "---", render_result("---"), history

    text_whisper, text_math, result = program.process_audio_pipeline(audio_path)

    if result not in ["BŁĄD", "ERROR", "---"]:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}]\nWHISPER: {text_whisper}\nBIELIK: {text_math} = {result}\n{'-'*30}\n"
        history = entry + history

    return text_whisper, text_math, render_result(result), history

# ======================================================
# CSS - ELASTYCZNY (Dla obu motywów)
# ======================================================
custom_css = """
/* Styl dla kontenera instrukcji */
.instruction-box {
    border-left: 5px solid #27ae60 !important;
    background-color: var(--background-fill-secondary) !important;
    padding: 15px !important;
    border-radius: 8px;
    margin-bottom: 20px;
    color: var(--body-text-color) !important;
}

/* Okno mikrofonu */
#mic_input { 
    border: 2px solid #27ae60 !important; 
    border-radius: 15px !important;
    background-color: var(--background-fill-primary) !important;
}

/* Przycisk Oblicz - gradient zawsze wygląda dobrze */
#calc_btn { 
    background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important; 
    color: white !important; 
    font-size: 20px !important; 
    font-weight: bold !important; 
    height: 70px !important; 
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(39,174,96,0.3) !important;
}

/* Historia - Terminal */
/* Tutaj wymuszamy styl "matrixa" niezależnie od motywu, bo to terminal */
#log_box textarea { 
    background-color: #050a06 !important; 
    color: #4ade80 !important; 
    font-family: 'Consolas', monospace !important; 
    border: 1px solid #27ae60 !important;
}

/* Drobne poprawki dla czytelności */
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
"""

# ======================================================
# INTERFEJS GRADIO
# ======================================================
# Używamy motywu 'Soft', który świetnie obsługuje przełączanie Dark/Light
theme = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, css=custom_css, title="Kalkulator głosowy") as demo:
    history_state = gr.State("")
    
    gr.Markdown("# Kalkulator głosowy")
    gr.Markdown("z użyciem narzędzi OpenAI Whisper i Bielik.AI")

    with gr.Row():
        # --- PANEL LEWY ---
        with gr.Column(scale=60):
            # Instrukcja
            with gr.Column(elem_classes="instruction-box"):
                gr.Markdown("""
                ### INSTRUKCJA OBSŁUGI
                1. Kliknij ikonę **mikrofonu** i wypowiedz działanie.
                2. Kliknij ikonę **STOP**, aby zakończyć nagrywanie.
                3. Naciśnij zielony przycisk **OBLICZ WYNIK**.
                """)

            audio_input = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="NAGRYWANIE GŁOSU",
                elem_id="mic_input"
            )
            
            btn_run = gr.Button("OBLICZ WYNIK", elem_id="calc_btn")

            with gr.Row():
                out_whisper = gr.Textbox(label="SŁYSZĘ (Whisper):", interactive=False)
                out_math = gr.Textbox(label="FORMUŁA (Bielik):", interactive=False)

            gr.Markdown("### WYNIK KOŃCOWY:")
            
            # Początkowy stan HTML - używa zmiennych CSS
            out_res = gr.HTML(render_result("---"))

        # --- PANEL PRAWY ---
        with gr.Column(scale=40):
            gr.Markdown("### HISTORIA OPERACJI")
            log_area = gr.TextArea(
                label="Historia operacji", 
                interactive=False, 
                lines=30, 
                elem_id="log_box",
                placeholder="Terminal logów..."
            )
            btn_clear = gr.Button("Wyczyść historię", variant="secondary")

    # Powiązania logiczne
    btn_run.click(
        fn=process_voice,
        inputs=[audio_input, history_state],
        outputs=[out_whisper, out_math, out_res, history_state]
    ).then(
        fn=lambda x: x,
        inputs=[history_state],
        outputs=[log_area]
    )

    btn_clear.click(lambda: ("", ""), None, [history_state, log_area])

if __name__ == "__main__":
    program.warmup()
    demo.launch(share=True)