import os
import re
import math
import time
import numpy as np
import gradio as gr
import ollama
from faster_whisper import WhisperModel

# ======================================================
# LOGIKA OBLICZENIOWA
# ======================================================
def clean_math_string(text: str):
    """Usuwa tagi AI, Markdown i zbędne słowa"""
    text = re.sub(r'<(.*?)>', '', text) 
    text = text.replace('`', '').replace('=', '')
    text = text.replace('wynik', '').replace('Result', '').replace('wyrażenie', '')
    return text.strip().split('\n')[0]

def calculate_safe(expression: str):
    allowed = {"sqrt": math.sqrt, "pow": math.pow, "abs": abs, "pi": math.pi}
    try:
        expr = clean_math_string(expression)
        result = eval(expr, {"__builtins__": {}}, allowed)
        return round(result, 2) if isinstance(result, float) else result
    except Exception:
        return "BŁĄD"

# ======================================================
# KONFIGURACJA AI
# ======================================================
WHISPER_MODEL_NAME = "base"
OLLAMA_MODEL_NAME = "mwiewior/bielik"
SYSTEM_PROMPT = (
    "Jesteś matematycznym parserem. Zamień polski tekst na wyrażenie matematyczne w Pythonie. "
    "Zasady: 1. Zwróć WYŁĄCZNIE liczby i operatory (np. 2+2). 2. NIE używaj słów ani znaku '='."
)

print("Inicjalizacja Whisper...")
whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")

# ======================================================
# BACKEND
# ======================================================
def process_voice(audio_path, history):
    if audio_path is None:
        return "Oczekiwanie na nagranie...", "---", gr.update(value="---"), history

    try:
        segments, _ = whisper_model.transcribe(audio_path, language="pl")
        text = " ".join(s.text for s in segments).strip()
        
        if not text:
            return "Nie wykryto mowy", "---", gr.update(value="BŁĄD DŹWIĘKU"), history

        resp = ollama.generate(model=OLLAMA_MODEL_NAME, system=SYSTEM_PROMPT, 
                               prompt=f"Tekst: {text}\nWyrażenie:", options={"temperature": 0.0})
        
        math_expr = clean_math_string(resp['response'])
        result = calculate_safe(math_expr)
        
        ts = time.strftime("%H:%M:%S")
        new_log = f"[{ts}] SŁYSZĘ: {text}\n[{ts}] FORMUŁA: {math_expr} | WYNIK: {result}\n" + ("-"*60) + "\n"
        updated_history = new_log + (history if history else "")

        color = "#27ae60"
        res_html = f"""
        <div style='font-size: 60px; font-weight: 900; color: {color}; 
             background: #f1f9f5; border: 4px solid {color}; padding: 30px; 
             border-radius: 20px; text-align: center; box-shadow: 0 10px 20px rgba(39,174,96,0.1);'>
             {result}
        </div>
        """
        
        return text, math_expr, res_html, updated_history

    except Exception as e:
        return "Błąd", "---", f"BŁĄD: {e}", history

# ======================================================
# INTERFEJS GRADIO
# ======================================================
green_theme = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate",
).set(
    block_background_fill="white",
    panel_background_fill="white",
    block_border_width="1px",
)

custom_css = """
/* Usuwa szare tło z kontenerów typu 'Group' i 'Box' */
.gradio-container { background-color: white !important; }
.gr-box, .gr-group, .gr-panel { background: white !important; border: none !important; }

/* Okno mikrofonu */
#mic_input { 
    border: 3px solid #27ae60 !important; 
    background-color: #f1f9f5 !important; 
    border-radius: 15px !important;
}

/* Przycisk Oblicz */
#calc_btn { 
    background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important; 
    color: white !important; font-size: 20px !important; font-weight: bold !important; 
    height: 70px !important; border-radius: 15px !important;
    box-shadow: 0 4px 15px rgba(39,174,96,0.2) !important;
}

/* Historia - Terminal */
#log_box textarea { 
    background-color: #0d1a10 !important; 
    color: #4ade80 !important; 
    font-family: 'Consolas', monospace !important; 
    border: 2px solid #27ae60 !important;
}

/* Instrukcja */
.instruction-box {
    border-left: 5px solid #27ae60 !important;
    background-color: #f1f9f5 !important;
    padding: 15px !important;
    border-radius: 5px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(theme=green_theme, css=custom_css, title="Kalkulator głosowy") as demo:
    history_state = gr.State("")
    
    gr.Markdown("# Kalkulator głosowy")
    gr.Markdown("z użyciem narzędzi OpenAI Whisper i Bielik.AI")

    with gr.Row():
        # --- PANEL LEWY (60%) ---
        with gr.Column(scale=60):
            # Instrukcja
            with gr.Column(elem_classes="instruction-box"):
                gr.Markdown("""
                ### INSTRUKCJA OBSŁUGI
                1. Kliknij ikonę **mikrofonu** i wypowiedz działanie (np. *"250 razy 4 dodać 6"*).
                2. Kliknij ikonę **STOP**, aby zakończyć nagrywanie.
                3. Naciśnij zielony przycisk **OBLICZ WYNIK**, aby otrzymać odpowiedź.
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
            out_res = gr.HTML("""
                <div style='font-size: 60px; font-weight: 900; color: #bdc3c7; 
                background: white; border: 4px solid #bdc3c7; padding: 30px; 
                border-radius: 20px; text-align: center;'>---</div>
            """)

        # --- PANEL PRAWY (40%) ---
        with gr.Column(scale=40):
            gr.Markdown("### HISTORIA OPERACJI")
            log_area = gr.TextArea(
                label="Historia operacji", 
                interactive=False, 
                lines=30, 
                elem_id="log_box",
                placeholder="Historia operacji..."
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
    demo.launch(share=True)