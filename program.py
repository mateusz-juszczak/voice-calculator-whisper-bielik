import os
import re
import math
import time
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import ollama
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from faster_whisper import WhisperModel

# ======================================================
# LOGIKA OBLICZENIOWA
# ======================================================
def calculate_safe(expression: str):
    if not expression: return "BŁĄD"
    allowed = {"sqrt": math.sqrt, "pow": math.pow, "abs": abs}
    try:
        expr = expression.replace("<s>", "").replace("</s>", "").strip()
        if "=" in expr: expr = expr.split("=")[-1]
        expr = expr.replace("wynik", "").replace("Result", "").strip()
        result = eval(expr, {"__builtins__": {}}, allowed)
        return round(result, 2) if isinstance(result, float) else result
    except Exception:
        return "BŁĄD"

# ======================================================
# GUI
# ======================================================
class VoiceCalcGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bielik Voice AI - Professional Calculator")
        self.root.geometry("1150x750")
        self.root.configure(bg="#f4f4f9")
        
        self.whisper_model = None
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.selected_mic_id = None
        
        # USTAWIONE NA SZTYWNO 16kHz
        self.FIXED_SAMPLE_RATE = 16000 

        self.WHISPER_MODEL_NAME = "base"
        self.OLLAMA_MODEL_NAME = "mwiewior/bielik"
        self.SYSTEM_PROMPT = (
            "Jesteś matematycznym parserem. Zamień polski tekst na wyrażenie matematyczne w Pythonie. "
            "Zasady:\n"
            "1. Zwróć WYŁĄCZNIE liczby i operatory (np. '2 + 2').\n"
            "2. NIE używaj słów 'Result', 'wynik', 'równa się'.\n"
            "3. NIE używaj znaku '='.\n"
            "4. Jeśli słyszysz liczebniki porządkowe (np. piątych), traktuj je jak ułamki lub liczby."
        )

        self.create_widgets()
        self.refresh_microphones()
        threading.Thread(target=self.init_models, daemon=True).start()

    def create_widgets(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(paned, padding="10")
        paned.add(left_panel, weight=3)

        # Sekcja Mikrofonu
        mic_frame = ttk.LabelFrame(left_panel, text=" Wybór Mikrofonu ", padding="10")
        mic_frame.pack(fill=tk.X, pady=(0, 15))

        self.mic_combo = ttk.Combobox(mic_frame, state="readonly", width=45, font=("Arial", 10))
        self.mic_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(mic_frame, text="Odśwież", command=self.refresh_microphones).pack(side=tk.LEFT, padx=2)
        self.btn_set_mic = ttk.Button(mic_frame, text="Zastosuj mikrofon", command=self.set_microphone)
        self.btn_set_mic.pack(side=tk.LEFT, padx=2)

        # Wyświetlacze
        res_frame = ttk.Frame(left_panel)
        res_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(res_frame, text="SŁYSZĘ (WHISPER):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.out_whisper = tk.Label(res_frame, text="...", font=("Arial", 18), bg="white", anchor="w", relief="sunken", padx=10, pady=10)
        self.out_whisper.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(res_frame, text="WYRAŻENIE (BIELIK):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.out_math = tk.Label(res_frame, text="...", font=("Arial", 22, "bold"), fg="#27ae60", bg="white", anchor="w", relief="sunken", padx=10, pady=10)
        self.out_math.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(res_frame, text="WYNIK KOŃCOWY:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.out_res = tk.Label(res_frame, text="0", font=("Arial", 60, "bold"), fg="#2980b9", bg="#ebf5fb", relief="ridge", pady=15)
        self.out_res.pack(fill=tk.X, pady=(0, 15))

        # Przycisk Akcji
        self.btn_main = tk.Button(
            left_panel, text="ŁADOWANIE MODELI...", font=("Arial", 18, "bold"),
            bg="#95a5a6", fg="white", height=2, state=tk.DISABLED, command=self.toggle_recording
        )
        self.btn_main.pack(fill=tk.X)

        self.status_lbl = ttk.Label(left_panel, text="System: Inicjalizacja...", foreground="gray")
        self.status_lbl.pack(anchor="w", pady=5)

        # Historia (Prawa strona)
        right_panel = ttk.Frame(paned, padding="10")
        paned.add(right_panel, weight=2)
        ttk.Label(right_panel, text="HISTORIA OPERACJI", font=("Arial", 10, "bold")).pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(right_panel, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_area.see(tk.END)

    def refresh_microphones(self):
        devices = sd.query_devices()
        mics = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.mic_combo['values'] = mics
        if mics: self.mic_combo.current(0)

    def set_microphone(self):
        try:
            selection = self.mic_combo.get()
            self.selected_mic_id = int(selection.split(":")[0])
            self.log(f"Wybrano: {selection} (Wymuszone 16kHz)")
            messagebox.showinfo("Sukces", f"Ustawiono mikrofon.\nCzęstotliwość wejściowa: 16000 Hz")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można ustawić mikrofonu: {e}")

    def init_models(self):
        try:
            self.whisper_model = WhisperModel(self.WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
            self.log("Modele AI gotowe.")
            self.btn_main.config(text="ZACZNIJ MÓWIĆ", bg="#27ae60", state=tk.NORMAL)
            self.status_lbl.config(text="System: Gotowy", foreground="green")
        except Exception as e:
            self.log(f"Błąd inicjalizacji: {e}")

    def audio_callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())

    def toggle_recording(self):
        if self.selected_mic_id is None:
            messagebox.showwarning("Błąd", "Najpierw wybierz mikrofon!")
            return

        if not self.is_recording:
            try:
                self.audio_data = []
                # Próba otwarcia strumienia na 16000 Hz
                self.stream = sd.InputStream(
                    samplerate=self.FIXED_SAMPLE_RATE,
                    channels=1,
                    device=self.selected_mic_id,
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_recording = True
                self.btn_main.config(text="STOP (KLIKNIJ BY OBLICZYĆ)", bg="#e74c3c")
                self.status_lbl.config(text="Status: Słucham (16kHz)...", foreground="red")
                self.log("Start nagrywania (16kHz)...")
            except Exception as e:
                self.log(f"Błąd PortAudio: {e}")
                messagebox.showerror("Błąd sprzętowy", 
                    f"Twoje urządzenie nie wspiera próbkowania 16000 Hz bezpośrednio.\n\n"
                    f"Szczegóły: {e}")
                if self.stream: 
                    self.stream.close()
                    self.stream = None
        else:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.btn_main.config(text="ANALIZOWANIE...", bg="#f39c12", state=tk.DISABLED)
            threading.Thread(target=self.process_audio).start()

    def process_audio(self):
        try:
            if not self.audio_data:
                self.handle_empty_audio()
                return

            full_audio = np.concatenate(self.audio_data, axis=0)
            
            if np.max(np.abs(full_audio)) < 0.01:
                self.handle_empty_audio()
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, full_audio, self.FIXED_SAMPLE_RATE)
                path = f.name

            # 1. Whisper
            self.status_lbl.config(text="Status: Przetwarzanie Whisper...")
            segments, _ = self.whisper_model.transcribe(path, language="pl")
            text = " ".join(s.text for s in segments).strip()
            self.out_whisper.config(text=text if text else "(Cisza)")

            if text:
                # ZMIENIONO LOG NA "Whisper:"
                self.log(f"Whisper: {text}") 
                self.status_lbl.config(text="Status: Przetwarzanie Bielik...")
                
                resp = ollama.generate(
                    model=self.OLLAMA_MODEL_NAME,
                    system=self.SYSTEM_PROMPT,
                    prompt=f"Tekst: {text}\nWyrażenie:",
                    options={"temperature": 0.0}
                )
                
                raw_math = resp['response']
                math_expr = re.search(r"```(?:python)?(.*?)```", raw_math, re.DOTALL)
                math_expr = math_expr.group(1).strip() if math_expr else raw_math.strip()
                math_expr = math_expr.replace("<s>", "").replace("</s>", "").split('\n')[0]
                
                self.out_math.config(text=math_expr)
                
                result = calculate_safe(math_expr)
                self.out_res.config(text=str(result))
                self.log(f"Bielik: {math_expr} = {result}")

            if os.path.exists(path): os.remove(path)

        except Exception as e:
            self.log(f"Błąd: {e}")
            messagebox.showerror("Błąd", str(e))
        finally:
            self.btn_main.config(text="ZACZNIJ MÓWIĆ", bg="#27ae60", state=tk.NORMAL)
            self.status_lbl.config(text="Status: Gotowy", foreground="green")

    def handle_empty_audio(self):
        messagebox.showerror("Błąd audio", "Brak sygnału z mikrofonu. Sprawdź urządzenie.")
        self.btn_main.config(text="ZACZNIJ MÓWIĆ", bg="#27ae60", state=tk.NORMAL)
        self.status_lbl.config(text="Status: Błąd mikrofonu", foreground="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceCalcGUI(root)
    root.mainloop()