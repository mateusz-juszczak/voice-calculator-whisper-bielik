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
    allowed = {"sqrt": math.sqrt, "pow": math.pow, "abs": abs, "pi": math.pi}
    try:
        # Usuwanie technicznych znaczników Bielika
        expr = expression.replace("<s>", "").replace("</s>", "").replace("`", "").strip()
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
        self.root.title("Bielik Voice AI - Profesjonalny Kalkulator Głosowy")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f4f4f9")
        
        self.whisper_model = None
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.monitor_stream = None 
        self.selected_mic_id = None
        self.current_db = -50
        
        # Animacja Messenger
        self.anim_running = False
        self.anim_stage = "IDLE" 
        self.anim_counter = 0

        self.FIXED_SAMPLE_RATE = 16000 
        self.WHISPER_MODEL_NAME = "base"
        self.OLLAMA_MODEL_NAME = "mwiewior/bielik"
        
        self.SYSTEM_PROMPT = (
            "Jesteś matematycznym parserem. Zamień polski tekst na wyrażenie matematyczne w Pythonie. "
            "Zasady:\n"
            "1. Zwróć WYŁĄCZNIE liczby i operatory (np. 2+2).\n"
            "2. NIE używaj słów ani znaku '='."
        )

        self.create_widgets()
        self.refresh_microphones()
        self.update_meter_loop() 
        self.animate_dots_loop()
        
        threading.Thread(target=self.init_models, daemon=True).start()

    def create_widgets(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- PANEL LEWY ---
        left_panel = ttk.Frame(self.paned, padding="10")
        self.paned.add(left_panel, weight=65)

        # Instrukcja
        inst_frame = ttk.LabelFrame(left_panel, text=" 📖 INSTRUKCJA OBSŁUGI ", padding="15")
        inst_frame.pack(fill=tk.X, pady=(0, 10))
        instructions = (
            "1. Wybierz mikrofon z listy i kliknij 'Zatwierdź urządzenie'\n"
            "2. Naciśnij zielony przycisk i wypowiedz działanie matematyczne\n"
            "3. Kliknij STOP, aby obliczyć wynik"
        )
        tk.Label(inst_frame, text=instructions, font=("Arial", 10), justify=tk.LEFT, fg="#444").pack(anchor="w")

        # Mikrofon + VU Meter
        mic_frame = ttk.LabelFrame(left_panel, text=" 🎤 KONFIGURACJA WEJŚCIA AUDIO ", padding="15")
        mic_frame.pack(fill=tk.X, pady=(0, 10))
        
        mic_inner = ttk.Frame(mic_frame)
        mic_inner.pack(fill=tk.X)
        mic_controls = ttk.Frame(mic_inner)
        mic_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mic_combo = ttk.Combobox(mic_controls, state="readonly", font=("Arial", 10))
        self.mic_combo.pack(fill=tk.X, pady=(0, 10))
        
        btn_grid = ttk.Frame(mic_controls)
        btn_grid.pack(fill=tk.X)
        ttk.Button(btn_grid, text="🔄 Odśwież", command=self.refresh_microphones).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_grid, text="✅ Zatwierdź urządzenie", command=self.set_microphone).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # VU Meter
        vu_container = ttk.Frame(mic_inner, padding=(20, 0, 0, 0))
        vu_container.pack(side=tk.RIGHT, fill=tk.Y)
        scale_frame = tk.Frame(vu_container, bg="#f4f4f9")
        scale_frame.pack(side=tk.LEFT, fill=tk.Y)
        for label in ["0", "-10", "-25", "-50"]:
            tk.Label(scale_frame, text=label, font=("Arial", 7), fg="#666").pack(expand=True)

        self.vu_canvas = tk.Canvas(vu_container, width=25, height=100, bg="#1a1a1a", highlightthickness=1)
        self.vu_canvas.pack(side=tk.LEFT, padx=5)

        # Wyniki
        res_frame = ttk.Frame(left_panel)
        res_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(res_frame, text="SŁYSZĘ:", font=("Arial", 9, "bold")).pack(anchor="w")
        self.out_whisper = tk.Label(res_frame, text="...", font=("Arial", 14), bg="white", relief="sunken", anchor="w", padx=10, pady=8)
        self.out_whisper.pack(fill=tk.X, pady=(0, 10))

        tk.Label(res_frame, text="FORMUŁA:", font=("Arial", 9, "bold")).pack(anchor="w")
        self.out_math = tk.Label(res_frame, text="---", font=("Consolas", 18, "bold"), fg="#27ae60", bg="white", relief="sunken", anchor="w", padx=10, pady=8)
        self.out_math.pack(fill=tk.X, pady=(0, 10))

        tk.Label(res_frame, text="WYNIK KOŃCOWY:", font=("Arial", 12, "bold"), fg="#2980b9").pack(anchor="w")
        self.out_res = tk.Label(res_frame, text="---", font=("Arial", 55, "bold"), fg="#2980b9", bg="#ebf5fb", relief="ridge", pady=15)
        self.out_res.pack(fill=tk.X, pady=(0, 15))

        self.btn_main = tk.Button(left_panel, text="ŁADOWANIE MODELI...", font=("Arial", 18, "bold"), bg="#95a5a6", fg="white", height=2, state=tk.DISABLED, command=self.toggle_recording)
        self.btn_main.pack(fill=tk.X)

        # --- PANEL PRAWY ---
        right_panel = ttk.Frame(self.paned, padding="10", width=420)
        self.paned.add(right_panel, weight=35)
        tk.Label(right_panel, text="📜 HISTORIA OPERACJI", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.log_area = scrolledtext.ScrolledText(right_panel, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 10), borderwidth=0)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def animate_dots_loop(self):
        if self.anim_running:
            dots = "." * (self.anim_counter % 4)
            if self.anim_stage == "WHISPER":
                self.out_whisper.config(text=f"Słucham{dots}")
            elif self.anim_stage == "BIELIK":
                self.out_math.config(text=f"Analizuję{dots}")
            elif self.anim_stage == "CALC":
                self.out_res.config(text=f"{dots}")
            self.anim_counter += 1
        self.root.after(300, self.animate_dots_loop)

    def monitor_callback(self, indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        self.current_db = 20 * math.log10(rms) if rms > 1e-6 else -50
        if self.current_db < -50: self.current_db = -50

    def update_meter_loop(self):
        self.vu_canvas.delete("led")
        active = int(((self.current_db + 50) / 50) * 15)
        h = 100 / 15
        for i in range(15):
            idx = 14 - i
            color = "#2ecc71" if idx < 9 else ("#f1c40f" if idx < 13 else "#e74c3c")
            self.vu_canvas.create_rectangle(3, i*h+2, 22, (i+1)*h-1, fill=color if idx < active else "#333", outline="#1a1a1a", tags="led")
        self.root.after(50, self.update_meter_loop)

    def refresh_microphones(self):
        devices = sd.query_devices()
        mics = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.mic_combo['values'] = mics
        if mics: self.mic_combo.current(0)

    def set_microphone(self):
        try:
            if self.monitor_stream: self.monitor_stream.stop(); self.monitor_stream.close()
            selection = self.mic_combo.get()
            self.selected_mic_id = int(selection.split(":")[0])
            self.monitor_stream = sd.InputStream(device=self.selected_mic_id, channels=1, callback=self.monitor_callback, samplerate=self.FIXED_SAMPLE_RATE)
            self.monitor_stream.start()
            self.log(f"Mikrofon: {selection}")
            messagebox.showinfo("Mikrofon", "Urządzenie gotowe do pracy.")
        except Exception as e:
            # Poprawiony komunikat o błędzie zgodnie z prośbą
            messagebox.showerror("Błąd urządzenia", 
                f"Nie można aktywować tego urządzenia ({e}).\n\n"
                "Upewnij się, że mikrofon jest podłączony i nie jest używany przez inny program, "
                "lub wybierz inne urządzenie z listy.")

    def init_models(self):
        try:
            self.whisper_model = WhisperModel(self.WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
            self.btn_main.config(text="KLIKNIJ, ABY ZACZĄĆ NAGRYWANIE", bg="#27ae60", state=tk.NORMAL)
        except Exception as e: self.log(f"Błąd AI: {e}")

    def audio_callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())
        self.monitor_callback(indata, frames, time, status)

    def toggle_recording(self):
        if self.selected_mic_id is None:
            messagebox.showwarning("Błąd urządzenia", "Najpierw wybierz mikrofon z listy i kliknij 'Zatwierdź urządzenie'!")
            return

        if not self.is_recording:
            self.out_whisper.config(text="...")
            self.out_math.config(text="---")
            self.out_res.config(text="---", fg="#2980b9")
            if self.monitor_stream: self.monitor_stream.stop()
            self.audio_data = []
            try:
                self.stream = sd.InputStream(samplerate=self.FIXED_SAMPLE_RATE, channels=1, device=self.selected_mic_id, callback=self.audio_callback)
                self.stream.start()
                self.is_recording = True
                self.anim_running, self.anim_stage = True, "WHISPER"
                self.btn_main.config(text="🛑 STOP (OBLICZ)", bg="#e74c3c")
            except Exception as e: 
                # Poprawiony komunikat o błędzie zgodnie z prośbą
                messagebox.showerror("Błąd nagrywania", 
                    f"Wystąpił problem z uruchomieniem nagrywania: {e}\n\n"
                    "Proszę wybrać inne urządzenie wejściowe w ustawieniach.")
                if self.monitor_stream: self.monitor_stream.start()
        else:
            self.is_recording = False
            if self.stream: self.stream.stop(); self.stream.close()
            if self.monitor_stream: self.monitor_stream.start()
            self.btn_main.config(text="⌛ ANALIZA...", bg="#f39c12", state=tk.DISABLED)
            threading.Thread(target=self.process_audio).start()

    def process_audio(self):
        try:
            if not self.audio_data: return
            self.anim_stage = "WHISPER"
            full_audio = np.concatenate(self.audio_data, axis=0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, full_audio, self.FIXED_SAMPLE_RATE)
                path = f.name

            segments, _ = self.whisper_model.transcribe(path, language="pl")
            text = " ".join(s.text for s in segments).strip()
            
            if text:
                self.out_whisper.config(text=text)
                self.log(f"Whisper: {text}")
                self.anim_stage = "BIELIK"
                resp = ollama.generate(model=self.OLLAMA_MODEL_NAME, system=self.SYSTEM_PROMPT, 
                                      prompt=f"Tekst: {text}\nWyrażenie:", options={"temperature": 0.0})
                math_expr = resp['response'].replace("<s>", "").replace("</s>", "").replace("`", "").strip().split('\n')[0]
                self.out_math.config(text=math_expr)
                self.anim_stage = "CALC"
                time.sleep(0.5)
                result = calculate_safe(math_expr)
                self.anim_running = False
                self.out_res.config(text=str(result), fg="#2980b9" if result != "BŁĄD" else "#e74c3c")
                self.log(f"Bielik: {math_expr} | Wynik: {result}")
            else:
                self.anim_running = False
                self.out_whisper.config(text="(Brak dźwięku)")

            if os.path.exists(path): os.remove(path)
        except Exception as e: 
            self.anim_running = False
            self.log(f"Błąd AI: {e}")
        finally:
            self.btn_main.config(text="KLIKNIJ, ABY ZACZĄĆ NAGRYWANIE", bg="#27ae60", state=tk.NORMAL)

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceCalcGUI(root)
    root.mainloop()