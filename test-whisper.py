import whisper
import os
import time
import ollama
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(BASE_DIR, "testowe_rownanie.wav")
model = whisper.load_model("base", device=device)
result = model.transcribe(audio_path, language="pl")

result_whisper = result["text"]
print("WHISPER: " + result_whisper)



result_whisper = "5 plus 2 razy 7"

response = ollama.generate(
    model='mwiewior/bielik',
    system='rozwiąż równanie i jako odpowiedź zwróć tylko liczbę, która jest końcowym wynikiem',
    prompt=result_whisper
)

text_answer = response["response"].split("<s>  ")


print("BIELIK: " + text_answer[1])
