# Voice Calculator – AI-Powered Speech Calculator

A Python-based voice calculator powered by modern AI tools. It features a clean and intuitive web interface built with Gradio.

- **Speech Recognition:** `faster-whisper` (base model) for accurate Polish speech transcription.
- **Mathematical Parser:** The `mwiewior/bielik` model, running locally through Ollama, converts natural language into mathematical expressions.
- **User Interface:** Gradio provides a modern, browser-based web interface.

This project was developed as part of the Speech Technology course as a graded university assignment.

## Authors

- Oliwia Chmura
- Mateusz Juszczak
- Henryk Kosała

## Requirements

- Python 3.10–3.12
- Windows / Linux / macOS
- Microphone
- Modern web browser (e.g. Chrome, Firefox)
- At least 8 GB of RAM (recommended for smooth AI model performance)

## Installation

### 1. Clone the repository

Clone the repository or download the ZIP archive:

```bash
git clone https://github.com/mateusz-juszczak/voice-calculator-whisper-bielik.git
cd voice-calculator-whisper-bielik
```

### 2. Install Python dependencies

It is recommended to create a virtual environment before installing the required packages.

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not available, install the dependencies manually:

```bash
pip install gradio ollama faster-whisper numpy
```

*Note: `faster-whisper` may require additional dependencies, such as `ctranslate2`, which should be installed automatically by pip.*

## Installing the Bielik Model (via Ollama)

### 1. Install Ollama

Download and install **Ollama** from the official website:

https://ollama.com

After installation, verify that it works by running the following command in your terminal:

```bash
ollama --version
```

### 2. Start the Ollama server

Open a new terminal window and run:

```bash
ollama serve
```

**Keep this terminal window open**, as the Ollama server must continue running in the background.

### 3. Download the Bielik model

Run the following command to download the model:

```bash
ollama pull mwiewior/bielik
```

### 4. Verify the model

Check whether the model is working correctly:

```bash
ollama run mwiewior/bielik
```

Enter a question in Polish. If the model responds, the installation was successful.

## Running the Application

Navigate to the project's root directory and run:

```bash
python gui_gradio.py
```

To access the web interface, open the URL displayed in the Python console after the application starts.
