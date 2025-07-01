# 🎙️ Meeting Transcription and Summarization

This project records audio from your microphone and system audio, transcribes it using OpenAI's Whisper, and then uses a local Ollama model to generate a structured summary of the conversation.

## ✨ Features

*   **🎤 Audio Recording:** Records both microphone and system audio.
*   **📝 Transcription:** Uses OpenAI's Whisper for accurate speech-to-text.
*   **🤖 AI Summarization:** Generates summaries using a local Ollama model.
*   **📄 Output:** Saves the transcription and summary to a text file.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### ✅ Prerequisites

*   Python 3.x
*   `pip` for installing Python packages

### 🛠️ Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a Python virtual environment:**

    This project uses a virtual environment to manage dependencies. This keeps your project's dependencies separate from your system-wide Python installation.

    *   **Create the virtual environment:**
        ```bash
        python -m venv .venv
        ```

    *   **Activate the virtual environment:**
        *   On **macOS and Linux**:
            ```bash
            source .venv/bin/activate
            ```
        *   On **Windows**:
            ```bash
            .\.venv\Scripts\activate
            ```

3.  **Install the required packages:**

    The `requirements.txt` file contains a list of all the Python libraries needed to run the project. Install them using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  Make sure your virtual environment is activated.
2.  Ensure you have an Ollama model (e.g., `llama3`) installed and running.
3.  Run the main script:
    ```bash
    python main.py
    ```
4.  The script will start recording. Press `Enter` in the terminal to stop the recording.
5.  The script will then process the audio, transcribe it, and generate a summary.
6.  The final output will be saved to `transcription.txt`.

## 📦 Dependencies

*   `soundcard`
*   `scipy`
*   `openai-whisper`
*   `ollama`
*   `numpy`
*   `torch`
*   `PyAudio`

