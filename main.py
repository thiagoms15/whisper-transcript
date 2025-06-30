import soundcard as sc
import numpy as np
from scipy.io.wavfile import write
import whisper
import ollama
import threading
import queue

# --- Configuration ---
OUTPUT_FILENAME = "output.wav"
TRANSCRIPTION_FILENAME = "transcription.txt"
SAMPLE_RATE = 48000
OLLAMA_MODEL = "llama3"  # Or any other model you have installed

# --- Global State ---
is_recording = threading.Event()
audio_q = queue.Queue()


def record_audio():
    """Records audio from the default microphone and loopback simultaneously."""
    try:
        with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic, \
             sc.default_microphone().recorder(samplerate=SAMPLE_RATE) as speaker:
            is_recording.set()
            print("Recording... Press Enter to stop.")
            while is_recording.is_set():
                data_mic = mic.record(numframes=SAMPLE_RATE)
                data_speaker = speaker.record(numframes=SAMPLE_RATE)
                # Simple mixing by adding the two signals
                mixed_data = data_mic + data_speaker
                # Convert to float32 and normalize to -1.0 to 1.0
                mixed_data = mixed_data.astype(np.float32) / np.max(np.abs(mixed_data))
                audio_q.put(mixed_data)
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        print("Recording stopped.")


def main():
    """Main function to control recording, transcription, and AI processing."""
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

    # Wait for the user to press Enter to stop recording
    input()
    is_recording.clear()
    record_thread.join()

    # --- Process Audio ---
    print("Processing audio...")
    audio_data = []
    while not audio_q.empty():
        audio_data.append(audio_q.get())

    if not audio_data:
        print("No audio data recorded.")
        return

    # Concatenate all the recorded chunks
    audio_data_np = np.concatenate(audio_data, axis=0)

    # --- Save to WAV ---
    print(f"Saving audio to {OUTPUT_FILENAME}...")
    write(OUTPUT_FILENAME, SAMPLE_RATE, audio_data_np)

    # --- Transcription ---
    print("Transcribing audio... (This may take a moment)")
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(OUTPUT_FILENAME)
        transcription = result["text"]
        print("Transcription complete.")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return

    # --- AI Processing (Ollama) ---
    print(f"Sending transcription to Ollama model: {OLLAMA_MODEL}...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize the following transcription:\n\n{transcription}",
                },
            ],
        )
        ai_summary = response["message"]["content"]
        print("Ollama processing complete.")
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        ai_summary = "Ollama processing failed."

    # --- Save Transcription ---
    final_output = f"--- Transcription ---\n{transcription}\n\n--- Ollama Summary ---\n{ai_summary}"
    with open(TRANSCRIPTION_FILENAME, "w") as f:
        f.write(final_output)
    print(f"Transcription and summary saved to {TRANSCRIPTION_FILENAME}")


if __name__ == "__main__":
    main()