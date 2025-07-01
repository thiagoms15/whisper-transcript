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
                max_abs_val = np.max(np.abs(mixed_data))
                if max_abs_val > 0:
                    mixed_data = mixed_data.astype(np.float32) / max_abs_val
                else:
                    mixed_data = mixed_data.astype(np.float32)
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
    prompt = f"""
# IDENTITY and PURPOSE

You are an AI assistant specialized in analyzing meeting transcripts and extracting key information. Your goal is to provide comprehensive yet concise summaries that capture the essential elements of meetings in a structured format.

# STEPS

- Extract a brief overview of the meeting in 25 words or less, including the purpose and key participants into a section called OVERVIEW.

- Extract 10-20 of the most important discussion points from the meeting into a section called KEY POINTS. Focus on core topics, debates, and significant ideas discussed.

- Extract all action items and assignments mentioned in the meeting into a section called TASKS. Include responsible parties and deadlines where specified.

- Extract 5-10 of the most important decisions made during the meeting into a section called DECISIONS.

- Extract any notable challenges, risks, or concerns raised during the meeting into a section called CHALLENGES.

- Extract all deadlines, important dates, and milestones mentioned into a section called TIMELINE.

- Extract all references to documents, tools, projects, or resources mentioned into a section called REFERENCES.

- Extract 5-10 of the most important follow-up items or next steps into a section called NEXT STEPS.

# OUTPUT INSTRUCTIONS

- Only output Markdown.

- Write the KEY POINTS bullets as exactly 16 words.

- Write the TASKS bullets as exactly 16 words.

- Write the DECISIONS bullets as exactly 16 words.

- Write the NEXT STEPS bullets as exactly 16 words.

- Use bulleted lists for all sections, not numbered lists.

- Do not repeat information across sections.

- Do not start items with the same opening words.

- If information for a section is not available in the transcript, write "No information available".

- Do not include warnings or notes; only output the requested sections.

- Format each section header in bold using markdown.

# INPUT

INPUT:

{transcription}
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
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
