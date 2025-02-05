from TTS.api import TTS
import os
import numpy as np
import json
import time
import soundfile as sf

load_dotenv()


def generate_neutral_voice(analysis_file, output_file):

    with open(analysis_file, "r") as f:
        speech_data = json.load(f)

    tts = TTS(os.getenv("TTS_MODEL", "tts_models/en/vctk/vits"))

    # Compute average loudness, pitch, and pause
    avg_loudness = np.mean([word.get("loudness", -25.0) for word in speech_data])
    avg_pitch = np.mean([word.get("pitch", 100.0) for word in speech_data])
    avg_pause = np.mean([word.get("pause", 0.2) for word in speech_data])

    print(
        f"Generating voice with avg loudness: {avg_loudness:.2f} dB, avg pitch: {avg_pitch:.2f} Hz"
    )

    # Generate speech segment by segment to handle pauses
    speech_segments = []
    for word_data in speech_data:
        word = word_data["word"]
        pause_duration = word_data.get("pause", 0.0)
        word_wav = tts.tts(text=word)
        speech_segments.append(word_wav)

        # Add silence if the pause is significant
        if pause_duration > avg_pause:
            silence = np.zeros(
                int(pause_duration * 16000)
            )  # Assuming 16 kHz sample rate
            speech_segments.append(silence)

    final_wav = np.concatenate(speech_segments, axis=0)

    sf.write(output_file, final_wav, samplerate=22050)  # Assuming 22.05 kHz sample rate

    print(f"Generated speech saved to: {output_file}")
