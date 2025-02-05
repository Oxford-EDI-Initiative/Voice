import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



def analyze_audio(file_path):
    # Load audio file
    audio = AudioSegment.from_file(file_path)

    # Use Whisper to transcribe audio and get word timestamps

    # Load a pre-trained Whisper model
    model = whisper.load_model("small")

    result = model.transcribe(file_path, word_timestamps=True)
    words_data = result["segments"]

    # Convert audio to samples
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # Calculate loudness and pauses
    word_list = []
    last_end_time = 0
    for segment in words_data:
        for word_info in segment.get("words", []):  # Ensure "words" key exists
            word = word_info.get("word", "UNKNOWN")  # Correct key
            start_time = word_info.get("start", 0.0)
            end_time = word_info.get("end", 0.0)

            # Extract the word's audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            word_audio = y[start_sample:end_sample]

            # Compute loudness (RMS)
            rms = np.sqrt(np.mean(np.square(word_audio)))
            loudness_db = 20 * np.log10(rms + 1e-10)  # Avoid log(0)

            # Compute pause
            pause_duration = start_time - last_end_time if last_end_time > 0 else 0
            last_end_time = end_time

            word_list.append([word, round(loudness_db, 2), round(pause_duration, 2)])
    return word_list



file_path = r"C:\Users\xiaon\Desktop\1.wav"  # Replace with actual file
results = analyze_audio(file_path)
print(results)
