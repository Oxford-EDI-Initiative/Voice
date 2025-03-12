import obspython as obs
import speech_recognition as speech_rec
import numpy as np
import librosa
import io
import soundfile as sf
import threading
import queue
import time
import torch
import os
import tempfile
from datetime import datetime
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from ctypes import windll  # For Windows systems

# Global variables
source_name = ""
audio_queue = queue.Queue()
is_running = False
recognizer = speech_rec.Recognizer()
output_device = None
device_index = 1  # Intel Microphone Array
pause_threshold = 0.5  # Minimum pause duration to detect (seconds)
analysis_thread = None
synthesis_thread = None
voice_seed = 42  # Seed for voice characteristics
pitch_shift_amount = -1.5  # Amount to shift pitch for more neutral voice

# Create output directory in a safe location
output_dir = os.path.join(tempfile.gettempdir(), "speech_neutralizer")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Load pretrained models
def load_models():
    global processor, model, vocoder
    
    print("Loading speech models...")
    try:
        # Using the correct SpeechT5 classes for TTS
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        # Add vocoder for better speech quality
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you have the transformers library installed and internet access.")
        import traceback
        traceback.print_exc()
        return False

# Initialize plugin
def script_load(settings):
    print("Speech Neutralizer Plugin loaded")
    try:
        load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")

def script_description():
    return "Speech Neutralizer: Analyzes speech, detects words, pauses, and loudness, and outputs neutralized speech in real-time."

def script_properties():
    props = obs.obs_properties_create()
    
    # Audio Source selection
    p = obs.obs_properties_add_list(props, "source", "Audio Source", obs.OBS_COMBO_TYPE_EDITABLE, obs.OBS_COMBO_FORMAT_STRING)
    sources = obs.obs_enum_sources()
    if sources:
        for source in sources:
            source_id = obs.obs_source_get_name(source)
            obs.obs_property_list_add_string(p, source_id, source_id)
        obs.source_list_release(sources)
    
    # Audio Device selection
    devices = obs.obs_properties_add_list(props, "device_index", "Audio Device", obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_INT)
    obs.obs_property_list_add_int(devices, "Intel Microphone Array (index 1)", 1)
    obs.obs_property_list_add_int(devices, "Intel Microphone Array Alternative (index 8)", 8)
    obs.obs_property_list_add_int(devices, "CABLE Output (VB-Audio Virtual) (index 9)", 9)
    
    # Output Device selection
    output_devices = obs.obs_properties_add_list(props, "output_device", "Output Device", obs.OBS_COMBO_TYPE_EDITABLE, obs.OBS_COMBO_FORMAT_STRING)
    try:
        import sounddevice as sd
        sd_devices = sd.query_devices()
        for i, device in enumerate(sd_devices):
            if device['max_output_channels'] > 0:  # Only include output devices
                device_name = device['name']
                obs.obs_property_list_add_string(output_devices, device_name, str(i))
    except ImportError:
        print("Warning: sounddevice library not found. Install it with 'pip install sounddevice' to list output devices.")
    except Exception as e:
        print(f"Error listing output devices: {e}")
    
    # Add output channels selection
    channels = obs.obs_properties_add_list(props, "output_channels", "Output Channels", obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_INT)
    obs.obs_property_list_add_int(channels, "Mono (1)", 1)
    obs.obs_property_list_add_int(channels, "Stereo (2)", 2)
    
    # Voice settings
    obs.obs_properties_add_int_slider(props, "voice_seed", "Voice Characteristics (Seed)", 1, 100, 1)
    obs.obs_properties_add_float_slider(props, "pitch_shift", "Voice Pitch Adjustment", -4.0, 4.0, 0.5)
    
    # Add output info
    info = obs.obs_properties_add_text(props, "output_info", f"Output files saved to: {output_dir}", obs.OBS_TEXT_INFO)
    
    # Add buttons and slider
    obs.obs_properties_add_button(props, "start_button", "Start Processing", start_processing_button)
    obs.obs_properties_add_button(props, "stop_button", "Stop Processing", stop_processing_button)
    obs.obs_properties_add_float_slider(props, "pause_threshold", "Pause Threshold (seconds)", 0.1, 2.0, 0.1)
    
    return props

def script_update(settings):
    global source_name, output_device, pause_threshold, device_index, output_channels, voice_seed, pitch_shift_amount
    
    source_name = obs.obs_data_get_string(settings, "source")
    output_device = obs.obs_data_get_string(settings, "output_device")
    pause_threshold = obs.obs_data_get_double(settings, "pause_threshold")
    
    # Get voice settings
    voice_seed = obs.obs_data_get_int(settings, "voice_seed")
    pitch_shift_amount = obs.obs_data_get_double(settings, "pitch_shift")
    
    # Get output channels
    output_channels = obs.obs_data_get_int(settings, "output_channels")
    if output_channels not in [1, 2]:
        output_channels = 2  # Default to stereo
    
    # Get the device index as an integer
    device_index = obs.obs_data_get_int(settings, "device_index")
    if device_index not in [1, 8, 9]:  # Validate it's one of our expected values
        device_index = 1  # Default to Intel Microphone Array
        print(f"Setting device index to default: {device_index}")
    else:
        print(f"Using device index: {device_index}")

def script_defaults(settings):
    obs.obs_data_set_default_int(settings, "device_index", 1)
    obs.obs_data_set_default_double(settings, "pause_threshold", 0.5)
    obs.obs_data_set_default_int(settings, "output_channels", 2)  # Default to stereo
    obs.obs_data_set_default_int(settings, "voice_seed", 42)
    obs.obs_data_set_default_double(settings, "pitch_shift", -1.5)

def start_processing_button(props, prop):
    start_processing()
    return True

def stop_processing_button(props, prop):
    stop_processing()
    return True

# Audio capture and processing
def start_processing():
    global is_running, analysis_thread, synthesis_thread
    
    if is_running:
        return
    
    is_running = True
    
    # Start audio capture thread
    analysis_thread = threading.Thread(target=audio_analysis_loop)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    # Start synthesis thread
    synthesis_thread = threading.Thread(target=speech_synthesis_loop)
    synthesis_thread.daemon = True
    synthesis_thread.start()
    
    print("Speech processing started")

def stop_processing():
    global is_running
    
    is_running = False
    print("Speech processing stopped")

def audio_analysis_loop():
    global is_running, audio_queue, device_index
    
    print(f"Starting audio analysis with device index: {device_index}")
    
    # Get the OBS source (for logging only)
    obs_source = None
    if source_name and source_name != "None":
        obs_source = obs.obs_get_source_by_name(source_name)
        if not obs_source:
            print(f"Warning: Source '{source_name}' not found in OBS")
    
    try:
        while is_running:
            try:
                # Create a new microphone context for each recording session
                with speech_rec.Microphone(device_index=device_index) as audio_source:
                    # Make sure we adjust for ambient noise to improve recognition
                    print("Adjusting for ambient noise...")
                    recognizer.adjust_for_ambient_noise(audio_source, duration=0.5)
                    
                    # Use a fixed recording duration to avoid timeouts
                    print("Recording audio chunk...")
                    audio = recognizer.record(audio_source, duration=5.0)
                    print("Audio chunk recorded")
                    
                    # Process the recorded audio
                    result = analyze_audio(audio)
                    if result:
                        audio_queue.put(result)
                        print("Audio analysis complete and queued")
            except Exception as e:
                print(f"Error in audio analysis loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Wait before retrying
    finally:
        if obs_source:
            obs.obs_source_release(obs_source)
        print("Audio analysis thread stopped")

def analyze_audio(audio_data):
    try:
        # Convert audio data to numpy array for analysis
        wav_data = io.BytesIO(audio_data.get_wav_data())
        y, sample_rate = librosa.load(wav_data, sr=16000)
        
        # Try speech recognition
        text = ""
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Recognized text: {text}")
        except speech_rec.UnknownValueError:
            print("No speech detected")
        except speech_rec.RequestError as e:
            print(f"Recognition error: {e}")
        
        # Detect words and pauses
        words = text.split() if text else []
        pauses = []
        loudness = []
        
        # Only proceed with detailed analysis if we have audio data
        if len(y) > 0:
            # Detect onsets for pause analysis
            onset_frames = librosa.onset.onset_detect(y=y, sr=sample_rate, units='time')
            
            if len(onset_frames) > 1:
                for i in range(len(onset_frames) - 1):
                    start_time = onset_frames[i]
                    end_time = onset_frames[i + 1]
                    duration = end_time - start_time
                    
                    # Extract segment for loudness analysis
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    
                    if start_sample < len(y) and end_sample <= len(y) and start_sample < end_sample:
                        segment = y[start_sample:end_sample]
                        if len(segment) > 0:
                            rms = np.sqrt(np.mean(segment**2))
                            loudness.append(rms)
                    
                    if duration > pause_threshold:
                        pauses.append(duration)
            
            # If no onsets were detected, use the whole audio for loudness
            if len(loudness) == 0 and len(y) > 0:
                rms = np.sqrt(np.mean(y**2))
                loudness.append(rms)
        
        return {
            "words": words,
            "pauses": pauses,
            "loudness": loudness,
            "original_audio": y,
            "sample_rate": sample_rate
        }
    
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def speech_synthesis_loop():
    global is_running, audio_queue
    
    while is_running:
        try:
            if not audio_queue.empty():
                analysis_result = audio_queue.get()
                neutralized_speech = synthesize_neutral_speech(analysis_result)
                play_audio(neutralized_speech, analysis_result["sample_rate"])
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

def synthesize_neutral_speech(analysis_result):
    try:
        words = analysis_result["words"]
        pauses = analysis_result.get("pauses", [])
        loudness = analysis_result.get("loudness", [])
        
        if not words:
            print("No words to synthesize")
            return None
            
        text = " ".join(words)
        print(f"Synthesizing: '{text}'")
        
        inputs = processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            # Use a fixed seed for more consistent voice characteristics
            torch.manual_seed(voice_seed)
            
            # Generate speaker embedding for a more neutral voice
            speaker_embeddings = torch.randn(1, model.config.speaker_embedding_dim)
            
            # Generate speech with vocoder
            speech = model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder  # Use vocoder for better quality
            )
        
        # Convert to numpy for post-processing
        speech_np = speech.numpy()
        
        # Apply pitch shifting for a more neutral voice if librosa is available
        try:
            if pitch_shift_amount != 0:
                print(f"Applying pitch shift: {pitch_shift_amount}")
                speech_np = librosa.effects.pitch_shift(
                    speech_np, 
                    sr=16000, 
                    n_steps=pitch_shift_amount
                )
        except Exception as e:
            print(f"Warning: Pitch shifting failed: {e}")
        
        # Apply other speech characteristics
        modified_speech = apply_speech_characteristics(speech_np, pauses, loudness)
        
        return modified_speech
        
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_speech_characteristics(speech, pauses, loudness):
    """Apply characteristics like pauses and loudness to the speech"""
    try:
        # Ensure we're working with numpy array
        if isinstance(speech, torch.Tensor):
            speech_np = speech.detach().cpu().numpy()
        else:
            speech_np = speech
        
        # Check for NaN or infinity values
        if np.isnan(speech_np).any() or np.isinf(speech_np).any():
            print("Warning: Speech contains NaN or Inf values. Fixing...")
            speech_np = np.nan_to_num(speech_np)
        
        # Apply loudness characteristics
        if loudness and len(loudness) > 0:
            avg_loudness = np.mean(loudness)
            
            # Sanity check the loudness value
            if np.isnan(avg_loudness) or avg_loudness < 0.1:
                print(f"Warning: Invalid loudness value {avg_loudness}, using default")
                avg_loudness = 0.8
            
            # Apply scaling with limits to prevent distortion
            max_scale = min(1.5, max(0.5, avg_loudness))
            print(f"Applying loudness scaling: {max_scale}")
            speech_np = speech_np * max_scale
        
        # Normalize the audio to ensure it's audible
        max_val = np.max(np.abs(speech_np))
        if max_val > 0:
            speech_np = speech_np * (0.9 / max_val)
            print(f"Normalized audio with factor: {0.9 / max_val:.2f}")
        
        # In a full implementation, you would also add pauses at appropriate positions
        # This would require alignment information between the text and audio
        
        return speech_np
        
    except Exception as e:
        print(f"Error applying speech characteristics: {e}")
        import traceback
        traceback.print_exc()
        # Return the original speech as fallback
        return speech if isinstance(speech, np.ndarray) else speech.numpy() if isinstance(speech, torch.Tensor) else None

def convert_to_stereo(mono_audio):
    """Convert mono audio (1 channel) to stereo (2 channels)"""
    return np.column_stack((mono_audio, mono_audio))

def play_audio(audio_data, sample_rate):
    if audio_data is None:
        return
    
    try:
        # Generate a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"neutralized_{timestamp}.wav")
        
        # Ensure we're working with numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        # Ensure we're working with mono audio (VB-Cable works best with mono)
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # Convert stereo to mono
            print("Converting stereo to mono")
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize the audio to make it louder
        if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
            abs_max = np.max(np.abs(audio_data))
            if abs_max > 0:
                audio_data = audio_data * (0.9 / abs_max)
                print(f"Final normalization factor: {0.9 / abs_max:.2f}")
        
        # Validate the audio data
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            print("Warning: Audio contains NaN or Inf values. Fixing...")
            audio_data = np.nan_to_num(audio_data)
        
        # Save the audio file with explicit format for VB-Cable compatibility
        sf.write(output_file, audio_data, sample_rate, subtype='PCM_16')
        print(f"Audio output saved to {output_file}")

        # Simple direct playback approach
        try:
            import subprocess
            
            # Use a simpler approach - direct the sound to default audio output
            # On Windows, this should be the VB-Cable if set as default recording device
            if os.name == 'nt':
                print("Playing audio file directly")
                
                # Safe path handling
                safe_path = output_file.replace('\\', '/')
                
                # Use PowerShell's built-in media player
                ps_command = "Add-Type -AssemblyName presentationCore; "
                ps_command += "$player = New-Object System.Windows.Media.MediaPlayer; "
                ps_command += f"$player.Open('{safe_path}'); "
                ps_command += "$player.Play(); "
                ps_command += "Start-Sleep -s 3; "  # Wait for playback to complete
                
                # Execute PowerShell command
                subprocess.Popen(["powershell", "-Command", ps_command], 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                print("Started audio playback via PowerShell")
                return
        except Exception as e:
            print(f"Error with PowerShell playback: {e}")
            import traceback
            traceback.print_exc()
            
        # Fallback to standard playback if PowerShell approach fails
        try:
            if os.name == 'nt':
                os.startfile(output_file)
                print("Fallback: Started default media player")
            else:
                print("Playback on non-Windows platforms not implemented")
        except Exception as e:
            print(f"Failed to play audio with fallback method: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error processing audio output: {e}")
        import traceback
        traceback.print_exc()

# Clean up resources when script is unloaded
def script_unload():
    global is_running
    is_running = False
    print("Speech Neutralizer Plugin unloaded")
