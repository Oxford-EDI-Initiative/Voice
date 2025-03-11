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

# Global variables
source_name = ""
audio_queue = queue.Queue()
is_running = False
recognizer = speech_rec.Recognizer()
output_device = None
analysis_thread = None
synthesis_thread = None
voice_seed = 42
pitch_shift_amount = -1.5
output_channels = 2
use_direct_output = True
bypass_original_audio = True
energy_threshold = 4000  # Energy threshold for detecting speech
dynamic_energy = True    # Automatically adjust energy threshold

# Create output directory in a safe location
output_dir = os.path.join(tempfile.gettempdir(), "voice_interpreter")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Load pretrained models
def load_models():
    global processor, model, vocoder
    
    print("Loading speech models...")
    try:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
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
    print("Voice Interpreter Plugin loaded")
    try:
        load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")

def script_description():
    return "Voice Interpreter: Transforms your voice in real-time with continuous streaming."

def group_and_filter_output_devices():
    """Group and filter output devices into categories for better organization"""
    # Categories for devices
    categories = {
        "virtual": [],     # Virtual audio cables
        "speakers": [],    # Physical speakers and outputs
        "bluetooth": [],   # Bluetooth devices
        "other": []        # Other devices that don't fit into above categories
    }
    
    # Set to track already processed device names to avoid duplicates
    processed_names = set()
    
    try:
        import sounddevice as sd
        sd_devices = sd.query_devices()
        
        for i, device in enumerate(sd_devices):
            if device['max_output_channels'] > 0:  # Only include output devices
                name = device['name']
                
                # Skip empty device names
                if not name.strip():
                    continue
                
                # Get the first part of the name for better deduplication
                short_name = name.split('(')[0].strip()
                
                # Skip if we've already processed this base name
                if short_name in processed_names:
                    continue
                
                processed_names.add(short_name)
                
                # Classify device by type
                if any(keyword in name.lower() for keyword in ['vb-audio', 'cable', 'virtual']):
                    categories["virtual"].append((name, str(i)))
                elif any(keyword in name.lower() for keyword in ['bluetooth', 'airpods', 'hands-free']):
                    categories["bluetooth"].append((name, str(i)))
                elif any(keyword in name.lower() for keyword in ['speaker', 'headphone', 'output', 'realtek']):
                    categories["speakers"].append((name, str(i)))
                else:
                    categories["other"].append((name, str(i)))
        
    except Exception as e:
        print(f"Error grouping output devices: {e}")
        # Fallback options if we couldn't get actual devices
        categories["speakers"].append(("Default Output", "0"))
        categories["virtual"].append(("CABLE Input (VB-Audio Virtual Cable)", "1"))
    
    return categories

def script_properties():
    props = obs.obs_properties_create()
    
    # Audio Source selection (from OBS)
    source_list = obs.obs_properties_add_list(props, "source", "OBS Audio Source", obs.OBS_COMBO_TYPE_EDITABLE, obs.OBS_COMBO_FORMAT_STRING)
    sources = obs.obs_enum_sources()
    if sources:
        for source in sources:
            source_id = obs.obs_source_get_name(source)
            obs.obs_property_list_add_string(source_list, source_id, source_id)
        obs.source_list_release(sources)
    
    # Output Device selection - organized in categories
    output_devices = obs.obs_properties_add_list(props, "output_device", "Output Device", obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING)
    
    # Get devices grouped by category
    device_categories = group_and_filter_output_devices()
    
    # Add virtual audio devices first (most likely to be used for interpretation)
    if device_categories["virtual"]:
        obs.obs_property_list_add_string(output_devices, "-- Virtual Audio Devices --", "")
        for name, idx in device_categories["virtual"]:
            obs.obs_property_list_add_string(output_devices, name, idx)
    
    # Add speaker outputs
    if device_categories["speakers"]:
        obs.obs_property_list_add_string(output_devices, "-- Speakers & Headphones --", "")
        for name, idx in device_categories["speakers"]:
            obs.obs_property_list_add_string(output_devices, name, idx)
    
    # Add bluetooth devices
    if device_categories["bluetooth"]:
        obs.obs_property_list_add_string(output_devices, "-- Bluetooth Devices --", "")
        for name, idx in device_categories["bluetooth"]:
            obs.obs_property_list_add_string(output_devices, name, idx)
    
    # Add other devices
    if device_categories["other"]:
        obs.obs_property_list_add_string(output_devices, "-- Other Devices --", "")
        for name, idx in device_categories["other"]:
            obs.obs_property_list_add_string(output_devices, name, idx)
    
    # Add output channels selection
    channels = obs.obs_properties_add_list(props, "output_channels", "Output Channels", obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_INT)
    obs.obs_property_list_add_int(channels, "Mono (1)", 1)
    obs.obs_property_list_add_int(channels, "Stereo (2)", 2)
    
    # Create sections for better organization
    voice_group = obs.obs_properties_create()
    obs.obs_properties_add_int_slider(voice_group, "voice_seed", "Voice Characteristics (Seed)", 1, 100, 1)
    obs.obs_properties_add_float_slider(voice_group, "pitch_shift", "Voice Pitch Adjustment", -4.0, 4.0, 0.5)
    obs.obs_properties_add_group(props, "voice_settings", "Voice Settings", obs.OBS_GROUP_NORMAL, voice_group)
    
    # Performance settings - In a separate group
    perf_group = obs.obs_properties_create()
    obs.obs_properties_add_int_slider(perf_group, "energy_threshold", "Microphone Sensitivity", 1000, 8000, 100)
    obs.obs_properties_add_bool(perf_group, "dynamic_energy", "Auto-adjust Microphone Sensitivity")
    obs.obs_properties_add_group(props, "performance_settings", "Performance Settings", obs.OBS_GROUP_NORMAL, perf_group)
    
    # Behavior settings
    behavior_group = obs.obs_properties_create()
    obs.obs_properties_add_bool(behavior_group, "bypass_original", "Bypass Original Voice (Only Output Changed Voice)")
    obs.obs_properties_add_bool(behavior_group, "direct_output", "Use Direct Output (Lower Latency)")
    obs.obs_properties_add_group(props, "behavior_settings", "Behavior Settings", obs.OBS_GROUP_NORMAL, behavior_group)
    
    # Add control buttons
    obs.obs_properties_add_button(props, "start_button", "Start Interpretation", start_processing_button)
    obs.obs_properties_add_button(props, "stop_button", "Stop Interpretation", stop_processing_button)
    
    return props

def script_update(settings):
    global source_name, output_device, output_channels, voice_seed, pitch_shift_amount
    global use_direct_output, bypass_original_audio, energy_threshold, dynamic_energy
    
    source_name = obs.obs_data_get_string(settings, "source")
    output_device = obs.obs_data_get_string(settings, "output_device")
    
    # Get voice settings
    voice_seed = obs.obs_data_get_int(settings, "voice_seed")
    pitch_shift_amount = obs.obs_data_get_double(settings, "pitch_shift")
    
    # Get output channels
    output_channels = obs.obs_data_get_int(settings, "output_channels")
    if output_channels not in [1, 2]:
        output_channels = 2  # Default to stereo
    
    # Get performance settings
    energy_threshold = obs.obs_data_get_int(settings, "energy_threshold")
    dynamic_energy = obs.obs_data_get_bool(settings, "dynamic_energy")
    use_direct_output = obs.obs_data_get_bool(settings, "direct_output")
    bypass_original_audio = obs.obs_data_get_bool(settings, "bypass_original")
    
    # If we're bypassing the original audio, need to handle source muting
    if bypass_original_audio and source_name:
        source = obs.obs_get_source_by_name(source_name)
        if source:
            # Setting up to mute the original source when processing starts
            obs.obs_source_release(source)

def script_defaults(settings):
    obs.obs_data_set_default_int(settings, "output_channels", 2)  # Default to stereo
    obs.obs_data_set_default_int(settings, "voice_seed", 42)
    obs.obs_data_set_default_double(settings, "pitch_shift", -1.5)
    obs.obs_data_set_default_int(settings, "energy_threshold", 4000)
    obs.obs_data_set_default_bool(settings, "dynamic_energy", True)
    obs.obs_data_set_default_bool(settings, "direct_output", True)
    obs.obs_data_set_default_bool(settings, "bypass_original", True)

def start_processing_button(props, prop):
    start_processing()
    return True

def stop_processing_button(props, prop):
    stop_processing()
    return True

def mute_original_source(mute=True):
    """Mute or unmute the original audio source if needed"""
    if bypass_original_audio and source_name:
        source = obs.obs_get_source_by_name(source_name)
        if source:
            # Get the mute setting first
            current_mute = obs.obs_source_muted(source)
            
            # Only change if different from requested state
            if current_mute != mute:
                obs.obs_source_set_muted(source, mute)
                print(f"Original source {source_name} {'muted' if mute else 'unmuted'}")
            
            obs.obs_source_release(source)

# Audio capture and processing
def start_processing():
    global is_running, analysis_thread, synthesis_thread, recognizer
    
    if is_running:
        return
    
    is_running = True
    
    # Mute the original source if needed
    mute_original_source(True)
    
    # Configure the recognizer
    recognizer = speech_rec.Recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy
    
    # Start continuous listening thread
    analysis_thread = threading.Thread(target=continuous_listen_loop)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    # Start synthesis thread
    synthesis_thread = threading.Thread(target=speech_synthesis_loop)
    synthesis_thread.daemon = True
    synthesis_thread.start()
    
    print("Interpretation processing started")

def stop_processing():
    global is_running
    
    is_running = False
    
    # Unmute the original source if we muted it
    mute_original_source(False)
    
    print("Interpretation processing stopped")

def continuous_listen_loop():
    """Continuously listen for speech using a phrase-based approach"""
    global is_running, recognizer
    
    print("Starting continuous listening...")
    
    try:
        # Create a microphone instance for each phrase to avoid context manager issues
        while is_running:
            try:
                # Create a new microphone instance for each iteration
                mic = speech_rec.Microphone()
                
                # Use listen to capture full phrases
                with mic as source:
                    print("Listening for speech...")
                    # Short calibration for background noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Listen for a phrase (automatically detects when speech starts and ends)
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)
                    print("Phrase detected, processing...")
                
                # Process the audio in a separate thread to keep listening loop active
                processing_thread = threading.Thread(target=process_audio, args=(audio,))
                processing_thread.daemon = True
                processing_thread.start()
                
            except Exception as e:
                print(f"Error in continuous listening: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.2)  # Brief pause before retrying
    except Exception as e:
        print(f"Fatal error in listening loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Continuous listening stopped")

def process_audio(audio_data):
    """Process a complete audio phrase"""
    try:
        # Try speech recognition
        text = ""
        try:
            text = recognizer.recognize_google(audio_data)
            if text:
                print(f"Recognized: '{text}'")
        except speech_rec.UnknownValueError:
            print("No speech detected in phrase")
            return
        except speech_rec.RequestError as e:
            print(f"Recognition error: {e}")
            return
        except Exception as e:
            print(f"Other recognition error: {e}")
            return
        
        # Only proceed if text was recognized
        if not text:
            return
            
        # Add to processing queue
        audio_queue.put({
            "text": text
        })
        
    except Exception as e:
        print(f"Error processing audio: {e}")

def speech_synthesis_loop():
    global is_running, audio_queue
    
    while is_running:
        try:
            if not audio_queue.empty():
                analysis_result = audio_queue.get()
                neutralized_speech = synthesize_neutral_speech(analysis_result)
                if neutralized_speech is not None:
                    play_audio(neutralized_speech, 16000)  # Use fixed sample rate
            else:
                time.sleep(0.05)  # Short sleep to prevent CPU spinning
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
            time.sleep(0.1)

def synthesize_neutral_speech(analysis_result):
    try:
        text = analysis_result.get("text", "")
        
        if not text:
            return None
            
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
                vocoder=vocoder
            )
        
        # Convert to numpy for post-processing
        speech_np = speech.numpy()
        
        # Apply pitch shifting for voice modification
        if pitch_shift_amount != 0:
            speech_np = librosa.effects.pitch_shift(
                speech_np, 
                sr=16000, 
                n_steps=pitch_shift_amount
            )
        
        # Normalize the audio to ensure it's audible
        max_val = np.max(np.abs(speech_np))
        if max_val > 0:
            speech_np = speech_np * (0.9 / max_val)
        
        return speech_np
        
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None

def play_audio(audio_data, sample_rate):
    if audio_data is None:
        return
    
    try:
        # Generate a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"interpreted_{timestamp}.wav")
        
        # Ensure we're working with numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        # Convert to stereo if needed
        if output_channels == 2 and len(audio_data.shape) == 1:
            audio_data = np.column_stack((audio_data, audio_data))
        
        # Normalize the audio again before output
        if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
            abs_max = np.max(np.abs(audio_data))
            if abs_max > 0:
                audio_data = audio_data * (0.9 / abs_max)
        
        # Validate the audio data
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            audio_data = np.nan_to_num(audio_data)
        
        # Save the audio file first
        sf.write(output_file, audio_data, sample_rate, subtype='PCM_16')
        
        # Use direct output approach for lower latency
        if use_direct_output:
            try:
                # Try direct playback using sounddevice if available
                import sounddevice as sd
                
                # If output_device was provided as an index string, convert to int
                output_idx = None
                if output_device and output_device.isdigit():
                    output_idx = int(output_device)
                
                # Play the audio directly
                sd.play(audio_data, sample_rate, device=output_idx)
                print(f"Direct audio playback to device {output_idx or 'default'}")
                return
            except ImportError:
                print("sounddevice not available, falling back to file playback")
            except Exception as e:
                print(f"Error with direct playback: {e}")
        
        # Fallback to file-based playback
        if os.name == 'nt':
            try:
                import subprocess
                
                # Use PowerShell for playback
                safe_path = output_file.replace('\\', '/')
                ps_command = f"(New-Object Media.SoundPlayer '{safe_path}').PlaySync()"
                
                subprocess.Popen(["powershell", "-Command", ps_command], 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                print("Started audio playback via PowerShell")
            except Exception as e:
                print(f"Error with PowerShell playback: {e}")
                os.startfile(output_file)
                print("Fallback: Started default media player")
        else:
            print("Playback on non-Windows platforms not implemented")
        
    except Exception as e:
        print(f"Error processing audio output: {e}")

# Clean up resources when script is unloaded
def script_unload():
    global is_running
    is_running = False
    
    # Ensure original source is unmuted
    mute_original_source(False)
    
    print("Interpretation Plugin unloaded")
