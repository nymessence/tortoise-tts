import logging
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write
import re 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s')

# Dictionary of common binaural beat frequencies
BINAURAL_BEATS = {
    'delta': 2,    # Deep sleep, healing
    'theta': 6,    # Meditation, intuition, deep relaxation
    'alpha': 10,   # Relaxed, focused, calm
    'beta': 18,    # Alert, active, concentration
    'gamma': 40    # High-level cognitive processing, insight
}

def _generate_binaural_beats_audio(duration_s, sample_rate, base_freq, beat_freq):
    """
    Helper function to generate a single binaural beat AudioSegment.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    left = np.sin(2 * np.pi * base_freq * t)
    right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
    stereo = np.vstack((left, right)).T
    stereo = stereo / np.max(np.abs(stereo))
    stereo_int16 = (stereo * 32767).astype(np.int16)
    
    return AudioSegment(
        stereo_int16.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=stereo_int16.dtype.itemsize, 
        channels=2
    )

def add_binaural_beats(story_path, beats_path, mixed_output_path, base_freq=200, beat_freq=7):
    """
    Generates single-frequency binaural beats and mixes them with the story audio.
    """
    try:
        story = AudioSegment.from_wav(story_path)
        hemisync = _generate_binaural_beats_audio(story.duration_seconds, 44100, base_freq, beat_freq)
        
        hemisync = hemisync[:len(story)]
        
        mixed = story.overlay(hemisync - 8)
        mixed.export(mixed_output_path, format="wav")
        logging.info(f"✅ Binaural beats (base: {base_freq}Hz, beat: {beat_freq}Hz) mixed: {mixed_output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to add binaural beats: {e}", exc_info=True)
        raise

def add_hemi_sync(story_path, mixed_output_path, beat_freqs=[6, 10], base_freq=200):
    """
    Simulates Hemi-Sync by layering multiple binaural beat frequencies with a single base frequency.
    """
    try:
        story = AudioSegment.from_wav(story_path)
        sample_rate = 44100

        combined_beats = AudioSegment.silent(duration=len(story), frame_rate=sample_rate)
        for beat_freq in beat_freqs:
            beat_track = _generate_binaural_beats_audio(story.duration_seconds, sample_rate, base_freq, beat_freq)
            combined_beats = combined_beats.overlay(beat_track, gain_during_overlay=-6)
            logging.info(f"-> Layered a beat frequency of {beat_freq}Hz.")
        
        mixed = story.overlay(combined_beats - 8)
        mixed.export(mixed_output_path, format="wav")
        logging.info(f"✅ Hemi-Sync audio created with layered frequencies: {beat_freqs} and base frequency: {base_freq}Hz")

    except Exception as e:
        logging.error(f"❌ Failed to create Hemi-Sync audio: {e}", exc_info=True)
        raise

def parse_beat_token(token_string):
    """
    Parses a beat token of the format: [BEAT: carrier; beat1, beat2, ...].
    Returns the carrier frequency (int) and a list of beat frequencies (list of ints).
    """
    pattern = r'\[BEAT:\s*(\d+)\s*;\s*([\d,\s]+)\s*]'
    match = re.search(pattern, token_string)

    if not match:
        raise ValueError(f"Invalid BEAT token format: {token_string}")

    carrier_freq = int(match.group(1))
    beat_freqs = [int(f.strip()) for f in match.group(2).split(',')]

    return carrier_freq, beat_freqs
