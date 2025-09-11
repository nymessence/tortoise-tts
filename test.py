import os
import sys
import torch
import torchaudio
from tortoise import api
from tortoise.utils.audio import load_voice
from scipy.io.wavfile import write

# Define a simple function to log progress.
def log(message):
    print(f"[TEST SCRIPT] {message}")

# Main test function.
def run_tortoise_test():
    try:
        api.prepare_tpu()
    
        log("Starting Tortoise-TTS test...")

        # 1. Initialize the TextToSpeech model.
        # This is where your dependency fixes will be tested.
        log("Initializing TextToSpeech model...")
        # Note: The 'half=True' and other parameters are included here to test
        # compatibility with your proposed dependency updates.
        tts = api.TextToSpeech(use_deepspeed=False, kv_cache=False, half=False)
        log("Model initialized successfully.")

        # 2. Define the text to be spoken.
        text = "Hello, world! This is a test of the Tortoise text-to-speech system."
        log(f"Text to be generated: '{text}'")

        # 3. Generate the audio.
        # We will use the 'random' voice to ensure the core generation pipeline is working.
        log("Generating audio using the 'random' voice...")
        gen = tts.tts_with_preset(text, preset='ultra_fast')
        log("Audio generation complete.")

        # 4. Save the generated audio.
        output_file = "test_output.wav"
        torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
        log(f"Audio saved to '{output_file}'.")

        # 5. Verify the file exists.
        if os.path.exists(output_file):
            log("Test passed! The output file was created successfully.")
            return True
        else:
            log("Test failed. The output file was not created.")
            return False

    except Exception as e:
        log(f"An error occurred during the test: {e}")
        return False

# Run the test.
if __name__ == "__main__":
    if run_tortoise_test():
        print("\nSUCCESS: All tests completed without error.")
    else:
        print("\nFAILURE: One or more tests failed.")

