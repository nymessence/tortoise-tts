import os
import sys
import logging
import torch
import torch_xla.distributed.xla_multiprocessing as xmp

# Import the core functions
from tortoise.api import run_generation_tpu, run_generation_local
from tortoise.utils.generate_tts_core import preprocess_script, stitch_audio
# Import from the new file
from tortoise.binaural import add_binaural_beats, add_hemi_sync, BINAURAL_BEATS 

def main():
    # Define and centralize all file paths here
    OUTPUT_BASE_DIR = "/kaggle/working"
    OUT_DIR = os.path.join(OUTPUT_BASE_DIR, "speech")
    SCRIPT_PATH = os.path.join(OUTPUT_BASE_DIR, "story_script.txt")
    PREPROCESSED_SCRIPT_PATH = os.path.join(OUTPUT_BASE_DIR, "story_script_preprocessed.txt")
    TTS_DIR = "/tmp/tortoise-tts"
    
    LINES_DIR = os.path.join(OUT_DIR, "lines")
    STORY_AUDIO_PATH = os.path.join(OUT_DIR, "story.wav")
    HEMISYNC_PATH = os.path.join(OUT_DIR, "story_with_beats.wav")
    
    preprocess_script(SCRIPT_PATH, PREPROCESSED_SCRIPT_PATH)
    
    with open(PREPROCESSED_SCRIPT_PATH, 'r', encoding='utf-8') as f:
        total_lines = len([line.strip() for line in f if line.strip()])
    
    flags = {
        'lines_file': PREPROCESSED_SCRIPT_PATH,
        'output_dir': LINES_DIR, 
        'hardware': 'tpu',
        'voice': "nya",
        'preset': "fast",
        'models_dir': TTS_DIR
    }

    if 'TPU_PROCESSES_COUNT' in os.environ and 'TPU_NAME' in os.environ:
        try:
            logging.info("✅ TPU detected. Starting generation with new API.")
            run_generation_tpu(flags)
        except Exception as e:
            logging.warning(f"⚠️ TPU multiprocessing failed ({e}). Falling back to single-process generation.")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            flags['hardware'] = device
            run_generation_local(flags)
    else:
        logging.info("⚠️ TPU not detected. Falling back to single-process generation.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        flags['hardware'] = device
        run_generation_local(flags)

    stitch_audio(LINES_DIR, total_lines, STORY_AUDIO_PATH)
    
    # Use the new, dedicated Hemi-Sync function
    # You can customize the layered frequencies here if you want
    add_hemi_sync(STORY_AUDIO_PATH, HEMISYNC_PATH, beat_freqs=[BINAURAL_BEATS['theta'], BINAURAL_BEATS['alpha']])
    
    logging.info("✅ All processing complete. Final files are in the 'speech' directory.")

if __name__ == "__main__":
    main()
