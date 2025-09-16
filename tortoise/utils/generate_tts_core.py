import os
import sys
import logging
import torch
import torchaudio
import glob
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import write

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s')

# For TPU support, will be imported only when hardware is 'tpu'
xm = None
xmp = None

# ========================
# Core TTS Functions
# ========================

def run_generation(args):
    """
    Main generation logic for a single process.
    """
    global xm
    try:
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_audio
    except ImportError:
        logging.error("Could not import Tortoise TTS. Ensure the repository is in your PYTHONPATH.")
        sys.exit(1)

    if args.hardware == 'tpu':
        if xm is None:
            import torch_xla.core.xla_model as xla_model
            xm = xla_model
        device = xm.xla_device()
        logging.info(f"TPU device initialized in worker: {device}")
    elif args.hardware == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using GPU for generation.")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU for generation.")

    tts = TextToSpeech(models_dir=args.models_dir)
    if hasattr(tts, 'to'):
        tts.to(device)
        logging.info("Main TTS model moved to device successfully.")
    else:
        logging.warning("TextToSpeech model does not have a .to() method. This may cause issues on non-CPU devices.")

    try:
        voice_dir = os.path.join(args.models_dir, 'tortoise/voices', args.voice)
        if not os.path.isdir(voice_dir):
            raise FileNotFoundError(f"Voice directory not found: {voice_dir}")
        voice_files = [os.path.join(voice_dir, f) for f in os.listdir(voice_dir) if f.endswith('.wav')]
        if not voice_files:
            raise ValueError(f"No .wav files found in voice directory: {voice_dir}")
        logging.info(f"Found {len(voice_files)} reference audio files for voice '{args.voice}'.")
        TARGET_SAMPLE_RATE = 22050
        voice_samples = []
        for f in voice_files:
            audio_tensor = load_audio(f, TARGET_SAMPLE_RATE).to(device)
            voice_samples.append(audio_tensor)
        logging.info(f"Successfully loaded voice samples for '{args.voice}'.")
    except Exception as e:
        logging.error(f"Failed to load voice samples for '{args.voice}': {e}", exc_info=True)
        raise

    try:
        with open(args.lines_file, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Text file not found: {args.lines_file}")
        sys.exit(1)

    if not all_lines:
        logging.error("No lines to process. Exiting.")
        return

    start_index = args.start_idx
    step = args.step
    lines_to_process = all_lines[start_index::step]
    logging.info(f"Found {len(all_lines)} lines in total.")
    logging.info(f"This process will handle {len(lines_to_process)} lines, starting from index {start_index} with a step of {step}.")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory '{args.output_dir}' is ready.")

    for i, line in enumerate(lines_to_process):
        original_index = start_index + i * step
        output_path = os.path.join(args.output_dir, f"{original_index:04d}.wav")
        logging.info(f"[{original_index + 1}/{len(all_lines)}] Generating audio for segment: '{line}'...")
        gen = tts.tts_with_preset(
            text=line,
            voice_samples=voice_samples,
            preset=args.preset,
            use_deterministic_seed=True,
            cvvp_amount=0.5,
            num_autoregressive_samples=8
        )
        if isinstance(gen, tuple):
            gen = gen[0]
        torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)
        logging.info(f"Audio saved to '{output_path}'.")
    logging.info("All lines for this process have been generated.")

def preprocess_script(script_path, preprocessed_script_path):
    """
    Cleans and tokenizes the script into sentences.
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f_in:
            content = f_in.read()
        sentences = content.replace('"', '').replace("'", '').replace(";", '.').replace("...", ".").split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        with open(preprocessed_script_path, "w", encoding="utf-8") as f_out:
            f_out.write('\n'.join(sentences))
        logging.info(f"✅ Preprocessed script ready: {preprocessed_script_path}")
    except Exception as e:
        logging.error(f"❌ Failed to preprocess script: {e}")
        sys.exit(1)

def post_process_audio(out_dir, preprocessed_script_path):
    """
    Stitches generated audio, adds binaural beats, and exports the final file.
    """
    story_file = os.path.join(out_dir, "story.wav")
    output_dir = os.path.join(out_dir, "lines")
    final_output = os.path.join(out_dir, "story.wav")
    hemisync_output = os.path.join(out_dir, "hemisync.wav")
    mixed_output = os.path.join(out_dir, "story_with_beats.wav")

    with open(preprocessed_script_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        total_lines = len(lines)
    logging.info(f"Found {total_lines} lines to stitch.")
    all_audio = []
    for i in range(total_lines):
        line_file = os.path.join(output_dir, f"{str(i).zfill(4)}.wav")
        if not os.path.exists(line_file):
            logging.warning(f"Audio file not found for line {i}. Adding silence.")
            silent = AudioSegment.silent(duration=1000)
            all_audio.append(silent)
        else:
            all_audio.append(AudioSegment.from_wav(line_file))
    if len(all_audio) == 0:
        raise RuntimeError("No audio lines loaded!")
    final_audio = all_audio[0]
    for audio in all_audio[1:]:
        final_audio = final_audio.append(audio, crossfade=150)
    final_audio.export(final_output, format="wav")
    logging.info(f"✅ Final stitched audio saved: {final_output}")

    story = AudioSegment.from_wav(final_output)
    duration = story.duration_seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    base_freq = 200
    beat_freq = 7
    left = np.sin(2 * np.pi * base_freq * t)
    right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
    stereo = np.vstack((left, right)).T
    stereo = stereo / np.max(np.abs(stereo))
    stereo_int16 = (stereo * 32767).astype(np.int16)
    write(hemisync_output, sample_rate, stereo_int16)
    logging.info(f"✅ Hemisync generated: {hemisync_output}")
    hemisync = AudioSegment.from_wav(hemisync_output)
    if len(hemisync) < len(story):
        repeat_count = int(len(story) / len(hemisync)) + 1
    hemisync = hemisync * repeat_count
    hemisync = hemisync[:len(story)]
    mixed = story.overlay(hemisync - 8)
    mixed.export(mixed_output, format="wav")
    logging.info("✅ Mixed audio created")
    logging.info("✅ All audio processing complete.")
    
def run_generation_local(flags):
    """
    Local generation function for GPU/CPU fallback.
    """
    class Args:
        pass
    args = Args()
    args.lines_file = flags['lines_file']
    args.output_dir = flags['output_dir']
    args.hardware = flags['hardware']
    args.voice = flags['voice']
    args.preset = flags['preset']
    args.models_dir = flags['models_dir']
    args.start_idx = 0
    args.step = 1
    run_generation(args)

def run_generation_for_spawn(index, flags):
    """
    Sets up arguments from flags provided by xmp.spawn and runs generation.
    This function is the entry point for each of the TPU core processes.
    """
    import torch_xla.core.xla_model as xla_model
    import torch_xla.distributed.xla_multiprocessing as xla_multiprocessing
    global xm
    global xmp
    xm = xla_model
    xmp = xla_multiprocessing
    
    class Args:
        pass
    args = Args()

    args.lines_file = flags['lines_file']
    args.output_dir = flags['output_dir']
    args.hardware = flags['hardware']
    args.voice = flags['voice']
    args.preset = flags['preset']
    args.models_dir = flags['models_dir']

    args.step = xm.xrt_world_size()
    args.start_idx = xm.get_ordinal()

    logging.info(f"Worker {args.start_idx} starting on {args.hardware} with step {args.step}.")
    run_generation(args)

if __name__ == "__main__":
    # This file should not be run directly, as it is a library of functions.
    # It should be imported by a driver script like `kaggle_demo.py`.
    logging.error("This is a library file and should not be run directly. Please use a driver script.")
    sys.exit(1)
