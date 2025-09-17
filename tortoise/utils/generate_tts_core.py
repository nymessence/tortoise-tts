import os
import sys
import logging
import torch
import torchaudio
import glob
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import write
import re
import json
import time
from gtts import gTTS
import tempfile

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
    Main generation logic for a single process with fallback.
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
        success = False
        
        # Tortoise-TTS Generation Loop
        for attempt in range(3):
            try:
                logging.info(f"[{original_index + 1}/{len(all_lines)}] Attempt {attempt + 1}: Generating audio for segment: '{line}'...")
                gen = tts.tts_with_preset(
                    text=line,
                    voice_samples=voice_samples,
                    preset=args.preset,
                    use_deterministic_seed=True,
                    cvvp_amount=0.5,
                    num_autoregressive_samples=8
                )
                
                if gen is None:
                    raise RuntimeError("TTS generation returned None, possibly due to an internal failure.")

                if isinstance(gen, tuple):
                    gen = gen[0]
                
                torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)
                logging.info(f"✅ Audio saved to '{output_path}'.")
                success = True
                break # Exit the retry loop on success
            
            except Exception as e:
                logging.error(f"❌ Attempt {attempt + 1} failed for line '{line}': {e}")
                if attempt < 2:
                    logging.warning("Retrying after a short delay...")
                    time.sleep(5)
        
        # Fallback Logic
        if not success:
            logging.critical(f"🛑 All Tortoise-TTS attempts failed. Initiating fallback generation for line '{line}'.")
            try:
                # Use gTTS for fallback. You can swap this for any other reliable engine.
                tts_fallback = gTTS(text=line, lang='en')
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts_fallback.save(temp_file.name)
                
                # Convert the MP3 to a WAV file to match your pipeline's format
                audio = AudioSegment.from_mp3(temp_file.name)
                audio.export(output_path, format="wav")
                os.remove(temp_file.name)
                
                logging.info(f"✅ Fallback audio saved to '{output_path}'.")
            except Exception as e:
                logging.critical(f"❌ Fallback generation also failed for line '{line}': {e}. Creating a silent audio file.")
                # Create a silent audio placeholder if everything fails
                AudioSegment.silent(duration=1000).export(output_path, format="wav")
                
    logging.info("All lines for this process have been generated.")

def preprocess_script(script_path, preprocessed_script_path, instructions_path=None):
    """
    Cleans the script for TTS generation and optionally extracts command tokens.
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        cleaned_lines = []
        instructions = []
        
        legacy_mode = instructions_path is None or not any(re.search(r'\[.*?\]', line) for line in lines)
        
        for i, line in enumerate(lines):
            if legacy_mode:
                cleaned_line = re.sub(r'\[.*?\]', '', line)
            else:
                tokens = re.findall(r'\[.*?\]', line)
                if tokens:
                    for token in tokens:
                        instructions.append({"line": i, "token": token})
                    cleaned_line = re.sub(r'\[.*?\]', '', line)
                else:
                    cleaned_line = line

            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line.strip())
        
        with open(preprocessed_script_path, "w", encoding="utf-8") as f_out:
            f_out.write('\n'.join(cleaned_lines))
        logging.info(f"✅ Preprocessed script ready: {preprocessed_script_path}")
        
        if not legacy_mode:
            with open(instructions_path, "w", encoding="utf-8") as f_out:
                json.dump(instructions, f_out, indent=4)
            logging.info(f"✅ Command instructions saved: {instructions_path}")
        
    except Exception as e:
        logging.error(f"❌ Failed to preprocess script: {e}")
        sys.exit(1)

def stitch_audio(lines_dir, num_lines, output_path):
    """
    Stitches generated audio segments together, with a 1-second gap between lines.
    """
    try:
        all_audio = []
        GAP_DURATION_MS = 1000
        silent_gap = AudioSegment.silent(duration=GAP_DURATION_MS)

        for i in range(num_lines):
            line_file = os.path.join(lines_dir, f"{str(i).zfill(4)}.wav")
            if not os.path.exists(line_file):
                logging.warning(f"Audio file not found for line {i}. Adding silence.")
                all_audio.append(AudioSegment.silent(duration=1000))
            else:
                all_audio.append(AudioSegment.from_wav(line_file))
        
        if not all_audio:
            raise RuntimeError("No audio lines loaded for stitching!")
        
        final_audio = all_audio[0]
        for audio in all_audio[1:]:
            final_audio = final_audio.append(silent_gap, crossfade=0)
            final_audio = final_audio.append(audio, crossfade=150)
        
        final_audio.export(output_path, format="wav")
        logging.info(f"✅ Final stitched audio saved: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to stitch audio: {e}", exc_info=True)
        raise

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
    
def run_generation_chunked(args):
    """
    Loads model once, then generates all lines in [args.start_idx, args.end_idx).
    Maximizes TPU utilization by keeping model resident and processing sequentially.
    Includes error handling, retries, and fallback to silent audio.
    """
    import torch
    import torch_xla.core.xla_model as xm
    import os
    import torchaudio
    import time

    device = xm.xla_device()

    # 🔥 LOAD MODEL ONCE PER PROCESS
    logging.info(f"[Core {args.rank}] Loading TTS model...")
    from tortoise.api import TextToSpeech
    tts_model = TextToSpeech(
        models_dir=args.models_dir,
        voice=args.voice,
        preset=args.preset,
        device=str(device)
    )
    logging.info(f"[Core {args.rank}] Model loaded successfully.")

    # Load all lines
    with open(args.lines_file, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    # Generate silent fallback (1 sec @ 22050 Hz)
    silent_fallback = torch.zeros(1, 22050)

    # Process assigned chunk
    for i in range(args.start_idx, args.end_idx):
        line = all_lines[i]
        output_path = os.path.join(args.output_dir, f"line_{i+1:04d}.wav")

        # Skip if already exists (for resuming)
        if os.path.exists(output_path):
            logging.info(f"[Core {args.rank}] Skipping line {i+1} (already exists)")
            continue

        logging.info(f"[Core {args.rank}] Generating line {i+1}/{args.total_lines}: {line[:60]}...")

        success = False
        for attempt in range(3):
            try:
                # 🔊 Generate audio
                audio = tts_model.generate(
                    line,
                    diffusion_iterations=args.diffusion_iterations,
                    num_autoregressive_samples=args.num_autoregressive_samples,
                    voice=args.voice,
                    preset=args.preset
                )

                # Handle list output
                if isinstance(audio, list):
                    if len(audio) == 0:
                        raise ValueError("Generated audio list is empty.")
                    audio = audio[0]  # Take first sample

                # Validate tensor
                if not isinstance(audio, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(audio)}")

                # Normalize shape: [samples] or [1, samples] → [1, samples]
                audio = audio.squeeze()  # Remove extra dims
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add channel dim

                # Save to disk
                torchaudio.save(output_path, audio.cpu(), sample_rate=22050)
                logging.info(f"[Core {args.rank}] ✅ Saved: {output_path}")
                success = True
                break  # Exit retry loop

            except Exception as e:
                logging.error(f"[Core {args.rank}] ❌ Attempt {attempt+1} failed for line {i+1}: {e}")
                if attempt < 2:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logging.warning(f"[Core {args.rank}] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logging.critical(f"[Core {args.rank}] 🛑 All retries failed. Saving silent fallback.")
                    torchaudio.save(output_path, silent_fallback, sample_rate=22050)

        # Optional: Force XLA sync to ensure progress is visible
        xm.mark_step()

    logging.info(f"[Core {args.rank}] ✅ Finished chunk [{args.start_idx} - {args.end_idx}).")

def run_generation_for_spawn(index, process_args_list):
    """
    Each TPU core processes a chunk of lines to maximize memory and compute utilization.
    """
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Get this process's assigned chunk
    flags = process_args_list[index]

    class Args:
        pass
    args = Args()

    # Carry over all config
    args.lines_file = flags['lines_file']
    args.output_dir = flags['output_dir']
    args.hardware = flags['hardware']
    args.voice = flags['voice']
    args.preset = flags['preset']
    args.models_dir = flags['models_dir']
    args.diffusion_iterations = flags.get('diffusion_iterations', 16)
    args.num_autoregressive_samples = flags.get('num_autoregressive_samples', 1)

    # NEW: Use assigned chunk — not strided!
    args.start_idx = flags['start_idx']
    args.end_idx = flags['end_idx']
    args.total_lines = flags['total_lines']
    args.rank = flags['rank']

    logging.info(f"▶️ TPU Core {args.rank} starting. Processing lines [{args.start_idx} - {args.end_idx})")

    # Run generation — this function must now loop internally over the chunk!
    run_generation_chunked(args)

    logging.info(f"✅ TPU Core {args.rank} finished processing {args.end_idx - args.start_idx} lines.")

if __name__ == "__main__":
    logging.error("This is a library file and should not be run directly. Please use a driver script.")
    sys.exit(1)
