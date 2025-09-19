import os
import sys
import logging
import torch
import torchaudio
import re
import time
import json
import tempfile
from gtts import gTTS
from pydub import AudioSegment

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s')

# ========================
# Core TTS Functions
# ========================

def run_generation(args):
    """
    Main generation logic for a single process with fallback.
    """
    try:
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_audio
    except ImportError:
        logging.error("Could not import Tortoise TTS. Ensure the repository is in your PYTHONPATH.")
        sys.exit(1)

    # This is the CRITICAL FIX. The import is now conditional.
    if args.hardware == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            logging.info(f"TPU device initialized in worker: {device}")
        except ImportError as e:
            logging.error(f"Failed to import torch_xla for TPU: {e}. Falling back to GPU/CPU.")
            args.hardware = 'gpu' # Fallback path
    
    if args.hardware == 'gpu' and torch.cuda.is_available():
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
                tts_fallback = gTTS(text=line, lang='en')
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts_fallback.save(temp_file.name)
                
                audio = AudioSegment.from_mp3(temp_file.name)
                audio.export(output_path, format="wav")
                os.remove(temp_file.name)
                
                logging.info(f"✅ Fallback audio saved to '{output_path}'.")
            except Exception as e:
                logging.critical(f"❌ Fallback generation also failed for line '{line}': {e}. Creating a silent audio file.")
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


def save_audio_fallback(path, tensor, sample_rate, rank=None):
    """
    Saves audio tensor using torchcodec (if available) -> falls back to torchaudio.
    Uses 192 kbps for high quality.
    """
    try:
        from torchcodec.encoders import AudioEncoder

        # Normalize to [-1, 1]
        if tensor.abs().max() > 1.0:
            tensor = tensor / tensor.abs().max()

        # Ensure shape [channels, samples]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        encoder = AudioEncoder(tensor, sample_rate=sample_rate)
        encoder.to_file(str(path), bit_rate=192000)

        if rank is not None:
            logging.debug(f"[Core {rank}] ✅ Saved with torchcodec: {path}")
        else:
            logging.debug(f"✅ Saved with torchcodec: {path}")

    except Exception as e:
        logging.debug(f"torchcodec failed or not available: {e}. Using torchaudio.")
        try:
            torchaudio.save(str(path), tensor, sample_rate=sample_rate)
        
            if rank is not None:
                logging.debug(f"[Core {rank}] ✅ Saved with torchaudio: {path}")
            else:
                logging.debug(f"✅ Saved with torchaudio: {path}")
        except Exception as save_e:
            logging.error(f"❌ Failed to save audio to {path}: {save_e}")
            raise


def run_generation_local(flags):
    """
    Local generation using modern chunked pipeline — even for single process.
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
    args.diffusion_iterations = flags.get('diffusion_iterations', 16)
    args.num_autoregressive_samples = flags.get('num_autoregressive_samples', 1)
    args.sample_rate = flags.get('sample_rate', 22050)
    args.batch_size = flags.get('batch_size', 4)
    args.force_regenerate = flags.get('force_regenerate', False)

    try:
        with open(args.lines_file, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
            total_lines = len(all_lines)
    except Exception as e:
        logging.error(f"Failed to read lines file: {e}")
        return

    args.start_idx = flags.get('start_idx', 0)
    args.end_idx = flags.get('end_idx', total_lines)
    args.total_lines = total_lines
    args.rank = 0

    logging.info(f"▶️ Local generation starting. Processing lines [{args.start_idx} - {args.end_idx})")
    run_generation_chunked(args)
    logging.info(f"✅ Local generation finished.")


def run_generation_chunked(args):
    """
    Loads model once, then generates all lines in [args.start_idx, args.end_idx) in micro-batches.
    """
    import torch
    import os
    import logging
    from tortoise.api import TextToSpeech, generate_batched_lines

    # Conditional import is the CRITICAL FIX here.
    if args.hardware == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            rank = xm.get_ordinal()
            logging.info(f"[Core {rank}] Using TPU device: {device}")
        except ImportError as e:
            logging.error(f"Failed to import torch_xla for TPU: {e}. Falling back to GPU/CPU.")
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            rank = args.rank
            logging.info(f"[Core {rank}] Using fallback device: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        rank = args.rank
        logging.info(f"[Core {rank}] Using GPU device: {device}")
    else:
        device = torch.device('cpu')
        rank = args.rank
        logging.info(f"[Core {rank}] Using CPU device")

    logging.info(f"[Core {rank}] Loading TTS model...")
    tts_model = TextToSpeech(
        models_dir=args.models_dir,
        device=str(device)
    )
    tts_model.device_rank = rank
    logging.info(f"[Core {rank}] Model loaded successfully.")

    with open(args.lines_file, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    sample_rate = getattr(args, 'sample_rate', 22050)
    silent_fallback = torch.zeros(1, sample_rate)

    chunk_lines = all_lines[args.start_idx:args.end_idx]
    chunk_indices = list(range(args.start_idx, args.end_idx))

    logging.info(f"[Core {rank}] Generating {len(chunk_lines)} lines in batches of {args.batch_size}...")

    audios = generate_batched_lines(
        tts_model=tts_model,
        lines=chunk_lines,
        diffusion_iterations=args.diffusion_iterations,
        num_autoregressive_samples=args.num_autoregressive_samples,
        voice=args.voice,
        preset=args.preset,
        max_chunk_size=args.batch_size,
        device=str(device),
        sample_rate=sample_rate
    )

    for audio, line_idx in zip(audios, chunk_indices):
        output_path = os.path.join(args.output_dir, f"line_{line_idx+1:04d}.wav")

        if os.path.exists(output_path) and not args.force_regenerate:
            logging.debug(f"[Core {rank}] Skipping line {line_idx+1} (already exists)")
            continue

        try:
            if not isinstance(audio, torch.Tensor):
                logging.warning(f"[Core {rank}] Invalid audio for line {line_idx+1}. Using silent fallback.")
                audio = silent_fallback

            save_audio_fallback(output_path, audio, sample_rate, rank=rank)
            logging.info(f"[Core {rank}] ✅ Saved: {output_path}")

        except Exception as e:
            logging.error(f"[Core {rank}] ❌ Failed to save line {line_idx+1}: {e}")
            save_audio_fallback(output_path, silent_fallback, sample_rate, rank=rank)

    logging.info(f"[Core {rank}] ✅ Finished chunk [{args.start_idx} - {args.end_idx}).")


def run_generation_for_spawn(index, process_args_list):
    """
    Each TPU core processes a chunk of lines to maximize memory and compute utilization.
    """
    flags = process_args_list[index]

    class Args:
        pass
    args = Args()

    args.lines_file = flags['lines_file']
    args.output_dir = flags['output_dir']
    args.hardware = flags['hardware']
    args.voice = flags['voice']
    args.preset = flags['preset']
    args.models_dir = flags['models_dir']
    args.diffusion_iterations = flags.get('diffusion_iterations', 16)
    args.num_autoregressive_samples = flags.get('num_autoregressive_samples', 1)
    args.start_idx = flags['start_idx']
    args.end_idx = flags['end_idx']
    args.total_lines = flags['total_lines']
    args.rank = flags['rank']
    args.force_regenerate = flags.get('force_regenerate', False)
    args.sample_rate = flags.get('sample_rate', 22050)
    args.batch_size = flags.get('batch_size', 4)

    logging.info(f"▶️ TPU Core {args.rank} starting. Processing lines [{args.start_idx} - {args.end_idx})")

    run_generation_chunked(args)

    logging.info(f"✅ TPU Core {args.rank} finished processing {args.end_idx - args.start_idx} lines.")


if __name__ == "__main__":
    logging.error("This is a library file and should not be run directly. Please use a driver script.")
    sys.exit(1)

