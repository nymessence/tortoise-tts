import os
import argparse
import sys
import time
import logging
import torch
import torchaudio

# For TPU support, will be imported only when hardware is 'tpu'
xm = None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s')

def run_generation(args):
    """Main generation logic for a single process."""
    global xm
    try:
        # Load the Tortoise TTS library dynamically
        try:
            from tortoise.api import TextToSpeech
            from tortoise.utils.audio import load_audio
        except ImportError:
            logging.error("Could not import Tortoise TTS. Ensure the repository is in your PYTHONPATH.")
            sys.exit(1)

        # 🚨 Real device initialization happens here
        if args.hardware == 'tpu':
            import torch_xla.core.xla_model as xla_model
            xm = xla_model
            device = xm.xla_device()  # SAFE: Called inside worker process
            logging.info(f"TPU device initialized in worker: {device}")
        elif args.hardware == 'gpu' and torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info("Using GPU for generation.")
        else:
            device = torch.device('cpu')
            logging.info("Using CPU for generation.")

        # Initialize the TTS model with the correct models_dir
        tts = TextToSpeech(models_dir=args.models_dir)
        
        # Ensure the model is moved to the device correctly if it has a .to() method.
        # This warning is for the main TTS model, not the audio samples.
        if hasattr(tts, 'to'):
            tts.to(device)
            logging.info("Main TTS model moved to device successfully.")
        else:
            logging.warning("TextToSpeech model does not have a .to() method. This may cause issues on non-CPU devices.")

        # --- CRITICAL FIX: Manually load voice samples, ensuring correct sample rate, and get conditioning latents ---
        try:
            voice_dir = os.path.join(args.models_dir, 'tortoise/voices', args.voice)
            if not os.path.isdir(voice_dir):
                raise FileNotFoundError(f"Voice directory not found: {voice_dir}")

            # Find all .wav files in the voice directory
            voice_files = [os.path.join(voice_dir, f) for f in os.listdir(voice_dir) if f.endswith('.wav')]
            if not voice_files:
                raise ValueError(f"No .wav files found in voice directory: {voice_dir}")

            logging.info(f"Found {len(voice_files)} reference audio files for voice '{args.voice}'.")

            TARGET_SAMPLE_RATE = 22050
            loaded_voice_samples = []
            for f in voice_files:
                audio_tensor = load_audio(f, TARGET_SAMPLE_RATE).to(device)
                loaded_voice_samples.append(audio_tensor)
                logging.debug(f"Loaded '{os.path.basename(f)}' at {TARGET_SAMPLE_RATE} Hz and moved to {device}.")

            # --- PATCH: This is the fix for the AttributeError ---
            # Remove the non-existent method call
            voice_samples = loaded_voice_samples
            
            # The tts_with_preset function will now handle the latent generation internally
            
            logging.info(f"Successfully loaded voice samples (resampled to {TARGET_SAMPLE_RATE} Hz) for '{args.voice}'.")
        except Exception as e:
            logging.error(f"Failed to load voice samples for '{args.voice}': {e}", exc_info=True)
            logging.error("Please ensure a directory named '{voice_name}' with .wav files exists inside '<models_dir>/tortoise/voices/' and contains valid .wav audio files.")
            raise

        # Load text lines from the script
        try:
            with open(args.lines_file, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Text file not found: {args.lines_file}")
            sys.exit(1)

        if not all_lines:
            logging.error("No lines to process. Exiting.")
            return

        # Determine the subset of lines for this process to handle
        start_index = args.start_idx
        step = args.step
        lines_to_process = all_lines[start_index::step]
        logging.info(f"Found {len(all_lines)} lines in total.")
        logging.info(f"This process will handle {len(lines_to_process)} lines, starting from index {start_index} with a step of {step}.")

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Output directory '{args.output_dir}' is ready.")

        # Main generation loop
        logging.info("Starting generation for this process's lines.")
        for i, line in enumerate(lines_to_process):
            original_index = start_index + i * step
            output_path = os.path.join(args.output_dir, f"{original_index:04d}.wav")
            
            logging.info(f"[{original_index + 1}/{len(all_lines)}] Generating audio for segment: '{line}'...")

            # Generate audio
            gen = tts.tts_with_preset(
                text=line,
                voice_samples=voice_samples,
                preset=args.preset,
                use_deterministic_seed=True,
                cvvp_amount=0.5,
                num_autoregressive_samples=8
            )

            # Fix for potential tuple return
            if isinstance(gen, tuple):
                gen = gen[0]
                
            # Save the generated audio (move to CPU first for saving)
            torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)
            logging.info(f"Audio saved to '{output_path}'.")

        logging.info("All lines for this process have been generated.")

    except Exception as e:
        logging.error(f"Generation failed: {e}", exc_info=True)
        raise  # Re-raise the exception to ensure the process fails correctly
    finally:
        logging.info("Script finished.")


# ========================
# TPU-SAFE ENTRY POINT FOR xmp.spawn
# ========================
def run_generation_for_spawn(flags):
    """
    Sets up arguments from flags provided by xmp.spawn and runs generation.
    This function is the entry point for each of the TPU core processes.
    """
    import torch_xla.core.xla_model as xm

    class Args:
        pass
    args = Args()

    # --- Retrieve arguments from the flags dictionary ---
    args.lines_file = flags['lines_file']
    args.output_dir = flags['output_dir']
    args.hardware = flags['hardware']
    args.voice = flags['voice']
    args.preset = flags['preset']
    args.models_dir = flags['models_dir']

    # --- FIXED: Dynamically calculate the step based on available TPU cores ---
    args.step = xm.xrt_world_size()
    args.start_idx = xm.get_ordinal()

    # Run the main generation logic with the configured arguments
    run_generation(args)

if __name__ == "__main__":
    logging.info("This script is designed to be run via `spawn_wrapper.py` for multi-process support. Exiting.")
    sys.exit(0)
