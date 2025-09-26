import os
from pydub import AudioSegment

# Configuration: Define the folder and file
# ----------------------------------------------------------------------
# NOTE: The name defined here must match the stem of your WAV file (e.g., 'liria.wav')
folder_name = "senor_joao"
input_file = f"{folder_name}.wav" 
# ----------------------------------------------------------------------

# 1. Create the output folder if it doesn't exist
try:
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder '{folder_name}' created or already exists.")
except OSError as e:
    print(f"Error creating folder: {e}")
    # Exit if we can't create the necessary directory
    exit()

# 2. Load the audio file
try:
    # Requires pydub and a system dependency like FFmpeg or Libav to load audio
    print(f"Loading audio from '{input_file}'...")
    audio = AudioSegment.from_wav(input_file)
    print(f"Audio loaded successfully (Duration: {len(audio) / 1000:.2f} seconds).")
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Make sure it's in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the audio: {e}")
    exit()

# 3. Split the audio into 10-second segments
ten_seconds = 10 * 1000  # pydub works in milliseconds
segment_count = 0

print("Starting segmentation...")

# Iterate through the audio, stepping every 10,000 milliseconds
for start_time in range(0, len(audio), ten_seconds):
    end_time = start_time + ten_seconds
    segment = audio[start_time:end_time]

    # Save the segment to a file
    segment_count += 1
    # Segments are named 1.wav, 2.wav, 3.wav, etc., inside the output folder
    output_filename = os.path.join(folder_name, f"{segment_count}.wav")
    
    # Export the segment in WAV format
    segment.export(output_filename, format="wav")
    print(f"Exported segment {segment_count}: '{output_filename}'")

print("\nAll segments have been successfully created!")
print(f"Total segments exported: {segment_count}")

