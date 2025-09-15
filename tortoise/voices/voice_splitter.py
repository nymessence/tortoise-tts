import os
from pydub import AudioSegment

# 📂 Define the folder and file
folder_name = "_" # edit this to match wav filename
input_file = f"{folder_name}.wav"  # Assumes the file is named "{folder_name}.wav"
# note that this is used as the voice name

# 💻 Create the output folder
try:
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder '{folder_name}' created or already exists.")
except OSError as e:
    print(f"Error creating folder: {e}")
    exit()

# 🎶 Load the audio file
try:
    audio = AudioSegment.from_wav(input_file)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Make sure it's in the same directory as the script.")
    exit()

# ✂️ Split the audio into 10-second segments
ten_seconds = 10 * 1000  # pydub works in milliseconds
segment_count = 0

for start_time in range(0, len(audio), ten_seconds):
    end_time = start_time + ten_seconds
    segment = audio[start_time:end_time]

    # Save the segment to a file
    segment_count += 1
    output_filename = os.path.join(folder_name, f"{segment_count}.wav")
    segment.export(output_filename, format="wav")
    print(f"Exported '{output_filename}'")

print("\nAll segments have been successfully created!")

