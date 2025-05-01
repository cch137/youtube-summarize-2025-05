import subprocess
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os

# YouTube video URL
youtube_url = "https://youtu.be/_3JJnfAoOt4"

# Define output audio file path
output_audio = "downloaded_audio.mp3"


# Function to download audio using yt-dlp
def download_audio(url, output_path):
    command = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format",
        "mp3",  # Convert to MP3 format
        "-o",
        output_path,  # Output file path
        url,
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Audio downloaded successfully to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download audio: {e}")
        return False


# Load Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Set model to evaluation mode
model.eval()
print("Whisper model loaded and set to evaluation mode")

# Download audio
if download_audio(youtube_url, output_audio):
    # Load the downloaded audio file
    print("Loading audio file...")
    audio, sampling_rate = librosa.load(
        output_audio, sr=16000
    )  # Whisper requires 16kHz sampling rate
    print("Audio file loaded successfully")

    # Process audio input
    print("Processing audio for transcription...")
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(input_features)

    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Transcription completed!")
    print("Transcription result:", transcription)

    # Clean up downloaded audio file (optional)
    os.remove(output_audio)
    print(f"Temporary audio file deleted: {output_audio}")
else:
    print("Transcription cannot proceed due to audio download failure.")
