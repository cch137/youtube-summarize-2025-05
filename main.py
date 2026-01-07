import os
import subprocess
import torch
from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.pipelines import pipeline


def download_audio(url, output_dir="downloads"):
    """
    Download audio from video URL using yt-dlp
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use yt-dlp to download audio
    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "mp3",  # or: wav
        "--audio-quality",
        "0",
        "--output",
        f"{output_dir}/%(title)s.%(ext)s",
        url,
    ]

    try:
        print(f"Downloading audio: {url}")
        subprocess.run(command, check=True)

        # Find the downloaded file in the output directory
        files = os.listdir(output_dir)
        if not files:
            print(f"No files found in {output_dir}")
            return None

        # Get the first audio file in the directory (should be our downloaded file)
        audio_file = os.path.join(output_dir, files[0])
        print(f"Found audio file: {audio_file}")

        # Check if file exists
        if os.path.exists(audio_file):
            return audio_file
        else:
            print(f"File doesn't exist: {audio_file}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return None


def transcribe_audio(audio_file, output_dir="transcripts"):
    """
    Transcribe audio file using Whisper model
    """
    # Verify file exists before proceeding
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")

    # Load Whisper model
    # model_id = "openai/whisper-large-v3"
    model_id = "openai/whisper-small"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create transcription pipeline with fixed warning
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"max_new_tokens": 128, "task": "transcribe"},  # Fixed warning
    )

    # Transcribe audio (with automatic language detection)
    print(f"Transcribing audio: {audio_file}")
    result = pipe(audio_file)

    # Save transcription results
    base_name = os.path.basename(audio_file)
    file_name = os.path.splitext(base_name)[0]
    output_file = f"{output_dir}/{file_name}_transcript.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        # Process different result formats
        if isinstance(result, dict):
            if "chunks" in result:
                # Process chunks with timestamps
                for chunk in result["chunks"]:
                    if (
                        isinstance(chunk, dict)
                        and "timestamp" in chunk
                        and "text" in chunk
                    ):
                        start = chunk["timestamp"][0]
                        end = chunk["timestamp"][1]
                        text = chunk["text"]
                        f.write(
                            f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}\n"
                        )
            elif "text" in result:
                # Extract text field from dictionary
                f.write(result["text"])
            else:
                # Convert dict to string if no recognized format
                f.write(str(result))
        elif isinstance(result, str):
            # Direct string output
            f.write(result)
        elif isinstance(result, list):
            # Handle list results (could be list of transcriptions or segments)
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    f.write(item["text"] + "\n")
                else:
                    f.write(str(item) + "\n")
        else:
            # Fallback for any other type
            f.write(str(result))

    print(f"Transcription completed, saved to: {output_file}")
    return output_file


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    # YouTube video URL to download
    url = "https://youtu.be/_3JJnfAoOt4"

    # Download audio
    audio_file = download_audio(url)
    if audio_file:
        print(f"Successfully downloaded audio to: {audio_file}")
        # Transcribe audio
        transcript_file = transcribe_audio(audio_file)
        if transcript_file:
            print(f"Process completed. Transcript saved as: {transcript_file}")
        else:
            print("Transcription failed.")
    else:
        print("Cannot continue because audio download failed.")


if __name__ == "__main__":
    main()
