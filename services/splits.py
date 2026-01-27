import torch
import torchaudio
import os
import tempfile
import urllib.parse
import requests
import hashlib
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager
import json

# Default cleanup delay in seconds (5 minutes)
CLEANUP_DELAY_SECONDS = 5 * 60


def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    return path.startswith(('http://', 'https://'))


def get_url_hash(url: str) -> str:
    """Generate a short hash from URL for folder naming."""
    # Use only the path part (without query params) for consistent hashing
    parsed = urllib.parse.urlparse(url)
    path_to_hash = parsed.path
    return hashlib.md5(path_to_hash.encode()).hexdigest()[:12]


def schedule_folder_cleanup(folder_path: str, delay_seconds: int = CLEANUP_DELAY_SECONDS):
    """Schedule a folder to be deleted after a delay."""
    def delete_folder():
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Auto-cleanup: Deleted folder '{folder_path}'")
        except Exception as e:
            print(f"Auto-cleanup failed for '{folder_path}': {e}")

    timer = threading.Timer(delay_seconds, delete_folder)
    timer.daemon = True  # Don't block program exit
    timer.start()
    print(f"Scheduled cleanup for '{folder_path}' in {delay_seconds // 60} minutes")


@contextmanager
def get_local_audio_path(audio_source: str):
    """
    Context manager that yields a local file path.
    If audio_source is a URL, downloads to a temp file and cleans up after.
    If audio_source is a local path, yields it directly.
    """
    if is_url(audio_source):
        # Download URL to temporary file
        suffix = Path(urllib.parse.urlparse(audio_source).path).suffix or '.wav'
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            print(f"Downloading audio from URL...")
            # Use requests library for better S3/presigned URL compatibility
            response = requests.get(audio_source, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            print(f"Download complete.")
            yield temp_file.name
        finally:
            # Clean up temp file
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    else:
        # Local path, yield directly
        yield audio_source

def split_audio_by_silence(
    audio_source: str,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30
) -> List[Dict[str, float]]:
    """
    Split audio file based on silence using Silero VAD.

    Args:
        audio_source: Local file path or URL to audio file.
    """

    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    with get_local_audio_path(audio_source) as local_path:
        # Read audio file
        wav = read_audio(local_path, sampling_rate=sample_rate)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False
        )

    # Convert to seconds and format output
    segments = []
    for timestamp in speech_timestamps:
        duration = round(timestamp['end'] / sample_rate, 3) - round(timestamp['start'] / sample_rate, 3)
        segment = {
            'start': round(timestamp['start'] / sample_rate, 3),
            'end': round(timestamp['end'] / sample_rate, 3),
            'duration':duration
        }
        segments.append(segment)

    return segments


def cut_and_save_audio_chunks(
    audio_source: str,
    segments: List[Dict[str, float]],
    output_folder: str = "audio_chunks",
    output_format: str = "wav",
    preserve_sample_rate: bool = True,
    return_absolute_paths: bool = False,
    base_name: Optional[str] = None
) -> List[Dict[str, any]]:
    """
    Cut audio into chunks based on segments and save to folder.

    Args:
        audio_source: Local file path or URL to audio file.
        base_name: Optional base name for output files. If not provided,
                   derived from the audio_source (filename or URL path).
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine base name for output files
    if base_name is None:
        if is_url(audio_source):
            # Extract filename from URL path
            url_path = urllib.parse.urlparse(audio_source).path
            base_name = Path(url_path).stem or "audio"
        else:
            base_name = Path(audio_source).stem

    with get_local_audio_path(audio_source) as local_path:
        # Load original audio
        waveform, sample_rate = torchaudio.load(local_path)
    
    result_segments = []
    
    print(f"Cutting audio into {len(segments)} chunks...")
    
    for i, segment in enumerate(segments, 1):
        # Convert time to samples
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        
        # Extract chunk
        chunk = waveform[:, start_sample:end_sample]
        
        # Generate output filename
        duration = segment['end'] - segment['start']
        output_filename = f"{base_name}_chunk_{i:03d}_{segment['start']:.2f}s-{segment['end']:.2f}s.{output_format}"
        output_file = output_path / output_filename
        
        # Save chunk
        torchaudio.save(
            str(output_file),
            chunk,
            sample_rate,
            format=output_format
        )
        
        # Get file path (absolute or relative)
        if return_absolute_paths:
            file_url = str(output_file.absolute())
        else:
            file_url = str(output_file)
        
        # Add to result with url
        result_segment = {
            'start': segment['start'],
            'end': segment['end'],
            'url': file_url,
            'duration': duration
        }
        result_segments.append(result_segment)
        
        print(f"Saved chunk {i}/{len(segments)}: {output_filename} (duration: {duration:.2f}s)")
    
    print(f"\n✓ All {len(segments)} chunks saved to '{output_folder}/'")
    return result_segments


def process_audio_file(
    audio_source: str,
    output_folder: str = "audio_chunks",
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    output_format: str = "wav",
    return_absolute_paths: bool = False
) -> List[Dict[str, any]]:
    """
    Complete pipeline: detect speech segments and save chunks.

    Args:
        audio_source: Local file path or URL to audio file.
    """

    # Determine base name and output subfolder
    if is_url(audio_source):
        url_path = urllib.parse.urlparse(audio_source).path
        base_name = Path(url_path).stem or "audio"
        # Create subfolder based on URL hash
        url_hash = get_url_hash(audio_source)
        output_folder = str(Path(output_folder) / url_hash)
    else:
        base_name = Path(audio_source).stem

    print(f"Processing: {audio_source}")
    print(f"Output folder: {output_folder}")
    print("-" * 60)

    # Use context manager to download URL once (if needed)
    with get_local_audio_path(audio_source) as local_path:
        # Step 1: Detect speech segments
        print("Step 1: Detecting speech segments with Silero VAD...")
        segments = split_audio_by_silence(
            local_path,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )

        print(f"Found {len(segments)} speech segments\n")

        # Step 2: Cut and save chunks
        print("Step 2: Cutting and saving audio chunks...")
        result_segments = cut_and_save_audio_chunks(
            local_path,
            segments,
            output_folder=output_folder,
            output_format=output_format,
            return_absolute_paths=return_absolute_paths,
            base_name=base_name
        )

    # Schedule automatic cleanup for URL-based sources
    if is_url(audio_source):
        schedule_folder_cleanup(output_folder)

    return result_segments


def batch_process_folder(
    input_folder: str,
    output_base_folder: str = "processed_audio",
    file_extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a'],
    return_absolute_paths: bool = False,
    **vad_params
) -> Dict[str, List[Dict[str, any]]]:
    """
    Process all audio files in a folder.
    """
    
    input_path = Path(input_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"❌ Error: Input folder '{input_folder}' does not exist!")
        print(f"Please create the folder and add audio files, or specify a different folder.")
        return {}
    
    # Find all audio files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"❌ No audio files found in '{input_folder}'")
        print(f"Looking for files with extensions: {file_extensions}")
        print(f"\nPlease add audio files to this folder or check the file extensions.")
        return {}
    
    print(f"Found {len(audio_files)} audio files to process\n")
    
    results = {}
    
    for audio_file in audio_files:
        # Create output folder for this file
        output_folder = Path(output_base_folder) / audio_file.stem
        
        try:
            segments = process_audio_file(
                str(audio_file),
                output_folder=str(output_folder),
                return_absolute_paths=return_absolute_paths,
                **vad_params
            )
            results[str(audio_file)] = segments
            print(f"\n{'='*60}\n")
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}\n")
            continue
    
    return results


# # Example usage
# if __name__ == "__main__":
    
#     # Example 1: Process single file
#     print("Example 1: Single file processing")
#     print("="*60)
    
#     # Check if example file exists
#     audio_file = "/Users/jeansauvenelbeaudry/Documents/Local-Project/datamio-py-api/input_audio/1db53de0-4161-4abe-89e3-77e609ced9db.wav"
    
#     if not Path(audio_file).exists():
#         print(f"❌ Audio file '{audio_file}' not found!")
#         print(f"Please provide a valid audio file path.\n")
        
#         # Try to find any audio file in current directory
#         current_dir = Path(".")
#         audio_files = list(current_dir.glob("*.wav")) + list(current_dir.glob("*.mp3"))
        
#         if audio_files:
#             print(f"Found these audio files in current directory:")
#             for f in audio_files:
#                 print(f"  - {f}")
#             print(f"\nUsing: {audio_files[0]}")
#             audio_file = str(audio_files[0])
#         else:
#             print("No audio files found in current directory.")
#             print("Skipping Example 1...\n")
#             audio_file = None
    
#     # if audio_file and Path(audio_file).exists():
#     #     segments = process_audio_file(
#     #         audio_file,
#     #         output_folder="audio_chunks",
#     #         threshold=0.5,
#     #         min_speech_duration_ms=250,
#     #         min_silence_duration_ms=100,
#     #         speech_pad_ms=30,
#     #         output_format="wav",
#     #         return_absolute_paths=False
#     #     )
        
#     #     # Print segments array with URLs
#     #     print("\nSegments array with URLs:")
#     #     print(json.dumps(segments, indent=2))
        
#     #     # Save to JSON
#     #     output_json = "audio_chunks/segments.json"
#     #     Path("audio_chunks").mkdir(exist_ok=True)
#     #     with open(output_json, 'w') as f:
#     #         json.dump(segments, f, indent=2)
#     #     print(f"\n✓ Segments saved to: {output_json}")
    
    
#     # Example 2: Batch processing
#     print("\n\nExample 2: Batch processing")
#     print("="*60)
    
#     input_folder = "input_audio"
    
#     # Create input folder if it doesn't exist
#     Path(input_folder).mkdir(exist_ok=True)
    
#     results = batch_process_folder(
#         input_folder=input_folder,
#         output_base_folder="processed_audio",
#         threshold=0.3,
#         return_absolute_paths=False
#     )
    
#     if results:
#         print("\nBatch processing results:")
#         for input_file, segments in results.items():
#             print(f"\n{input_file}: {len(segments)} segments")
#             for seg in segments[:2]:  # Show first 2
#                 print(f"  - {seg}")
        
#         # Create output folder and save batch results to JSON
#         output_base = Path("processed_audio")
#         output_base.mkdir(exist_ok=True)
        
#         batch_results_file = output_base / "batch_results.json"
#         with open(batch_results_file, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         print(f"\n✓ Batch results saved to: {batch_results_file}")
#     else:
#         print("\nNo files were processed.")
#         print(f"To use batch processing:")
#         print(f"  1. Create folder: mkdir {input_folder}")
#         print(f"  2. Add audio files (.wav, .mp3, .flac, .m4a)")
#         print(f"  3. Run the script again")
    
    
#     # Example 3: Simple usage
#     print("\n\nExample 3: Simple usage example")
#     print("="*60)
    
#     print("""
# # Simple usage in your code:

# from audio_splitter import process_audio_file

# # Process a single file
# segments = process_audio_file(
#     "my_audio.wav",
#     output_folder="chunks",
#     threshold=0.5
# )

# # segments will be:
# # [
# #     {"start": 0.0, "end": 2.5, "url": "chunks/my_audio_chunk_001_0.00s-2.50s.wav"},
# #     {"start": 3.1, "end": 5.8, "url": "chunks/my_audio_chunk_002_3.10s-5.80s.wav"},
# #     ...
# # ]

# # Access the URLs
# for seg in segments:
#     print(f"Play: {seg['url']}")
#     print(f"Duration: {seg['end'] - seg['start']:.2f}s")
#     """)