import torch
import torchaudio
import os
import tempfile
import urllib.parse
import requests
import hashlib
import shutil
import threading
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import json
import uuid
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

# Default cleanup delay in seconds (60 minutes)
CLEANUP_DELAY_SECONDS = 10 * 60

# In-memory job storage (use Redis or database in production)
splits_jobs_db: Dict[str, Dict[str, Any]] = {}


class SplitJobStatus(str, Enum):
    DOWNLOADING = "downloading"
    SPLITTING = "splitting"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


class SplitAudioRequest(BaseModel):
    audio_source: str
    output_folder: str = "audio_chunks"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    output_format: str = "wav"
    return_absolute_paths: bool = False
    clear_folder: bool = False
    webhookUrl: Optional[str] = None
    transcription: bool = False
    open_ai_key: Optional[str] = None


class SplitJobResponse(BaseModel):
    job_id: str
    status: SplitJobStatus
    message: str
    created_at: str
    updated_at: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Webhook events to send notifications for
SPLIT_WEBHOOK_EVENTS = {
    SplitJobStatus.DOWNLOADING: "DOWNLOADING",
    SplitJobStatus.SPLITTING: "SPLITTING",
    SplitJobStatus.TRANSCRIBING: "TRANSCRIBING",
    SplitJobStatus.COMPLETED: "COMPLETED",
    SplitJobStatus.FAILED: "FAILED",
}


async def send_split_webhook(webhook_url: str, job_data: Dict[str, Any], event: str):
    """Send webhook notification for split job status change"""
    if not webhook_url:
        return

    payload = {
        "event": event,
        "job_id": job_data["job_id"],
        "status": job_data["status"].value if isinstance(job_data["status"], SplitJobStatus) else job_data["status"],
        "message": job_data["message"],
        "created_at": job_data["created_at"],
        "updated_at": job_data["updated_at"],
        "progress": job_data["progress"],
        "result": job_data.get("result"),
        "error": job_data.get("error"),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status >= 400:
                    print(f"Webhook failed with status {response.status}: {await response.text()}")
    except Exception as e:
        print(f"Failed to send webhook: {e}")


def create_split_job(user_id: str = "default", webhook_url: Optional[str] = None) -> str:
    """Create a new split job and return job ID"""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    splits_jobs_db[job_id] = {
        "job_id": job_id,
        "user_id": user_id,
        "status": SplitJobStatus.DOWNLOADING,
        "message": "Downloading audio from URL...",
        "created_at": now,
        "updated_at": now,
        "progress": {
            "segments_found": 0,
            "chunks_saved": 0,
            "total_chunks": 0,
        },
        "result": None,
        "error": None,
        "webhook_url": webhook_url,
    }

    # Schedule webhook for CREATED event
    if webhook_url:
        asyncio.create_task(send_split_webhook(webhook_url, splits_jobs_db[job_id], "CREATED"))

    return job_id


def update_split_job(
    job_id: str,
    status: Optional[SplitJobStatus] = None,
    message: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    """Update split job information"""
    if job_id not in splits_jobs_db:
        return

    job = splits_jobs_db[job_id]
    old_status = job["status"]

    if status:
        job["status"] = status
    if message:
        job["message"] = message
    if progress:
        job["progress"].update(progress)
    if result:
        job["result"] = result
    if error:
        job["error"] = error

    job["updated_at"] = datetime.utcnow().isoformat()

    # Send webhook if status changed to a tracked event
    if status and status != old_status and status in SPLIT_WEBHOOK_EVENTS:
        webhook_url = job.get("webhook_url")
        if webhook_url:
            asyncio.create_task(send_split_webhook(webhook_url, job, SPLIT_WEBHOOK_EVENTS[status]))


def get_split_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get split job information by ID"""
    return splits_jobs_db.get(job_id)


def get_user_split_jobs(user_id: str = "default") -> List[Dict[str, Any]]:
    """Get all split jobs for a user"""
    return [
        job for job in splits_jobs_db.values()
        if job["user_id"] == user_id
    ]


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


async def transcribe_audio_chunk(file_path: str, open_ai_key: str) -> str:
    """
    Transcribe a single audio chunk using OpenAI Whisper API.

    Args:
        file_path: Path to the audio file to transcribe
        open_ai_key: OpenAI API key

    Returns:
        Transcribed text
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {open_ai_key}"
    }

    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as audio_file:
            form_data = aiohttp.FormData()
            form_data.add_field('file', audio_file, filename=os.path.basename(file_path))
            form_data.add_field('model', 'gpt-4o-transcribe')
            form_data.add_field('prompt', 'In haitian creole')
            

            async with session.post(url, headers=headers, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")

                result = await response.json()
                return result.get('text', '')


async def transcribe_audio_chunks(
    segments: List[Dict[str, Any]],
    open_ai_key: str,
    job_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Transcribe all audio chunks and add transcription to segments.

    Args:
        segments: List of segment dictionaries with 'url' key pointing to audio files
        open_ai_key: OpenAI API key
        job_id: Optional job ID for progress updates

    Returns:
        Updated segments with 'transcription' field added
    """
    total = len(segments)

    for i, segment in enumerate(segments, 1):
        file_path = segment.get('url', '')
        if not file_path or not os.path.exists(file_path):
            segment['transcription'] = ''
            continue

        try:
            transcription = await transcribe_audio_chunk(file_path, open_ai_key)
            segment['transcription'] = transcription
            print(f"Transcribed chunk {i}/{total}: {transcription[:50]}..." if len(transcription) > 50 else f"Transcribed chunk {i}/{total}: {transcription}")

            if job_id:
                update_split_job(
                    job_id,
                    message=f"Transcribing audio chunks ({i}/{total})",
                    progress={"chunks_transcribed": i}
                )
        except Exception as e:
            print(f"Failed to transcribe chunk {i}: {e}")
            segment['transcription'] = ''

    return segments


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
    base_name: Optional[str] = None,
    clear_folder: bool = False
) -> List[Dict[str, any]]:
    """
    Cut audio into chunks based on segments and save to folder.

    Args:
        audio_source: Local file path or URL to audio file.
        base_name: Optional base name for output files. If not provided,
                   derived from the audio_source (filename or URL path).
        clear_folder: If True, clears the output folder before adding new files.
    """

    # Create output folder (clear first if requested)
    output_path = Path(output_folder)
    if clear_folder and output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleared existing folder: {output_folder}")
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
    return_absolute_paths: bool = False,
    clear_folder: bool = False
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
            base_name=base_name,
            clear_folder=clear_folder
        )

    # Compute silence gaps between speech segments
    detected_silences = []
    for i in range(len(segments) - 1):
        silence_start = segments[i]['end']
        silence_end = segments[i + 1]['start']
        if silence_end > silence_start:
            detected_silences.append({
                "start": silence_start,
                "end": silence_end,
                "duration": round(silence_end - silence_start, 3),
            })

    # Schedule automatic cleanup for URL-based sources
    if is_url(audio_source):
        schedule_folder_cleanup(output_folder)

    return {
        "segments": result_segments,
        "splitPoints": [{"start": s["start"], "end": s["end"], "duration": s["duration"]} for s in segments],
        "detectedSilences": detected_silences,
    }


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
            file_result = process_audio_file(
                str(audio_file),
                output_folder=str(output_folder),
                return_absolute_paths=return_absolute_paths,
                **vad_params
            )
            results[str(audio_file)] = file_result["segments"]
            print(f"\n{'='*60}\n")
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}\n")
            continue

    # Schedule automatic cleanup for the output folder
    schedule_folder_cleanup(output_base_folder)

    return results


async def process_split_job(
    job_id: str,
    audio_source: str,
    output_folder: str = "audio_chunks",
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    output_format: str = "wav",
    return_absolute_paths: bool = False,
    clear_folder: bool = False,
    transcription: bool = False,
    open_ai_key: Optional[str] = None,
):
    """Background task to process audio splitting"""
    try:
        # Determine base name and output subfolder
        if is_url(audio_source):
            url_path = urllib.parse.urlparse(audio_source).path
            base_name = Path(url_path).stem or "audio"
            url_hash = get_url_hash(audio_source)
            output_folder = str(Path(output_folder) / url_hash)
            
        else:
            base_name = Path(audio_source).stem

        with get_local_audio_path(audio_source) as local_path:
            # Splitting
            update_split_job(
                job_id,
                status=SplitJobStatus.SPLITTING,
                message="Detecting speech segments with Silero VAD",
            )

            segments = split_audio_by_silence(
                local_path,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms
            )

            update_split_job(
                job_id,
                message=f"Found {len(segments)} speech segments",
                progress={
                    "segments_found": len(segments),
                    "total_chunks": len(segments),
                }
            )

            result_segments = cut_and_save_audio_chunks(
                local_path,
                segments,
                output_folder=output_folder,
                output_format=output_format,
                return_absolute_paths=return_absolute_paths,
                base_name=base_name,
                clear_folder=clear_folder
            )

            update_split_job(
                job_id,
                progress={"chunks_saved": len(result_segments)}
            )

        # Transcribe audio chunks if requested and API key is provided
        if transcription and open_ai_key:
            update_split_job(
                job_id,
                status=SplitJobStatus.TRANSCRIBING,
                message="Transcribing audio chunks with OpenAI Whisper",
                progress={"chunks_transcribed": 0}
            )

            result_segments = await transcribe_audio_chunks(
                result_segments,
                open_ai_key,
                job_id=job_id
            )

        # Schedule automatic cleanup for URL-based sources
        if is_url(audio_source):
            schedule_folder_cleanup(output_folder)

        # Compute silence gaps between speech segments
        detected_silences = []
        for i in range(len(segments) - 1):
            silence_start = segments[i]['end']
            silence_end = segments[i + 1]['start']
            if silence_end > silence_start:
                detected_silences.append({
                    "start": silence_start,
                    "end": silence_end,
                    "duration": round(silence_end - silence_start, 3),
                })

        # Job completed successfully
        update_split_job(
            job_id,
            status=SplitJobStatus.COMPLETED,
            message="Audio splitting completed successfully" if not (transcription and open_ai_key) else "Audio splitting and transcription completed successfully",
            result={
                "success": True,
                "output_folder": output_folder,
                "total_segments": len(result_segments),
                "splitPoints": [{"start": s["start"], "end": s["end"], "duration": s["duration"]} for s in segments],
                "segments": result_segments,
                "detectedSilences": detected_silences,
                "settings": {
                    "threshold": threshold,
                    "min_speech_duration_ms": min_speech_duration_ms,
                    "min_silence_duration_ms": min_silence_duration_ms,
                    "speech_pad_ms": speech_pad_ms,
                    "output_format": output_format,
                },
            }
        )

    except Exception as error:
        import traceback
        update_split_job(
            job_id,
            status=SplitJobStatus.FAILED,
            message=f"Processing failed: {str(error)}",
            error=traceback.format_exc(),
        )
