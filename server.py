"""
FastAPI endpoint for uploading multi-domain audio datasets to Hugging Face
Using async job processing with job tracking
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import asyncio
from services.hg import jobs_db, UploadAudioDatasetRequest, JobResponse, JobStatus, create_job, update_job, get_job, get_user_jobs, process_upload_job
from services.splits import (
    process_audio_file,
    splits_jobs_db,
    SplitJobStatus,
    SplitJobResponse,
    create_split_job,
    update_split_job,
    get_split_job,
    get_user_split_jobs,
    transcribe_audio_chunks,
)
from services.auth import (
    require_api_key,
    generate_api_key,
    revoke_api_key,
    delete_api_key,
    list_api_keys,
    get_api_key_by_id,
)
app = FastAPI()

# Mount static files for serving audio chunks
Path("audio_chunks").mkdir(exist_ok=True)
Path("processed_audio").mkdir(exist_ok=True)
app.mount("/audio_chunks", StaticFiles(directory="audio_chunks"), name="audio_chunks")
app.mount("/processed_audio", StaticFiles(directory="processed_audio"), name="processed_audio")

class CreateApiKeyRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class SplitsRequest(BaseModel):
    audio_url: str
    output_folder: Optional[str] = "audio_chunks"
    threshold: Optional[float] = 0.5
    min_speech_duration_ms: Optional[int] = 250
    min_silence_duration_ms: Optional[int] = 100
    speech_pad_ms: Optional[int] = 30
    output_format: Optional[str] = "wav"
    return_absolute_paths: Optional[bool] = False
    webhookUrl: Optional[str] = None
    transcription: Optional[bool] = False
    open_ai_key: Optional[str] = None

class BatchSplitsRequest(BaseModel):
    audio_urls: List[str]
    output_base_folder: Optional[str] = "processed_audio"
    threshold: Optional[float] = 0.5
    min_speech_duration_ms: Optional[int] = 250
    min_silence_duration_ms: Optional[int] = 100
    speech_pad_ms: Optional[int] = 30
    output_format: Optional[str] = "wav"
    return_absolute_paths: Optional[bool] = False
    webhookUrl: Optional[str] = None
    transcription: Optional[bool] = False
    open_ai_key: Optional[str] = None


# ============== SPLITS JOB PROCESSING ==============

async def process_splits_file_job(
    job_id: str,
    audio_url: str,
    output_folder: str,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    output_format: str,
    return_absolute_paths: bool,
    base_url: str,
    transcription: bool = False,
    open_ai_key: str = None,
):
    """Background task to process single file splits job"""
    try:
        # Splitting
        update_split_job(
            job_id,
            status=SplitJobStatus.SPLITTING,
            message="Detecting speech segments with Silero VAD",
        )

        # Always use absolute paths internally (needed for transcription)
        audio_result = await asyncio.to_thread(
            process_audio_file,
            audio_url,
            output_folder=output_folder,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            output_format=output_format,
            return_absolute_paths=True
        )
        segments = audio_result["segments"]
        split_points = audio_result["splitPoints"]
        detected_silences = audio_result["detectedSilences"]

        # Transcribe audio chunks if requested and API key is provided
        if transcription and open_ai_key:
            update_split_job(
                job_id,
                status=SplitJobStatus.TRANSCRIBING,
                message="Transcribing audio chunks with OpenAI Whisper",
                progress={"chunks_transcribed": 0, "total_chunks": len(segments)}
            )

            segments = await transcribe_audio_chunks(
                segments,
                open_ai_key,
                job_id=job_id
            )

        # Convert paths to URLs or keep as absolute paths based on user preference
        for segment in segments:
            file_path = segment['url']
            if return_absolute_paths:
                # Keep the absolute path as-is
                pass
            else:
                # Extract the relative path and convert to URL
                if 'audio_chunks' in file_path:
                    relative_path = file_path[file_path.index('audio_chunks'):]
                else:
                    relative_path = file_path
                segment['url'] = f"{base_url}/{relative_path}"

        # Completed
        update_split_job(
            job_id,
            status=SplitJobStatus.COMPLETED,
            message=f"Successfully processed {len(segments)} segments" + (" with transcription" if transcription and open_ai_key else ""),
            progress={"processed_segments": len(segments)},
            result={
                "splitPoints": split_points,
                "segments": segments,
                "detectedSilences": detected_silences,
                "settings": {
                    "threshold": threshold,
                    "min_speech_duration_ms": min_speech_duration_ms,
                    "min_silence_duration_ms": min_silence_duration_ms,
                    "speech_pad_ms": speech_pad_ms,
                    "output_format": output_format,
                },
                "count": len(segments),
            }
        )

    except Exception as error:
        import traceback
        update_split_job(
            job_id,
            status=SplitJobStatus.FAILED,
            message=f"Splits processing failed: {str(error)}",
            error=traceback.format_exc(),
        )


async def process_splits_batch_job(
    job_id: str,
    audio_urls: List[str],
    output_base_folder: str,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    output_format: str,
    return_absolute_paths: bool,
    base_url: str,
    transcription: bool = False,
    open_ai_key: str = None,
):
    """Background task to process batch splits job"""
    try:
        # Splitting
        update_split_job(
            job_id,
            status=SplitJobStatus.SPLITTING,
            message=f"Processing batch of {len(audio_urls)} files",
            progress={
                "total_files": len(audio_urls),
                "processed_files": 0,
                "total_segments": 0,
            }
        )

        from urllib.parse import urlparse
        results = {}
        total_segments = 0

        for i, audio_url in enumerate(audio_urls, 1):
            update_split_job(
                job_id,
                message=f"Processing file {i}/{len(audio_urls)}: {audio_url}",
                progress={"current_file": i}
            )

            # Extract filename from URL for output folder
            url_path = urlparse(audio_url).path
            file_stem = Path(url_path).stem or f"audio_{i}"
            output_folder = str(Path(output_base_folder) / file_stem)

            try:
                # Always use absolute paths internally (needed for transcription)
                segments = await asyncio.to_thread(
                    process_audio_file,
                    audio_url,
                    output_folder=output_folder,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms,
                    output_format=output_format,
                    return_absolute_paths=True
                )

                # Transcribe audio chunks if requested and API key is provided
                if transcription and open_ai_key:
                    update_split_job(
                        job_id,
                        status=SplitJobStatus.TRANSCRIBING,
                        message=f"Transcribing file {i}/{len(audio_urls)}",
                        progress={"current_file_transcribing": i}
                    )

                    segments = await transcribe_audio_chunks(
                        segments,
                        open_ai_key,
                        job_id=job_id
                    )

                # Convert paths to URLs or keep as absolute paths based on user preference
                for segment in segments:
                    file_path = segment['url']
                    if return_absolute_paths:
                        pass
                    else:
                        if 'processed_audio' in file_path:
                            relative_path = file_path[file_path.index('processed_audio'):]
                        else:
                            relative_path = file_path
                        segment['url'] = f"{base_url}/{relative_path}"

                results[audio_url] = {"segments": segments, "count": len(segments), "status": "success"}
                total_segments += len(segments)
            except Exception as e:
                results[audio_url] = {"segments": [], "count": 0, "status": "failed", "error": str(e)}

            update_split_job(
                job_id,
                progress={
                    "processed_files": i,
                    "total_segments": total_segments,
                }
            )

        # Completed
        update_split_job(
            job_id,
            status=SplitJobStatus.COMPLETED,
            message=f"Batch processing complete: {len(audio_urls)} files, {total_segments} segments" + (" with transcription" if transcription and open_ai_key else ""),
            result={
                "results": results,
                "total_files": len(results),
                "total_segments": total_segments,
            }
        )

    except Exception as error:
        import traceback
        update_split_job(
            job_id,
            status=SplitJobStatus.FAILED,
            message=f"Batch splits processing failed: {str(error)}",
            error=traceback.format_exc(),
        )


@app.post("/api/splits/file/job", response_model=SplitJobResponse)
async def splits_file_job(
    request: SplitsRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    _: Dict = Depends(require_api_key)
):
    """
    Create a job for splitting audio file by silence detection.
    Returns immediately with job ID for tracking progress.
    """
    try:
        job_id = create_split_job(webhook_url=request.webhookUrl)
        base_url = str(http_request.base_url).rstrip('/')

        # Set DOWNLOADING status immediately and return to user
        update_split_job(
            job_id,
            status=SplitJobStatus.DOWNLOADING,
            message=f"Downloading audio from URL...",
            progress={"audio_url": request.audio_url}
        )

        background_tasks.add_task(
            process_splits_file_job,
            job_id=job_id,
            audio_url=request.audio_url,
            output_folder=request.output_folder,
            threshold=request.threshold,
            min_speech_duration_ms=request.min_speech_duration_ms,
            min_silence_duration_ms=request.min_silence_duration_ms,
            speech_pad_ms=request.speech_pad_ms,
            output_format=request.output_format,
            return_absolute_paths=request.return_absolute_paths,
            base_url=base_url,
            transcription=request.transcription,
            open_ai_key=request.open_ai_key,
        )

        # Return job info immediately with DOWNLOADING status
        job_info = get_split_job(job_id)
        return SplitJobResponse(**job_info)

    except HTTPException:
        raise
    except Exception as error:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to create splits job",
                "details": str(error),
                "stack": traceback.format_exc(),
            }
        )


@app.post("/api/splits/batch/job", response_model=SplitJobResponse)
async def splits_batch_job(
    request: BatchSplitsRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    _: Dict = Depends(require_api_key)
):
    """
    Create a job for splitting multiple audio files by silence detection.
    Returns immediately with job ID for tracking progress.
    """
    try:
        job_id = create_split_job(webhook_url=request.webhookUrl)
        base_url = str(http_request.base_url).rstrip('/')

        # Set DOWNLOADING status immediately and return to user
        update_split_job(
            job_id,
            status=SplitJobStatus.DOWNLOADING,
            message=f"Downloading audio files...",
            progress={
                "total_files": len(request.audio_urls),
                "processed_files": 0,
            }
        )

        background_tasks.add_task(
            process_splits_batch_job,
            job_id=job_id,
            audio_urls=request.audio_urls,
            output_base_folder=request.output_base_folder,
            threshold=request.threshold,
            min_speech_duration_ms=request.min_speech_duration_ms,
            min_silence_duration_ms=request.min_silence_duration_ms,
            speech_pad_ms=request.speech_pad_ms,
            output_format=request.output_format,
            return_absolute_paths=request.return_absolute_paths,
            base_url=base_url,
            transcription=request.transcription,
            open_ai_key=request.open_ai_key,
        )

        # Return job info immediately with DOWNLOADING status
        job_info = get_split_job(job_id)
        return SplitJobResponse(**job_info)

    except HTTPException:
        raise
    except Exception as error:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to create batch splits job",
                "details": str(error),
                "stack": traceback.format_exc(),
            }
        )


@app.get("/api/splits/job/{job_id}", response_model=SplitJobResponse)
async def get_split_job_status(job_id: str, _: Dict = Depends(require_api_key)):
    """Get status of a specific split job"""
    job = get_split_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Split job {job_id} not found"
        )

    return SplitJobResponse(**job)


@app.get("/api/splits/jobs", response_model=List[SplitJobResponse])
async def get_all_split_jobs(user_id: str = "default", auth: Dict = Depends(require_api_key)):
    """Get all split jobs for a user. Requires admin API key."""
    if not auth.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only admin can list all jobs")
    jobs = get_user_split_jobs(user_id)
    return [SplitJobResponse(**job) for job in jobs]


@app.delete("/api/splits/job/{job_id}")
async def delete_split_job(job_id: str, _: Dict = Depends(require_api_key)):
    """Delete a split job from the database"""
    if job_id not in splits_jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Split job {job_id} not found"
        )

    del splits_jobs_db[job_id]
    return {"message": f"Split job {job_id} deleted successfully"}


@app.post("/api/upload-audio-dataset", response_model=JobResponse)
async def upload_audio_dataset(
    request: UploadAudioDatasetRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(require_api_key)
):
    """
    Create an upload job for audio dataset
    Returns immediately with job ID
    """
    try:
        user_id = auth.get("id", "default")

        # Check if user already has a pending job
        user_jobs = get_user_jobs(user_id)
        pending_statuses = {JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.DOWNLOADING,
                           JobStatus.SPLITTING, JobStatus.CREATING_DATASET, JobStatus.UPLOADING}
        for job in user_jobs:
            if job["status"] in pending_statuses:
                raise HTTPException(
                    status_code=429,
                    detail=f"You already have a pending job (ID: {job['job_id']}). Please wait for it to complete before starting a new one."
                )

        dataset_items = [item.dict() for item in request.dataset]

        if not dataset_items or not request.datasetName or not request.token:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: dataset, datasetName, token"
            )
        
        # Create job
        job_id = create_job(user_id, webhook_url=request.webhookUrl)
        
        # Update with initial info
        update_job(
            job_id,
            message=f"Job created for dataset: {request.datasetName}",
            progress={
                "total_files": len(dataset_items),
                "downloaded_files": 0,
                "uploaded": False,
            }
        )
        
        # Add background task
        background_tasks.add_task(
            process_upload_job,
            job_id=job_id,
            dataset_items=dataset_items,
            dataset_name=request.datasetName,
            token=request.token,
            is_private=request.isPrivate,
        )
        
        # Return job info immediately
        job_info = get_job(job_id)
        return JobResponse(**job_info)
        
    except HTTPException:
        raise
    except Exception as error:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to create upload job",
                "details": str(error),
                "stack": traceback.format_exc(),
            }
        )


@app.get("/api/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, _: Dict = Depends(require_api_key)):
    """Get status of a specific job"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return JobResponse(**job)


@app.get("/api/jobs", response_model=List[JobResponse])
async def get_all_jobs(user_id: str = "default", auth: Dict = Depends(require_api_key)):
    """Get all jobs for a user. Requires admin API key."""
    if not auth.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only admin can list all jobs")
    jobs = get_user_jobs(user_id)
    return [JobResponse(**job) for job in jobs]


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str, _: Dict = Depends(require_api_key)):
    """Delete a job from the database"""
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    del jobs_db[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


# ============== API KEY MANAGEMENT ==============

@app.post("/api/keys")
async def create_api_key(request: CreateApiKeyRequest, auth: Dict = Depends(require_api_key)):
    """
    Generate a new API key.
    The full key is only returned once at creation time.
    Requires admin API key.
    """
    if not auth.get("is_admin"):
        raise HTTPException(
            status_code=403,
            detail="Only admin can generate new API keys"
        )
    result = generate_api_key(name=request.name, description=request.description)
    return result


@app.get("/api/keys")
async def list_all_api_keys(auth: Dict = Depends(require_api_key)):
    """List all API keys (without exposing the actual keys). Requires admin API key."""
    if not auth.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only admin can list API keys")
    keys = list_api_keys()
    return {"keys": keys, "total": len(keys)}


@app.get("/api/keys/{key_id}")
async def get_api_key(key_id: str, auth: Dict = Depends(require_api_key)):
    """Get details of a specific API key. Requires admin API key."""
    if not auth.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only admin can view API key details")
    key = get_api_key_by_id(key_id)
    if not key:
        raise HTTPException(status_code=404, detail=f"API key {key_id} not found")
    return key


@app.post("/api/keys/{key_id}/revoke")
async def revoke_key(key_id: str, auth: Dict = Depends(require_api_key)):
    """Revoke an API key (soft delete). Requires admin API key."""
    if not auth.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only admin can revoke API keys")
    if revoke_api_key(key_id):
        return {"message": f"API key {key_id} has been revoked"}
    raise HTTPException(status_code=404, detail=f"API key {key_id} not found")


@app.delete("/api/keys/{key_id}")
async def delete_key(key_id: str, auth: Dict = Depends(require_api_key)):
    """Permanently delete an API key. Requires admin API key."""
    if not auth.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only admin can delete API keys")
    if delete_api_key(key_id):
        return {"message": f"API key {key_id} has been deleted"}
    raise HTTPException(status_code=404, detail=f"API key {key_id} not found")


@app.get("/api/health")
async def health():
    """API health check"""
    return {
        "status": "healthy",
        "total_hg_jobs": len(jobs_db),
        "total_split_jobs": len(splits_jobs_db),
    }


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Beautiful landing page for Datamio API - served from templates/index.html"""
    html_file = Path(__file__).parent / "templates" / "index.html"
    html_content = html_file.read_text()
    return HTMLResponse(content=html_content)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)