"""
FastAPI endpoint for uploading multi-domain audio datasets to Hugging Face
Using async job processing with job tracking
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from services.hg import jobs_db,UploadAudioDatasetRequest,JobResponse,JobStatus,create_job,update_job,get_job,get_user_jobs,process_upload_job
from services.splits import process_audio_file
app = FastAPI()

# Mount static files for serving audio chunks
Path("audio_chunks").mkdir(exist_ok=True)
Path("processed_audio").mkdir(exist_ok=True)
app.mount("/audio_chunks", StaticFiles(directory="audio_chunks"), name="audio_chunks")
app.mount("/processed_audio", StaticFiles(directory="processed_audio"), name="processed_audio")

class SplitsRequest(BaseModel):
    audio_url: str
    output_folder: Optional[str] = "audio_chunks"
    threshold: Optional[float] = 0.5
    min_speech_duration_ms: Optional[int] = 250
    min_silence_duration_ms: Optional[int] = 100
    speech_pad_ms: Optional[int] = 30
    output_format: Optional[str] = "wav"
    return_absolute_paths: Optional[bool] = False

class BatchSplitsRequest(BaseModel):
    audio_urls: List[str]
    output_base_folder: Optional[str] = "processed_audio"
    threshold: Optional[float] = 0.5
    min_speech_duration_ms: Optional[int] = 250
    min_silence_duration_ms: Optional[int] = 100
    speech_pad_ms: Optional[int] = 30
    output_format: Optional[str] = "wav"
    return_absolute_paths: Optional[bool] = False


@app.post("/api/splits/file")
async def splits_file(request: SplitsRequest, http_request: Request):
    """
    Split audio file by silence detection.
    Accepts a URL to an audio file and returns segments with chunk file paths.
    """
    try:
        print(f"Processing audio from: {request.audio_url}")
        print("="*60)

        segments = process_audio_file(
            request.audio_url,
            output_folder=request.output_folder,
            threshold=request.threshold,
            min_speech_duration_ms=request.min_speech_duration_ms,
            min_silence_duration_ms=request.min_silence_duration_ms,
            speech_pad_ms=request.speech_pad_ms,
            output_format=request.output_format,
            return_absolute_paths=request.return_absolute_paths
        )

        # Convert relative paths to full URLs
        base_url = str(http_request.base_url).rstrip('/')
        for segment in segments:
            segment['url'] = f"{base_url}/{segment['url']}"

        print(f"\n✓ Processed {len(segments)} segments")

        return {"segments": segments, "count": len(segments)}

    except HTTPException:
        raise
    except Exception as error:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process audio file",
                "details": str(error),
                "stack": traceback.format_exc(),
            }
        )


@app.post("/api/splits/batch")
async def splits_batch(request: BatchSplitsRequest, http_request: Request):
    """
    Split multiple audio files by silence detection.
    Accepts a list of URLs to audio files and returns segments for each.
    """
    try:
        print(f"\nBatch processing {len(request.audio_urls)} audio files")
        print("="*60)

        base_url = str(http_request.base_url).rstrip('/')
        results = {}

        for i, audio_url in enumerate(request.audio_urls, 1):
            print(f"\nProcessing file {i}/{len(request.audio_urls)}: {audio_url}")

            # Extract filename from URL for output folder
            from urllib.parse import urlparse
            url_path = urlparse(audio_url).path
            file_stem = Path(url_path).stem or f"audio_{i}"
            output_folder = str(Path(request.output_base_folder) / file_stem)

            try:
                segments = process_audio_file(
                    audio_url,
                    output_folder=output_folder,
                    threshold=request.threshold,
                    min_speech_duration_ms=request.min_speech_duration_ms,
                    min_silence_duration_ms=request.min_silence_duration_ms,
                    speech_pad_ms=request.speech_pad_ms,
                    output_format=request.output_format,
                    return_absolute_paths=request.return_absolute_paths
                )
                # Convert relative paths to full URLs
                for segment in segments:
                    segment['url'] = f"{base_url}/{segment['url']}"
                results[audio_url] = {"segments": segments, "count": len(segments), "status": "success"}
                print(f"✓ Processed {len(segments)} segments")
            except Exception as e:
                results[audio_url] = {"segments": [], "count": 0, "status": "failed", "error": str(e)}
                print(f"✗ Failed: {str(e)}")

        print(f"\n✓ Batch processing complete: {len(results)} files processed")

        return {"results": results, "total_files": len(results)}

    except HTTPException:
        raise
    except Exception as error:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process batch",
                "details": str(error),
                "stack": traceback.format_exc(),
            }
        )


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
):
    """Background task to process single file splits job"""
    try:
        update_job(
            job_id,
            status=JobStatus.SPLITTING,
            message=f"Processing audio: {audio_url}",
        )

        segments = process_audio_file(
            audio_url,
            output_folder=output_folder,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            output_format=output_format,
            return_absolute_paths=return_absolute_paths
        )

        # Convert relative paths to full URLs
        for segment in segments:
            segment['url'] = f"{base_url}/{segment['url']}"

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message=f"Successfully processed {len(segments)} segments",
            progress={"processed_segments": len(segments)},
            result={
                "segments": segments,
                "count": len(segments),
            }
        )

    except Exception as error:
        import traceback
        update_job(
            job_id,
            status=JobStatus.FAILED,
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
):
    """Background task to process batch splits job"""
    try:
        update_job(
            job_id,
            status=JobStatus.SPLITTING,
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
            update_job(
                job_id,
                message=f"Processing file {i}/{len(audio_urls)}: {audio_url}",
                progress={"current_file": i}
            )

            # Extract filename from URL for output folder
            url_path = urlparse(audio_url).path
            file_stem = Path(url_path).stem or f"audio_{i}"
            output_folder = str(Path(output_base_folder) / file_stem)

            try:
                segments = process_audio_file(
                    audio_url,
                    output_folder=output_folder,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms,
                    output_format=output_format,
                    return_absolute_paths=return_absolute_paths
                )
                # Convert relative paths to full URLs
                for segment in segments:
                    segment['url'] = f"{base_url}/{segment['url']}"
                results[audio_url] = {"segments": segments, "count": len(segments), "status": "success"}
                total_segments += len(segments)
            except Exception as e:
                results[audio_url] = {"segments": [], "count": 0, "status": "failed", "error": str(e)}

            update_job(
                job_id,
                progress={
                    "processed_files": i,
                    "total_segments": total_segments,
                }
            )

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message=f"Batch processing complete: {len(audio_urls)} files, {total_segments} segments",
            result={
                "results": results,
                "total_files": len(results),
                "total_segments": total_segments,
            }
        )

    except Exception as error:
        import traceback
        update_job(
            job_id,
            status=JobStatus.FAILED,
            message=f"Batch splits processing failed: {str(error)}",
            error=traceback.format_exc(),
        )


@app.post("/api/splits/file/job", response_model=JobResponse)
async def splits_file_job(
    request: SplitsRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
):
    """
    Create a job for splitting audio file by silence detection.
    Returns immediately with job ID for tracking progress.
    """
    try:
        job_id = create_job()
        base_url = str(http_request.base_url).rstrip('/')

        update_job(
            job_id,
            message=f"Job created for audio: {request.audio_url}",
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
        )

        job_info = get_job(job_id)
        return JobResponse(**job_info)

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


@app.post("/api/splits/batch/job", response_model=JobResponse)
async def splits_batch_job(
    request: BatchSplitsRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
):
    """
    Create a job for splitting multiple audio files by silence detection.
    Returns immediately with job ID for tracking progress.
    """
    try:
        job_id = create_job()
        base_url = str(http_request.base_url).rstrip('/')

        update_job(
            job_id,
            message=f"Batch job created for {len(request.audio_urls)} files",
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
        )

        job_info = get_job(job_id)
        return JobResponse(**job_info)

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


@app.post("/api/upload-audio-dataset", response_model=JobResponse)
async def upload_audio_dataset(
    request: UploadAudioDatasetRequest,
    background_tasks: BackgroundTasks
):
    """
    Create an upload job for audio dataset
    Returns immediately with job ID
    """
    try:
        dataset_items = [item.dict() for item in request.dataset]
        
        if not dataset_items or not request.datasetName or not request.token:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: dataset, datasetName, token"
            )
        
        # Create job
        job_id = create_job()
        
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
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return JobResponse(**job)


@app.get("/api/jobs", response_model=List[JobResponse])
async def get_all_jobs(user_id: str = "default"):
    """Get all jobs for a user"""
    jobs = get_user_jobs(user_id)
    return [JobResponse(**job) for job in jobs]


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from the database"""
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    del jobs_db[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "total_jobs": len(jobs_db),
        "endpoints": {
            "splits_file": "POST /api/splits/file",
            "splits_batch": "POST /api/splits/batch",
            "splits_file_job": "POST /api/splits/file/job",
            "splits_batch_job": "POST /api/splits/batch/job",
            "upload_dataset_job": "POST /api/upload-audio-dataset",
            "get_job": "GET /api/job/{job_id}",
            "get_all_jobs": "GET /api/jobs?user_id=default",
            "delete_job": "DELETE /api/job/{job_id}",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)