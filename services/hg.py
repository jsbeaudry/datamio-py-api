"""
FastAPI endpoint for uploading multi-domain audio datasets to Hugging Face
Using async job processing with job tracking
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from huggingface_hub import (
    HfApi,
    create_repo,
    whoami,
)
from datasets import Dataset, DatasetDict, Audio
import json
import requests
from datetime import datetime
import asyncio
import aiohttp
import tempfile
import os
from pathlib import Path
import uuid
from enum import Enum

app = FastAPI()

# In-memory job storage (use Redis or database in production)
jobs_db: Dict[str, Dict[str, Any]] = {}


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DOWNLOADING = "downloading"
    SPLITTING = "splitting"
    CREATING_DATASET = "creating_dataset"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioDatasetItem(BaseModel):
    id: str
    text: str
    audio: str
    speaker_id: Optional[str] = None
    nature: str
    language: str
    domain: str


class UploadAudioDatasetRequest(BaseModel):
    dataset: List[AudioDatasetItem]
    datasetName: str
    token: str
    isPrivate: int = 0


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    created_at: str
    updated_at: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def create_job(user_id: str = "default") -> str:
    """Create a new job and return job ID"""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    jobs_db[job_id] = {
        "job_id": job_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "message": "Job created, waiting to start",
        "created_at": now,
        "updated_at": now,
        "progress": {
            "total_files": 0,
            "downloaded_files": 0,
            "uploaded": False,
        },
        "result": None,
        "error": None,
    }
    
    return job_id


def update_job(
    job_id: str,
    status: Optional[JobStatus] = None,
    message: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    """Update job information"""
    if job_id not in jobs_db:
        return
    
    job = jobs_db[job_id]
    
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


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job information by ID"""
    return jobs_db.get(job_id)


def get_user_jobs(user_id: str = "default") -> List[Dict[str, Any]]:
    """Get all jobs for a user"""
    return [
        job for job in jobs_db.values()
        if job["user_id"] == user_id
    ]


async def download_audio_file(url: str, save_path: str) -> str:
    """Download audio file from URL and save to local path"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch audio: {response.reason}")
            
            # Save to file
            with open(save_path, 'wb') as f:
                f.write(await response.read())
            
            return save_path


def create_readme_content(
    dataset_records: List[Dict[str, Any]],
    final_repo_name: str,
    unique_languages: List[str],
    unique_domains: List[str],
    unique_natures: List[str],
    total_audio_files: int,
    avg_text_length: int,
    language_codes: List[str],
    language_names: List[str],
) -> str:
    """Generate comprehensive README content with dataset card"""
    
    primary_language = unique_languages[0].lower() if unique_languages else "multilingual"
    
    # Determine size category
    if total_audio_files < 1000:
        size_category = "n<1K"
    elif total_audio_files < 10000:
        size_category = "1K<n<10K"
    elif total_audio_files < 100000:
        size_category = "10K<n<100K"
    else:
        size_category = "n>100K"
    
    # Language sections
    language_list = "\n".join([f"- {code}" for code in language_codes])
    
    languages_desc = "\n".join([
        f"- **{language_names[i]}** ({language_codes[i]}): {unique_languages[i]}"
        for i in range(len(unique_languages))
    ])
    
    # Domain sections
    domains_desc = "\n".join([
        f"- **{domain.capitalize()}**: Domain-specific terminology and context"
        for domain in unique_domains
    ])
    
    # Nature sections
    natures_desc = "\n".join([
        f"- **{nature.capitalize()}**: {'Computer-generated speech' if nature == 'synthetic' else 'Natural human speech'}"
        for nature in unique_natures
    ])
    
    # Distribution table
    lang_rows = "\n".join([
        f"| {language_names[i]} | {sum(1 for item in dataset_records if item.get('language') == lang)} samples |"
        for i, lang in enumerate(unique_languages)
    ])
    
    domain_rows = "\n".join([
        f"| {domain.capitalize()} | {sum(1 for item in dataset_records if item.get('domain') == domain)} samples |"
        for domain in unique_domains
    ])
    
    nature_rows = "\n".join([
        f"| {nature.capitalize()} | {sum(1 for item in dataset_records if item.get('nature') == nature)} samples |"
        for nature in unique_natures
    ])
    
    current_year = datetime.now().year
    
    readme = f"""---
license: mit
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- speech
- audio
- {primary_language.replace(' ', '-')}
- multilingual
language:
{language_list}
size_categories:
- {size_category}
pretty_name: Multi-Domain {' & '.join(language_names)} Speech Dataset
---

# Multi-Domain {' & '.join(language_names)} Speech Dataset

This dataset contains {total_audio_files} audio recordings with corresponding text transcriptions across multiple languages and domains.

## Dataset Structure

Each entry contains:
- `id`: Unique identifier (UUID)
- `text`: Transcription text in the specified language
- `audio`: Audio data (automatically loaded by datasets library)
- `speaker_id`: Speaker identifier
- `nature`: Type of audio (e.g., "synthetic", "natural")
- `language`: Language of the audio/text
- `domain`: Domain/topic category

## Languages

{languages_desc}

## Domains

{domains_desc}

## Audio Nature

{natures_desc}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{final_repo_name}")
train_data = dataset["train"]

# Filter by language
swahili_data = train_data.filter(lambda x: x['language'] == 'swahili')
```

## Dataset Statistics

- **Total audio files**: {total_audio_files}
- **Languages**: {len(unique_languages)} ({', '.join(language_names)})
- **Domains**: {len(unique_domains)} ({', '.join(unique_domains)})
- **Audio types**: {', '.join(unique_natures)}
- **Average text length**: {avg_text_length} characters

### Distribution by Category

| Category | Values |
|----------|---------|
{lang_rows}
{domain_rows}
{nature_rows}

## License

MIT License
"""
    
    return readme


async def process_upload_job(
    job_id: str,
    dataset_items: List[Dict[str, Any]],
    dataset_name: str,
    token: str,
    is_private: int,
):
    """Background task to process the upload"""
    temp_dir = None
    
    try:
        update_job(
            job_id,
            status=JobStatus.PROCESSING,
            message="Starting upload process",
        )
        
        # Parse repository name
        if "/" in dataset_name:
            username, repo_name = dataset_name.split("/", 1)
        else:
            username = None
            repo_name = dataset_name
        
        # Get current user info if no username provided
        if not username:
            user_info = whoami(token=token)
            final_repo_name = f"{user_info['name']}/{repo_name}"
        else:
            final_repo_name = dataset_name
        
        update_job(
            job_id,
            message=f"Processing dataset: {final_repo_name}",
        )
        
        # Create temporary directory for audio files
        temp_dir = tempfile.mkdtemp()
        
        # Analyze dataset for statistics
        total_audio_files = len(dataset_items)
        avg_text_length = round(
            sum(len(item["text"]) for item in dataset_items) / total_audio_files
        )
        
        # Extract unique values
        unique_languages = list(set(item["language"] for item in dataset_items))
        unique_domains = list(set(item["domain"] for item in dataset_items))
        unique_natures = list(set(item["nature"] for item in dataset_items))
        
        # Language mapping
        language_mapping = {
            "swahili": {"code": "sw", "name": "Swahili"},
            "english": {"code": "en", "name": "English"},
            "french": {"code": "fr", "name": "French"},
            "spanish": {"code": "es", "name": "Spanish"},
            "haitian creole": {"code": "ht", "name": "Haitian Creole"},
            "portuguese": {"code": "pt", "name": "Portuguese"},
            "arabic": {"code": "ar", "name": "Arabic"},
        }
        
        language_codes = [
            language_mapping.get(lang.lower(), {}).get("code", lang.lower()[:2])
            for lang in unique_languages
        ]
        
        language_names = [
            language_mapping.get(lang.lower(), {}).get("name", lang.capitalize())
            for lang in unique_languages
        ]
        
        # Update progress
        update_job(
            job_id,
            status=JobStatus.DOWNLOADING,
            message="Downloading audio files",
            progress={
                "total_files": total_audio_files,
                "downloaded_files": 0,
            }
        )
        
        # Download audio files and prepare dataset records
        dataset_records = []
        
        for i, record in enumerate(dataset_items):
            # Create local filename
            audio_filename = f"{record['id']}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            # Download audio file
            await download_audio_file(record["audio"], audio_path)
            
            # Create dataset record with local audio path
            dataset_record = {
                "id": record["id"],
                "speaker_id": record.get("speaker_id", ""),
                "text": record["text"],
                "audio": audio_path,
                "nature": record["nature"],
                "language": record["language"],
                "domain": record["domain"],
            }
            
            dataset_records.append(dataset_record)
            
            # Update progress
            update_job(
                job_id,
                message=f"Downloaded {i+1}/{total_audio_files} audio files",
                progress={"downloaded_files": i + 1}
            )
        
        # Create dataset using datasets library
        update_job(
            job_id,
            status=JobStatus.CREATING_DATASET,
            message="Creating dataset structure",
        )
        
        dataset = DatasetDict({
            "train": Dataset.from_list(dataset_records).cast_column(
                "audio", 
                Audio(sampling_rate=24000)
            )
        })
        
        # Generate README
        readme_content = create_readme_content(
            dataset_records=dataset_items,
            final_repo_name=final_repo_name,
            unique_languages=unique_languages,
            unique_domains=unique_domains,
            unique_natures=unique_natures,
            total_audio_files=total_audio_files,
            avg_text_length=avg_text_length,
            language_codes=language_codes,
            language_names=language_names,
        )
        
        # Check if repository exists
        repository_exists = False
        api = HfApi(token=token)
        
        try:
            api.repo_info(repo_id=final_repo_name, repo_type="dataset", token=token)
            repository_exists = True
        except Exception:
            pass
        
        # Push dataset to hub
        update_job(
            job_id,
            status=JobStatus.UPLOADING,
            message=f"Uploading dataset to {final_repo_name}",
        )
        
        dataset.push_to_hub(
            final_repo_name,
            token=token,
            private=is_private == 1,
        )
        
        # Upload README separately
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=final_repo_name,
            repo_type="dataset",
            token=token,
        )
        
        # Calculate distribution statistics
        distribution_stats = {
            "byLanguage": {
                lang: sum(1 for item in dataset_items if item["language"] == lang)
                for lang in unique_languages
            },
            "byDomain": {
                domain: sum(1 for item in dataset_items if item["domain"] == domain)
                for domain in unique_domains
            },
            "byNature": {
                nature: sum(1 for item in dataset_items if item["nature"] == nature)
                for nature in unique_natures
            },
        }
        
        # Job completed successfully
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            message="Upload completed successfully",
            progress={"uploaded": True},
            result={
                "success": True,
                "repoUrl": f"https://huggingface.co/datasets/{final_repo_name}",
                "action": "updated" if repository_exists else "created",
                "stats": {
                    "totalAudioFiles": total_audio_files,
                    "languages": unique_languages,
                    "languageCodes": language_codes,
                    "domains": unique_domains,
                    "natures": unique_natures,
                    "avgTextLength": avg_text_length,
                    "distribution": distribution_stats,
                },
            }
        )
        
    except Exception as error:
        import traceback
        update_job(
            job_id,
            status=JobStatus.FAILED,
            message=f"Upload failed: {str(error)}",
            error=traceback.format_exc(),
        )
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory: {e}")


# @app.post("/api/upload-audio-dataset", response_model=JobResponse)
# async def upload_audio_dataset(
#     request: UploadAudioDatasetRequest,
#     background_tasks: BackgroundTasks
# ):
#     """
#     Create an upload job for audio dataset
#     Returns immediately with job ID
#     """
#     try:
#         dataset_items = [item.dict() for item in request.dataset]
        
#         if not dataset_items or not request.datasetName or not request.token:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Missing required fields: dataset, datasetName, token"
#             )
        
#         # Create job
#         job_id = create_job()
        
#         # Update with initial info
#         update_job(
#             job_id,
#             message=f"Job created for dataset: {request.datasetName}",
#             progress={
#                 "total_files": len(dataset_items),
#                 "downloaded_files": 0,
#                 "uploaded": False,
#             }
#         )
        
#         # Add background task
#         background_tasks.add_task(
#             process_upload_job,
#             job_id=job_id,
#             dataset_items=dataset_items,
#             dataset_name=request.datasetName,
#             token=request.token,
#             is_private=request.isPrivate,
#         )
        
#         # Return job info immediately
#         job_info = get_job(job_id)
#         return JobResponse(**job_info)
        
#     except HTTPException:
#         raise
#     except Exception as error:
#         import traceback
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "error": "Failed to create upload job",
#                 "details": str(error),
#                 "stack": traceback.format_exc(),
#             }
#         )


# @app.get("/api/job/{job_id}", response_model=JobResponse)
# async def get_job_status(job_id: str):
#     """Get status of a specific job"""
#     job = get_job(job_id)
    
#     if not job:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Job {job_id} not found"
#         )
    
#     return JobResponse(**job)


# @app.get("/api/jobs", response_model=List[JobResponse])
# async def get_all_jobs(user_id: str = "default"):
#     """Get all jobs for a user"""
#     jobs = get_user_jobs(user_id)
#     return [JobResponse(**job) for job in jobs]


# @app.delete("/api/job/{job_id}")
# async def delete_job(job_id: str):
#     """Delete a job from the database"""
#     if job_id not in jobs_db:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Job {job_id} not found"
#         )
    
#     del jobs_db[job_id]
#     return {"message": f"Job {job_id} deleted successfully"}


# @app.get("/")
# async def root():
#     """API health check"""
#     return {
#         "status": "healthy",
#         "total_jobs": len(jobs_db),
#         "endpoints": {
#             "create_job": "POST /api/upload-audio-dataset",
#             "get_job": "GET /api/job/{job_id}",
#             "get_all_jobs": "GET /api/jobs?user_id=default",
#             "delete_job": "DELETE /api/job/{job_id}",
#         }
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)