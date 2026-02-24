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
    gender: Optional[str] = None
    nature: Optional[str] = None
    status: Optional[str] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    topic: Optional[str] = None
    others: Optional[Dict[str, Any]] = None


class UploadAudioDatasetRequest(BaseModel):
    dataset: List[AudioDatasetItem]
    datasetName: str
    token: str
    isPrivate: int = 0
    webhookUrl: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    created_at: str
    updated_at: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Webhook events to send notifications for
WEBHOOK_EVENTS = {
    JobStatus.PENDING: "CREATED",
    JobStatus.DOWNLOADING: "DOWNLOADING",
    JobStatus.COMPLETED: "COMPLETED",
    JobStatus.FAILED: "ERROR",
}


async def send_webhook(webhook_url: str, job_data: Dict[str, Any], event: str):
    """Send webhook notification for job status change"""
    if not webhook_url:
        return

    payload = {
        "event": event,
        "job_id": job_data["job_id"],
        "status": job_data["status"].value if isinstance(job_data["status"], JobStatus) else job_data["status"],
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


def create_job(user_id: str = "default", webhook_url: Optional[str] = None) -> str:
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
        "webhook_url": webhook_url,
    }

    # Schedule webhook for CREATED event
    if webhook_url:
        asyncio.create_task(send_webhook(webhook_url, jobs_db[job_id], "CREATED"))

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
    if status and status != old_status and status in WEBHOOK_EVENTS:
        webhook_url = job.get("webhook_url")
        if webhook_url:
            asyncio.create_task(send_webhook(webhook_url, job, WEBHOOK_EVENTS[status]))


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
    unique_genders: List[str],
    unique_statuses: List[str],
    unique_topics: List[str],
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
        f"- **{nature.capitalize()}**: {'Computer-generated speech' if nature == 'synthetic' else 'human human speech'}"
        for nature in unique_natures
    ])
    
    # Distribution table
    lang_rows = "\n".join([
        f"| {language_names[i]} | {sum(1 for item in dataset_records if item.get('language') == lang)} samples |"
        for i, lang in enumerate(unique_languages)
    ]) if unique_languages else ""

    domain_rows = "\n".join([
        f"| {domain.capitalize()} | {sum(1 for item in dataset_records if item.get('domain') == domain)} samples |"
        for domain in unique_domains
    ]) if unique_domains else ""

    nature_rows = "\n".join([
        f"| {nature.capitalize()} | {sum(1 for item in dataset_records if item.get('nature') == nature)} samples |"
        for nature in unique_natures
    ]) if unique_natures else ""

    gender_rows = "\n".join([
        f"| {gender.capitalize()} | {sum(1 for item in dataset_records if item.get('gender') == gender)} samples |"
        for gender in unique_genders
    ]) if unique_genders else ""

    status_rows = "\n".join([
        f"| {status.capitalize()} | {sum(1 for item in dataset_records if item.get('status') == status)} samples |"
        for status in unique_statuses
    ]) if unique_statuses else ""

    topic_rows = "\n".join([
        f"| {topic.capitalize()} | {sum(1 for item in dataset_records if item.get('topic') == topic)} samples |"
        for topic in unique_topics
    ]) if unique_topics else ""
    
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

[![Built with Datamio](https://img.shields.io/badge/Built%20with-Datamio-blue)](https://www.datamio.dev)

# Multi-Domain {' & '.join(language_names)} Speech Dataset

This dataset contains {total_audio_files} audio recordings with corresponding text transcriptions across multiple languages and domains.

## Dataset Structure

Each entry contains:
- `id`: Unique identifier (string)
- `text`: Transcription text in the specified language
- `audio`: Audio data (automatically loaded by datasets library)
- `speaker_id`: Speaker identifier (optional)
- `gender`: Speaker gender (optional)
- `nature`: Type of audio, e.g., "synthetic", "human" (optional)
- `status`: Recording status (optional)
- `language`: Language of the audio/text (optional)
- `domain`: Domain/topic category (optional)
- `topic`: Specific topic within domain (optional)
- `others`: Additional metadata as JSON (optional)

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
{primary_language}_data = train_data.filter(lambda x: x['language'] == '{primary_language}')
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
{gender_rows}
{status_rows}
{topic_rows}

## License

MIT License

---

<p align="center">
  <sub>Created with <a href="https://www.datamio.dev">Datamio</a> - The AI Data Platform</sub>
</p>
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
        
        # Get current user info if no username provided (run in thread pool to avoid blocking)
        if not username:
            user_info = await asyncio.to_thread(whoami, token=token)
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
        
        # Extract unique values (filter out None values)
        unique_languages = list(set(item.get("language") for item in dataset_items if item.get("language")))
        unique_domains = list(set(item.get("domain") for item in dataset_items if item.get("domain")))
        unique_natures = list(set(item.get("nature") for item in dataset_items if item.get("nature")))
        unique_genders = list(set(item.get("gender") for item in dataset_items if item.get("gender")))
        unique_statuses = list(set(item.get("status") for item in dataset_items if item.get("status")))
        unique_topics = list(set(item.get("topic") for item in dataset_items if item.get("topic")))
        
        # Language mapping
        language_mapping = {
            "swahili": {"code": "sw", "name": "Swahili"},
            "english": {"code": "en", "name": "English"},
            "french": {"code": "fr", "name": "French"},
            "spanish": {"code": "es", "name": "Spanish"},
            "haitian creole": {"code": "ht", "name": "Haitian Creole"},
            "haitian_creole": {"code": "ht", "name": "Haitian Creole"},
            "haitian": {"code": "ht", "name": "Haitian Creole"},
            "portuguese": {"code": "pt", "name": "Portuguese"},
            "arabic": {"code": "ar", "name": "Arabic"},
            "german": {"code": "de", "name": "German"},
            "italian": {"code": "it", "name": "Italian"},
            "dutch": {"code": "nl", "name": "Dutch"},
            "russian": {"code": "ru", "name": "Russian"},
            "chinese": {"code": "zh", "name": "Chinese"},
            "mandarin": {"code": "zh", "name": "Chinese"},
            "japanese": {"code": "ja", "name": "Japanese"},
            "korean": {"code": "ko", "name": "Korean"},
            "hindi": {"code": "hi", "name": "Hindi"},
            "bengali": {"code": "bn", "name": "Bengali"},
            "urdu": {"code": "ur", "name": "Urdu"},
            "turkish": {"code": "tr", "name": "Turkish"},
            "vietnamese": {"code": "vi", "name": "Vietnamese"},
            "thai": {"code": "th", "name": "Thai"},
            "indonesian": {"code": "id", "name": "Indonesian"},
            "malay": {"code": "ms", "name": "Malay"},
            "tagalog": {"code": "tl", "name": "Tagalog"},
            "filipino": {"code": "tl", "name": "Tagalog"},
            "polish": {"code": "pl", "name": "Polish"},
            "ukrainian": {"code": "uk", "name": "Ukrainian"},
            "czech": {"code": "cs", "name": "Czech"},
            "romanian": {"code": "ro", "name": "Romanian"},
            "hungarian": {"code": "hu", "name": "Hungarian"},
            "greek": {"code": "el", "name": "Greek"},
            "hebrew": {"code": "he", "name": "Hebrew"},
            "persian": {"code": "fa", "name": "Persian"},
            "farsi": {"code": "fa", "name": "Persian"},
            "swedish": {"code": "sv", "name": "Swedish"},
            "norwegian": {"code": "no", "name": "Norwegian"},
            "danish": {"code": "da", "name": "Danish"},
            "finnish": {"code": "fi", "name": "Finnish"},
            "catalan": {"code": "ca", "name": "Catalan"},
            "amharic": {"code": "am", "name": "Amharic"},
            "yoruba": {"code": "yo", "name": "Yoruba"},
            "igbo": {"code": "ig", "name": "Igbo"},
            "hausa": {"code": "ha", "name": "Hausa"},
            "zulu": {"code": "zu", "name": "Zulu"},
            "xhosa": {"code": "xh", "name": "Xhosa"},
            "afrikaans": {"code": "af", "name": "Afrikaans"},
            "somali": {"code": "so", "name": "Somali"},
            "tamil": {"code": "ta", "name": "Tamil"},
            "telugu": {"code": "te", "name": "Telugu"},
            "marathi": {"code": "mr", "name": "Marathi"},
            "gujarati": {"code": "gu", "name": "Gujarati"},
            "punjabi": {"code": "pa", "name": "Punjabi"},
            "kannada": {"code": "kn", "name": "Kannada"},
            "malayalam": {"code": "ml", "name": "Malayalam"},
            "nepali": {"code": "ne", "name": "Nepali"},
            "sinhala": {"code": "si", "name": "Sinhala"},
            "burmese": {"code": "my", "name": "Burmese"},
            "khmer": {"code": "km", "name": "Khmer"},
            "lao": {"code": "lo", "name": "Lao"},
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
                "text": record["text"],
                "audio": audio_path,
                "speaker_id": record.get("speaker_id"),
                "gender": record.get("gender"),
                "nature": record.get("nature"),
                "status": record.get("status"),
                "language": record.get("language"),
                "domain": record.get("domain"),
                "topic": record.get("topic"),
                "others": json.dumps(record.get("others")) if record.get("others") else None,
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

        # Run blocking dataset creation in thread pool to avoid blocking event loop
        def create_dataset_sync():
            return DatasetDict({
                "train": Dataset.from_list(dataset_records).cast_column(
                    "audio",
                    Audio(sampling_rate=24000)
                )
            })

        dataset = await asyncio.to_thread(create_dataset_sync)
        
        # Generate README
        readme_content = create_readme_content(
            dataset_records=dataset_items,
            final_repo_name=final_repo_name,
            unique_languages=unique_languages,
            unique_domains=unique_domains,
            unique_natures=unique_natures,
            unique_genders=unique_genders,
            unique_statuses=unique_statuses,
            unique_topics=unique_topics,
            total_audio_files=total_audio_files,
            avg_text_length=avg_text_length,
            language_codes=language_codes,
            language_names=language_names,
        )
        
        # Check if repository exists (run in thread pool to avoid blocking)
        repository_exists = False
        api = HfApi(token=token)

        try:
            await asyncio.to_thread(
                api.repo_info,
                repo_id=final_repo_name,
                repo_type="dataset",
                token=token,
            )
            repository_exists = True
        except Exception:
            pass
        
        # Push dataset to hub
        update_job(
            job_id,
            status=JobStatus.UPLOADING,
            message=f"Uploading dataset to {final_repo_name}",
        )

        # Run blocking push_to_hub in thread pool to avoid blocking event loop
        await asyncio.to_thread(
            dataset.push_to_hub,
            final_repo_name,
            token=token,
            private=is_private == 1,
        )

        # Upload README separately (also blocking, run in thread pool)
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=final_repo_name,
            repo_type="dataset",
            token=token,
        )
        
        # Calculate distribution statistics
        distribution_stats = {
            "byLanguage": {
                lang: sum(1 for item in dataset_items if item.get("language") == lang)
                for lang in unique_languages
            },
            "byDomain": {
                domain: sum(1 for item in dataset_items if item.get("domain") == domain)
                for domain in unique_domains
            },
            "byNature": {
                nature: sum(1 for item in dataset_items if item.get("nature") == nature)
                for nature in unique_natures
            },
            "byGender": {
                gender: sum(1 for item in dataset_items if item.get("gender") == gender)
                for gender in unique_genders
            },
            "byStatus": {
                status: sum(1 for item in dataset_items if item.get("status") == status)
                for status in unique_statuses
            },
            "byTopic": {
                topic: sum(1 for item in dataset_items if item.get("topic") == topic)
                for topic in unique_topics
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
                    "genders": unique_genders,
                    "statuses": unique_statuses,
                    "topics": unique_topics,
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

