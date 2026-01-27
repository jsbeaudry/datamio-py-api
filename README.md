# Datamio Audio API

A FastAPI service for audio processing and dataset management. Split audio files by speech detection and upload audio datasets to Hugging Face Hub.

## Features

- **Audio Splitting**: Detect speech segments using Silero VAD and split audio files automatically
- **Batch Processing**: Process multiple audio files in a single request
- **Async Job Processing**: Long-running tasks run in the background with job tracking
- **HuggingFace Integration**: Upload audio datasets directly to Hugging Face Hub
- **URL Support**: Process audio files from URLs (including S3 presigned URLs)

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (for audio processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/datamio-py-api.git
   cd datamio-py-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your HuggingFace token
   ```

## Usage

### Start the Server

```bash
python server.py
```

Or with uvicorn directly:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check

```
GET /
```

Returns API status and available endpoints.

---

### Audio Splitting

#### Split Single File

```
POST /api/splits/file
```

Split an audio file by silence detection.

**Request Body:**
```json
{
  "audio_url": "https://example.com/audio.wav",
  "output_folder": "audio_chunks",
  "threshold": 0.5,
  "min_speech_duration_ms": 250,
  "min_silence_duration_ms": 100,
  "speech_pad_ms": 30,
  "output_format": "wav"
}
```

**Response:**
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "duration": 2.5,
      "url": "http://localhost:8000/audio_chunks/abc123/audio_chunk_001_0.00s-2.50s.wav"
    }
  ],
  "count": 1
}
```

#### Split Multiple Files (Batch)

```
POST /api/splits/batch
```

Process multiple audio files at once.

**Request Body:**
```json
{
  "audio_urls": [
    "https://example.com/audio1.wav",
    "https://example.com/audio2.wav"
  ],
  "output_base_folder": "processed_audio"
}
```

---

### Async Job Processing

For long-running tasks, use job-based endpoints that return immediately with a job ID.

#### Create Split Job (Single File)

```
POST /api/splits/file/job
```

#### Create Split Job (Batch)

```
POST /api/splits/batch/job
```

#### Upload Audio Dataset to HuggingFace

```
POST /api/upload-audio-dataset
```

**Request Body:**
```json
{
  "dataset": [
    {
      "id": "uuid-1",
      "text": "Transcription text",
      "audio": "https://example.com/audio.wav",
      "speaker_id": "speaker_1",
      "nature": "natural",
      "language": "english",
      "domain": "general"
    }
  ],
  "datasetName": "my-audio-dataset",
  "token": "hf_your_token",
  "isPrivate": 0
}
```

---

### Job Management

#### Get Job Status

```
GET /api/job/{job_id}
```

#### List All Jobs

```
GET /api/jobs?user_id=default
```

#### Delete Job

```
DELETE /api/job/{job_id}
```

---

## Configuration

### VAD Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Speech detection sensitivity (0-1) |
| `min_speech_duration_ms` | 250 | Minimum speech segment length |
| `min_silence_duration_ms` | 100 | Minimum silence to split on |
| `speech_pad_ms` | 30 | Padding around speech segments |
| `output_format` | wav | Output audio format |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for dataset uploads |

## Project Structure

```
datamio-py-api/
├── server.py              # Main FastAPI application
├── services/
│   ├── splits.py          # Audio splitting with Silero VAD
│   └── hg.py              # HuggingFace upload logic
├── diarisation/           # Speaker diarization modules
│   ├── onnx_diarization.py
│   └── vad_diarization.py
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
