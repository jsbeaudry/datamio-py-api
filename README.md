# Datamio Audio API

A FastAPI service for audio processing and dataset management. Split audio files by speech detection and upload audio datasets to Hugging Face Hub.

## Features

- **Audio Splitting**: Detect speech segments using Silero VAD and split audio files automatically
- **Batch Processing**: Process multiple audio files in a single request
- **Async Job Processing**: Long-running tasks run in the background with job tracking
- **HuggingFace Integration**: Upload audio datasets directly to Hugging Face Hub
- **URL Support**: Process audio files from URLs (including S3 presigned URLs)
- **API Key Authentication**: Secure endpoints with API key-based authentication

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
   # Edit .env and add your configuration
   ```

5. Set your admin API key (required for generating user API keys):
   ```bash
   export ADMIN_API_KEY="your-secure-admin-key-here"
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

## Authentication

All API endpoints (except `GET /`) require an API key passed in the `X-API-Key` header.

### Setting Up Authentication

1. **Set the admin API key** (environment variable):
   ```bash
   export ADMIN_API_KEY="your-secure-admin-key-here"
   ```

2. **Generate API keys for users** (using the admin key):
   ```bash
   curl -X POST http://localhost:8000/api/keys \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $ADMIN_API_KEY" \
     -d '{"name": "user1", "description": "API key for user 1"}'
   ```

   Response:
   ```json
   {
     "api_key": "datamio_abc123...",
     "key_id": "a1b2c3d4",
     "name": "user1",
     "key_prefix": "datamio_abc1...",
     "message": "Store this API key securely. It will not be shown again."
   }
   ```

3. **Use the API key in requests**:
   ```bash
   curl http://localhost:8000/api/splits/jobs \
     -H "X-API-Key: datamio_abc123..."
   ```

### Key Types

| Key Type | Source | Capabilities |
|----------|--------|--------------|
| Admin Key | `ADMIN_API_KEY` env var | Full access + can create/manage API keys |
| User Key | Generated via API | Access all endpoints except key management |

### API Key Management Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/api/keys` | POST | Admin only | Generate a new API key |
| `/api/keys` | GET | Admin only | List all API keys |
| `/api/keys/{key_id}` | GET | Admin only | Get details of a specific key |
| `/api/keys/{key_id}/revoke` | POST | Admin only | Revoke (disable) a key |
| `/api/keys/{key_id}` | DELETE | Admin only | Permanently delete a key |

---

### API Endpoints

#### Health Check

```
GET /
```

Returns API status and available endpoints.

---

### Audio Splitting

> **Note:** All endpoints below require the `X-API-Key` header.

#### Split Single File

```
POST /api/splits/file
```

Split an audio file by silence detection.

**Headers:**
```
X-API-Key: your-api-key
Content-Type: application/json
```

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

| Variable | Required | Description |
|----------|----------|-------------|
| `ADMIN_API_KEY` | Yes | Admin API key for generating and managing user API keys |
| `HF_TOKEN` | No | HuggingFace API token for dataset uploads |

## Project Structure

```
datamio-py-api/
├── server.py              # Main FastAPI application
├── services/
│   ├── auth.py            # API key authentication
│   ├── splits.py          # Audio splitting with Silero VAD
│   └── hg.py              # HuggingFace upload logic
├── diarisation/           # Speaker diarization modules
│   ├── onnx_diarization.py
│   └── vad_diarization.py
├── api_keys.json          # Generated API keys storage (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
