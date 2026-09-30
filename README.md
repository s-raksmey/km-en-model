
# Khmer Translation API

A FastAPI backend for translating Khmer text into English using the fine-tuned `facebook/nllb-200-distilled-600M` model with a LoRA adapter.

## Features

- Khmer-to-English translation
- Fine-tuned NLLB-200 distilled 600M model
- LoRA adapter integration
- FastAPI REST API
- Interactive Swagger documentation
- Health-check endpoint
- CORS support for a Next.js frontend

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- NLLB-200 distilled 600M

## Project Structure

```text
khmer-translation-api/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── nllb_lora_cleaned_final/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

The adapter files shown above are stored locally and are not committed to GitHub.

## Requirements

- Git
- Python 3.10 or 3.11 recommended
- Internet connection
- Several GB of free disk space for the base model and its cache
- Sufficient RAM for model inference

A compatible NVIDIA GPU can improve inference speed. CPU inference is supported but may be slow.

## Installation on a New Device

### 1. Clone the Repository

Replace the URL with your GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd khmer-translation-api
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you use a different Python version, replace `py -3.11` with the appropriate command.

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a GPU-specific PyTorch installation, follow the official guide:

https://pytorch.org/get-started/locally/

### 4. Add the Trained LoRA Adapter

Obtain the trained adapter from your private backup or Google Drive.

Place its files inside:

```text
models/nllb_lora_cleaned_final/
```

Ensure the directory contains:

- `adapter_config.json`
- `adapter_model.safetensors`

Keep the files directly inside this directory, without an extra nested folder.

The adapter is required because it contains your fine-tuned weights. The original NLLB base model is downloaded automatically from Hugging Face when the API starts for the first time.

Do not commit large model weights or private adapter files to GitHub.

### 5. Start the API

From the project root, run:

```bash
python -m uvicorn main:app --reload
```

Wait until the terminal reports that application startup is complete.

The first startup may take several minutes because the base model must be downloaded and loaded.

### 6. Open the API Documentation

Once the server is running, open:

http://127.0.0.1:8000/docs

You can use the interactive Swagger UI to test the endpoints.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check whether the model has loaded |
| POST | `/translate` | Translate Khmer text into English |

## Translation Example

### Request

**POST** `/translate`

```json
{
  "text": "ប្រទេសកម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។"
}
```

### Example Response

```json
{
  "source_language": "km",
  "target_language": "en",
  "original_text": "ប្រទេសកម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។",
  "translated_text": "Cambodia is in Southeast Asia."
}
```

The exact English wording may vary.

## Connecting to Next.js

The API currently allows these frontend origins:

- `http://localhost:3000`
- `http://127.0.0.1:3000`

If your frontend uses another origin, add it to `ALLOWED_ORIGINS` in `main.py`.

When the frontend is deployed, deploy the API to a reachable server as well. A deployed website cannot access your local computer through `127.0.0.1:8000`.

## GitHub Files

Commit these project files:

- `main.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

Do not commit:

- `.venv/`
- `models/`
- Model weight files
- `.env` files or secrets
- Python cache files

## Troubleshooting

### Connection Refused

Make sure Uvicorn is running and application startup has completed. Check the terminal for errors.

### Adapter Directory Not Found

Verify that your adapter exists at:

```text
models/nllb_lora_cleaned_final/
```

### Pylance Cannot Resolve `peft`

Select your project's `.venv` Python interpreter in VS Code, then run:

```bash
python -m pip install -r requirements.txt
```

### Slow First Startup

The NLLB base model is large and needs to be downloaded the first time. Subsequent runs can reuse the local cache.

### Out of Memory

Close other memory-intensive applications. If necessary, use a computer with more RAM or a compatible GPU.

## Current Limitation

This API currently supports Khmer-to-English translation only. English-to-Khmer requires a separately configured and evaluated translation model.