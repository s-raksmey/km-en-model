
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL = "facebook/nllb-200-distilled-600M"

ADAPTER_DIR = (
    BASE_DIR
    / "models"
    / "nllb_lora_cleaned_final"
)

SOURCE_LANGUAGE = "khm_Khmr"
TARGET_LANGUAGE = "eng_Latn"

MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128

# Add your deployed frontend URL here when you deploy.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# Model state
# =========================================================

tokenizer = None
model = None
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Load and release model
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    logger.info("Starting Khmer Translation API...")
    logger.info("Using device: %s", device)

    if not ADAPTER_DIR.is_dir():
        raise RuntimeError(
            f"LoRA adapter directory not found: {ADAPTER_DIR}"
        )

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (ADAPTER_DIR / filename).is_file()
    ]

    if missing_files:
        raise RuntimeError(
            "Missing adapter files: "
            + ", ".join(missing_files)
            + ". Check your extracted adapter directory."
        )

    try:
        # Load the tokenizer for Khmer source text.
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            src_lang=SOURCE_LANGUAGE,
        )

        # Use half precision on NVIDIA GPU to reduce memory use.
        model_dtype = (
            torch.float16
            if device.type == "cuda"
            else torch.float32
        )

        # Download the base model on first run if not cached.
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=model_dtype,
        )

        # Load your fine-tuned LoRA adapter.
        model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_DIR),
        )

        model.to(device)
        model.eval()

        logger.info("Translation model loaded successfully.")

        yield

    except Exception:
        logger.exception("Failed to load translation model.")
        raise

    finally:
        model = None
        tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Translation model resources released.")


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Khmer Translation API",
    description=(
        "Translate Khmer text into English using "
        "a fine-tuned NLLB-200 model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# =========================================================
# Request and response schemas
# =========================================================

class TranslationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="Khmer text to translate into English.",
            examples=["សួស្តី តើអ្នកសុខសប្បាយជាទេ?"],
        ),
    ]


class TranslationResponse(BaseModel):
    source_language: str
    target_language: str
    original_text: str
    translated_text: str


# =========================================================
# Routes
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Khmer Translation API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post(
    "/translate",
    response_model=TranslationResponse,
)
def translate(request: TranslationRequest):
    # Ensure the model has finished loading.
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Translation model is not ready.",
        )

    text = request.text

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty.",
        )

    try:
        # Tell NLLB the source language.
        tokenizer.src_lang = SOURCE_LANGUAGE

        # Convert Khmer text into model input tensors.
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)

        # Generate English translation.
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=(
                    tokenizer.convert_tokens_to_ids(
                        TARGET_LANGUAGE
                    )
                ),
                max_new_tokens=MAX_OUTPUT_LENGTH,
                num_beams=4,
            )

        translated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        logger.info("Translation completed successfully.")

        return TranslationResponse(
            source_language="km",
            target_language="en",
            original_text=text,
            translated_text=translated_text,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Translation failed.")

        raise HTTPException(
            status_code=500,
            detail="Translation failed. Please try again.",
        ) from exc