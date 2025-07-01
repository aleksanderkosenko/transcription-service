from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import whisperx
import tempfile
import os
import gc
from typing import Dict, Optional
from enum import Enum
from itertools import groupby
import whisper
from whisperx.utils import get_writer
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import uvicorn

# Define an Enum for model names for better validation and OpenAPI spec
class ModelName(str, Enum):
    small = "small"
    medium = "medium"
    large_v2 = "large-v2"

# Define an Enum for supported languages
class Language(str, Enum):
    uk = "uk"
    en = "en"
    de = "de"
    es = "es"
    fr = "fr"
    it = "it"

# --- Model Caching ---
# ASR models
models: Dict[str, any] = {}
DEFAULT_MODEL = ModelName.small
# Alignment models
alignment_models: Dict[str, tuple] = {}
# Diarization model
diarization_model = None

# Define compute type for Mac
compute_type = "int8"
# Define device
device = "cpu"

def load_model_into_cache(model_name: str):
    """Loads a model into the global 'models' dictionary if not already loaded."""
    if model_name not in models:
        print(f"Loading model '{model_name}' from cache...")
        models[model_name] = whisperx.load_model(model_name, device, compute_type=compute_type)
        print(f"Model '{model_name}' loaded into memory.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the default model on startup
    print("--- Application starting up ---")
    load_model_into_cache(DEFAULT_MODEL.value)

    # Try to load the diarization model
    global diarization_model
    print("Attempting to load Diarization model...")
    try:
        # The token is only used for the initial download of a gated model.
        # If the model is already cached, it will load without a token.
        token = os.getenv("HF_TOKEN")
        diarization_model = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
        print("Diarization model loaded successfully.")
    except Exception as e:
        diarization_model = None
        print("\n--- Diarization Model Loading Failed ---")
        print("To enable speaker separation (diarization), a one-time setup is required:")
        print("1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1 and agree to the terms.")
        print("2. Go to https://huggingface.co/settings/tokens to create a 'read' access token.")
        print("3. Set the token as an environment variable: export HF_TOKEN='your_token_here'")
        print("4. Restart the server. The model will be downloaded once and cached for offline use.")
        print(f"(Error details: {e})\n")

    yield
    # Clean up resources on shutdown
    print("--- Application shutting down ---")
    global models, alignment_models
    models.clear()
    alignment_models.clear()
    diarization_model = None
    gc.collect()


app = FastAPI(
    title="Whisper-X ASR API",
    description="An API for highly accurate speech-to-text transcription and speaker identification.",
    version="1.0.0",
    lifespan=lifespan
)

def format_output(result: dict, diarize: bool) -> str:
    """Formats the transcription result into a human-readable string."""
    if not diarize or "speaker" not in result.get("segments", [{}])[0].get("words", [{}])[0]:
        # Simple text output if not diarized or if words don't have speaker info
        return result.get("text", "No text transcribed.")

    # Human-readable diarized output
    lines = []
    for segment in result["segments"]:
        words = segment.get("words", [])
        if not words:
            continue

        # Group consecutive words by the same speaker
        for speaker, group in groupby(words, key=lambda x: x.get('speaker', 'UNKNOWN')):
            word_list = list(group)
            if not word_list:
                continue
            
            start = word_list[0]['start']
            end = word_list[-1]['end']
            text = " ".join([w['word'].strip() for w in word_list])
            
            # Format timestamps
            start_m, start_s = divmod(start, 60)
            
            lines.append(f"[{int(start_m):02}:{start_s:05.2f}] {speaker}: {text}")
            
    return "\n".join(lines) if lines else "Transcription complete, but no speaker segments found."

# Mount the 'frontend' directory to serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Mount the 'frontend/icons' directory to serve static files
app.mount("/icons", StaticFiles(directory="frontend/icons"), name="icons")

@app.post("/api/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_name: ModelName = DEFAULT_MODEL,
    language: Optional[Language] = None,
    diarize: bool = False,
    num_speakers: Optional[int] = None,
):
    """
    Transcribes an audio file using the specified WhisperX model.
    """
    # Ensure the requested ASR model is loaded into memory
    load_model_into_cache(model_name.value)
    model = models[model_name.value]
    
    if diarize and not diarization_model:
        return Response(content="Diarization is not available. Please set the HF_TOKEN environment variable and restart the server.", status_code=503)

    # Create a temporary file to store the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        # Load audio and transcribe
        audio = whisperx.load_audio(tmp_file_path)
        lang_code = language.value if language else None
        
        result = model.transcribe(audio, batch_size=16, language=lang_code)

        if diarize:
            # 1. Align transcription
            detected_lang = result["language"]
            if detected_lang not in alignment_models:
                print(f"Loading alignment model for language: {detected_lang}")
                model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
                alignment_models[detected_lang] = (model_a, metadata)
                print("Alignment model loaded.")
            
            model_a, metadata = alignment_models[detected_lang]
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # 2. Diarize and assign speakers
            print("Performing diarization...")
            diarize_segments = diarization_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("Diarization complete.")

        # Format the output and return as plain text
        formatted_text = format_output(result, diarize)
        return Response(content=formatted_text, media_type="text/plain; charset=utf-8")
        
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)


@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.on_event("startup")
async def startup_event():
    """
    Asynchronously loads all necessary models into memory on application startup.
    This includes the default ASR model and, if configured, the diarization model.
    """
    print("--- Application starting up (via @on_event) ---")
    
    # Load default ASR model
    load_model_into_cache(DEFAULT_MODEL.value)

    # Attempt to load diarization model
    global diarization_model
    print("Attempting to load Diarization model...")
    try:
        # The token is only used for the initial download of a gated model.
        token = os.getenv("HF_TOKEN")
        diarization_model = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
        print("Diarization model loaded successfully.")
    except Exception as e:
        diarization_model = None
        print("\n--- Diarization Model Loading Failed ---")
        print("To enable speaker separation (diarization), a one-time setup is required:")
        print("1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1 and agree to the terms.")
        print("2. Go to https://huggingface.co/settings/tokens to create a 'read' access token.")
        print("3. Set the token as an environment variable: export HF_TOKEN='your_token_here'")
        print("4. Restart the server. The model will be downloaded once and cached for offline use.")
        print(f"(Error details: {e})\n")

# --- Application Startup ---
# This part is deprecated and its logic is already handled by the lifespan manager.
# It will be removed. 