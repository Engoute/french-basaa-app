import os
import torch
import zipfile
import gdown
import numpy as np
import io
import soundfile as sf
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import the exact classes we discovered
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor, M2M100ForConditionalGeneration, M2M100Tokenizer
from snac import SNAC

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")

MODELS_DIR = "/app/models"
# Define final paths for the extracted model folders
ASR_MODEL_PATH = os.path.join(MODELS_DIR, "whisper_fr_inference_v1")
MT_MODEL_PATH = os.path.join(MODELS_DIR, "m2m100_basaa_inference_v1")
TTS_MODEL_PATH = os.path.join(MODELS_DIR, "orpheus_basaa_bundle_16bit_final")

MODEL_FILES = {
    "whisper_fr_inference_v1.zip": { "url": "https://drive.google.com/file/d/1reOzKsylgFqPVaWZcSG4knPVNrJW-Hca/view?usp=sharing", "extract_path": ASR_MODEL_PATH },
    "m2m100_basaa_inference_v1.zip": { "url": "https://drive.google.com/file/d/15iOWnGQnaVTB5mTKRNRHE2CKAKca3dxN/view?usp=sharing", "extract_path": MT_MODEL_PATH },
    "orpheus_basaa_bundle_16bit_final.zip": { "url": "https://drive.google.com/file/d/1MzgKfF7tvVr8QSTrwQ0bvb95aqfjj4S6/view?usp=sharing", "extract_path": TTS_MODEL_PATH }
}

# --- Model Placeholders ---
asr_model, asr_processor = None, None
mt_model, mt_tokenizer = None, None
tts_acoustic_model, tts_tokenizer, tts_vocoder = None, None, None

# --- FastAPI Application ---
app = FastAPI()

def download_and_unzip(zip_name, file_info):
    """
    Downloads and unzips model files, handling nested folders to create a clean path.
    """
    if not os.path.exists(file_info["extract_path"]):
        zip_path = os.path.join(MODELS_DIR, zip_name)
        temp_extract_dir = os.path.join(MODELS_DIR, "temp_unzip")

        print(f"Downloading {zip_name}...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        gdown.download(url=file_info["url"], output=zip_path, quiet=False, fuzzy=True)

        print(f"Unzipping to temp directory...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # Find the actual content directory (might be nested)
        unzipped_root_dir = temp_extract_dir
        # If the zip created a single folder, go inside it
        items = os.listdir(temp_extract_dir)
        if len(items) == 1 and os.path.isdir(os.path.join(temp_extract_dir, items[0])):
            unzipped_root_dir = os.path.join(temp_extract_dir, items[0])

        # Move the content to the final clean path
        print(f"Moving model to final path: {file_info['extract_path']}")
        shutil.move(unzipped_root_dir, file_info["extract_path"])

        # Clean up
        os.remove(zip_path)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

        print(f"Model ready at {file_info['extract_path']}")
    else:
        print(f"Model folder {file_info['extract_path']} already exists. Skipping download.")

@app.on_event("startup")
async def startup_event():
    global asr_model, asr_processor, mt_model, mt_tokenizer, tts_acoustic_model, tts_tokenizer, tts_vocoder
    for zip_name, file_info in MODEL_FILES.items():
        download_and_unzip(zip_name, file_info)

    print("--- Loading models into memory... ---")

    # 1. Load ASR Model (Whisper) - Path is now clean
    print("Loading ASR model...")
    asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
    asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_PATH).to(DEVICE)
    print("✅ ASR model loaded.")

    # 2. Load MT Model (M2M100) - Path is now clean
    print("Loading MT model...")
    mt_tokenizer = M2M100Tokenizer.from_pretrained(MT_MODEL_PATH, src_lang="fr", tgt_lang="bas")
    mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH).to(DEVICE)
    print("✅ MT model loaded.")

    # 3. Load TTS Model (Orpheus) - Path is now clean
    print("Loading TTS model...")
    acoustic_model_path = os.path.join(TTS_MODEL_PATH, "acoustic_model")
    vocoder_path = os.path.join(TTS_MODEL_PATH, "vocoder")

    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(acoustic_model_path, torch_dtype="auto").to(DEVICE).eval()
    tts_tokenizer = AutoTokenizer.from_pretrained(acoustic_model_path)
    tts_vocoder = SNAC.from_pretrained(vocoder_path).to(DEVICE).eval()
    print("✅ TTS model loaded.")

    print("--- Startup complete. All models are ready. ---")

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            # ... (rest of the function is the same) ...
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason="An internal error occurred.")
