import os
import torch
import zipfile
import gdown
import numpy as np
import io
import soundfile as sf
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import the exact classes we discovered
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor, M2M100ForConditionalGeneration, M2M100Tokenizer
from snac import SNAC

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")

MODELS_DIR = Path("/app/models")
# Define final paths for the extracted model folders
ASR_MODEL_PATH = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_PATH = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_PATH = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"

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

def safe_unzip(zip_path: Path, final_dir: Path):
    """
    Unzips to a temp folder, finds the true model root (containing config.json),
    and moves its contents to the final directory, ensuring a clean structure.
    """
    if final_dir.exists():
        print(f"Model folder {final_dir} already exists. Skipping download and unzip.")
        return

    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to {zip_path}...")
    gdown.download(url=MODEL_FILES[zip_path.name]["url"], output=str(zip_path), quiet=False, fuzzy=True)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root_path = Path(tmp_root)
        print(f"Unzipping {zip_path.name} to temp directory...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_root_path)

        # Find the directory that actually contains the model's config.json
        candidate_dirs = list(tmp_root_path.rglob("config.json"))
        if not candidate_dirs:
            raise RuntimeError(f"`config.json` not found in {zip_path.name}")

        model_root = candidate_dirs[0].parent
        print(f"Located model root at: {model_root}")

        # Move the *children* of model_root into the final dir
        print(f"Moving contents of {model_root} to {final_dir}...")
        for child in model_root.iterdir():
            shutil.move(str(child), final_dir)

    os.remove(zip_path) # Clean up the zip file
    print(f"Model ready at {final_dir}")


@app.on_event("startup")
async def startup_event():
    global asr_model, asr_processor, mt_model, mt_tokenizer, tts_acoustic_model, tts_tokenizer, tts_vocoder
    # Use the new, safe unzipping function
    for zip_name, file_info in MODEL_FILES.items():
        safe_unzip(MODELS_DIR / zip_name, file_info["extract_path"])

    print("--- Loading models into memory... ---")

    # Load all models using local_files_only=True for safety

    # 1. Load ASR Model (Whisper)
    print("Loading ASR model...")
    asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
    asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_PATH, local_files_only=True).to(DEVICE)
    print("✅ ASR model loaded.")

    # 2. Load MT Model (M2M100)
    print("Loading MT model...")
    mt_tokenizer = M2M100Tokenizer.from_pretrained(MT_MODEL_PATH, src_lang="fr", tgt_lang="bas", local_files_only=True)
    mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, local_files_only=True).to(DEVICE)
    print("✅ MT model loaded.")

    # 3. Load TTS Model (Orpheus)
    print("Loading TTS model...")
    acoustic_model_path = TTS_MODEL_PATH / "acoustic_model"
    vocoder_path = TTS_MODEL_PATH / "vocoder"

    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(acoustic_model_path, torch_dtype="auto", local_files_only=True).to(DEVICE).eval()
    tts_tokenizer = AutoTokenizer.from_pretrained(acoustic_model_path, local_files_only=True)
    tts_vocoder = SNAC.from_pretrained(vocoder_path, local_files_only=True).to(DEVICE).eval()
    print("✅ TTS model loaded.")

    print("--- Startup complete. All models are ready. ---")

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # ... (rest of the function is the same as the last full version) ...

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason="An internal error occurred.")
