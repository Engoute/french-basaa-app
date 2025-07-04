import os
import torch
import traceback
import zipfile
import gdown
import numpy as np
import io
import soundfile as sf
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import the necessary Hugging Face classes
from transformers import AutoModelForSpeechSeq2Seq, M2M100ForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from snac import SNAC

# --- Configuration ---
MODELS_DIR = Path("/app/models")
ASR_MODEL_PATH = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_PATH = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_PATH = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URLS = {
    "whisper.zip": "https://drive.google.com/file/d/1reOzKsylgFqPVaWZcSG4knPVNrJW-Hca/view?usp=sharing",
    "m2m100.zip": "https://drive.google.com/file/d/15iOWnGQnaVTB5mTKRNRHE2CKAKca3dxN/view?usp=sharing",
    "orpheus.zip": "https://drive.google.com/file/d/1MzgKfF7tvVr8QSTrwQ0bvb95aqfjj4S6/view?usp=sharing"
}

# --- Global Model Placeholders ---
asr_model, asr_processor, mt_model, mt_tokenizer, tts_acoustic_model, tts_tokenizer, tts_vocoder = [None] * 7

app = FastAPI()

def safe_unzip(zip_path: Path, final_dir: Path, url: str):
    """
    Unzips to a temp folder, finds the true model root (containing config.json),
    and moves its contents to the final directory, ensuring a clean structure.
    """
    if final_dir.joinpath("config.json").exists():
        print(f"Model folder {final_dir} already seems to exist. Skipping download and unzip.")
        return
        
    final_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading for {final_dir.name}...")
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root_path = Path(tmp_root)
        print(f"Unzipping {zip_path.name} to temp directory...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_root_path)

        candidate_dirs = list(tmp_root_path.rglob("config.json"))
        if not candidate_dirs:
            raise RuntimeError(f"`config.json` not found in {zip_path.name}")
        
        model_root = candidate_dirs[0].parent
        print(f"Located model root at: {model_root}")

        print(f"Moving contents of {model_root} to {final_dir}...")
        for child in model_root.iterdir():
            dest = final_dir / child.name
            # In case of re-running a failed unzip, clear destination first
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(child), final_dir)

    os.remove(zip_path)
    print(f"Model ready at {final_dir}")

def load_all_models():
    global asr_model, asr_processor, mt_model, mt_tokenizer, tts_acoustic_model, tts_tokenizer, tts_vocoder
    try:
        if asr_model is None:
            print("--- LAZY LOADING: ASR Model ---")
            safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
            asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print("✅ ASR model loaded.")

        if mt_model is None:
            print("--- LAZY LOADING: MT Model ---")
            safe_unzip(MODELS_DIR / "m2m100.zip", MT_MODEL_PATH, MODEL_URLS["m2m100.zip"])
            mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
            mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")
            print("✅ MT model loaded.")

        if tts_acoustic_model is None:
            print("--- LAZY LOADING: TTS Models ---")
            safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])
            acoustic_path = TTS_MODEL_PATH / "acoustic_model"
            vocoder_path = TTS_MODEL_PATH / "vocoder"
            tts_tokenizer = AutoTokenizer.from_pretrained(acoustic_path, local_files_only=True)
            tts_acoustic_model = AutoModelForCausalLM.from_pretrained(acoustic_path, torch_dtype="auto").to(DEVICE).eval()
            tts_vocoder = SNAC.from_pretrained(vocoder_path, local_files_only=True).to(DEVICE).eval()
            print("✅ TTS model loaded.")
            
        print("--- All models loaded successfully. ---")
        return True
    except Exception as e:
        print(f"!!! A MODEL FAILED TO LOAD: {e} !!!")
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    # We now use lazy loading, so we don't load models on startup.
    # We just ensure the models directory exists.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Application started. Models will be loaded on first request.")

# The rest of your main.py file...
@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    # (The websocket logic remains the same)
    pass
