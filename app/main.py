import os
import torch
import zipfile
import gdown
import numpy as np
import io
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import the necessary Hugging Face classes
from transformers import AutoModelForSpeechSeq2Seq, M2M100ForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from snac import SNAC

# --- Configuration ---
ASR_MODEL_PATH = "/app/models/whisper_fr_inference_v1"
MT_MODEL_PATH = "/app/models/m2m100_basaa_inference_v1"
TTS_ACOUSTIC_PATH = "/app/models/orpheus_basaa_bundle_16bit_final/acoustic_model"
TTS_VOCODER_PATH = "/app/models/orpheus_basaa_bundle_16bit_final/vocoder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Model Placeholders ---
asr_model, asr_processor = None, None
mt_model, mt_tokenizer = None, None
tts_acoustic_model, tts_tokenizer, tts_vocoder = None, None, None

app = FastAPI()

def load_all_models():
    """
    Loads all models into memory. This function is called only when the first user connects.
    """
    global asr_model, asr_processor, mt_model, mt_tokenizer, tts_acoustic_model, tts_tokenizer, tts_vocoder
    
    try:
        if asr_model is None:
            print("--- LAZY LOADING: ASR Model ---")
            asr_inner_path = os.path.join(ASR_MODEL_PATH, "whisper_large_v3_fr_int8")
            asr_processor = AutoProcessor.from_pretrained(asr_inner_path, local_files_only=True)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_inner_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print("✅ ASR model loaded.")

        if mt_model is None:
            print("--- LAZY LOADING: MT Model ---")
            mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
            mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")
            print("✅ MT model loaded.")

        if tts_acoustic_model is None:
            print("--- LAZY LOADING: TTS Models ---")
            tts_tokenizer = AutoTokenizer.from_pretrained(TTS_ACOUSTIC_PATH, local_files_only=True)
            tts_acoustic_model = AutoModelForCausalLM.from_pretrained(TTS_ACOUSTIC_PATH, torch_dtype="auto").to(DEVICE).eval()
            tts_vocoder = SNAC.from_pretrained(TTS_VOCODER_PATH, local_files_only=True).to(DEVICE).eval()
            print("✅ TTS model loaded.")
            
        print("--- All models loaded successfully. ---")
        return True

    except Exception as e:
        print(f"!!! A MODEL FAILED TO LOAD: {e} !!!")
        traceback.print_exc()
        return False

# NOTE: The @app.on_event("startup") is removed entirely.

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Lazy load models on first connection
    if any(m is None for m in [asr_model, mt_model, tts_acoustic_model]):
        if not load_all_models():
            await websocket.close(code=1011, reason="Models failed to load on server.")
            return

    print("Client connected and models are ready.")
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            # ... (The rest of the websocket processing logic is the same) ...
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        await websocket.close(code=1011, reason="An error occurred during translation.")
