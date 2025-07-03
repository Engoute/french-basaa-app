# In app/main.py
import os
import torch
import zipfile
import gdown
import numpy as np
import io
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import the exact classes we discovered
from transformers import WhisperForConditionalGeneration, WhisperProcessor, M2M100ForConditionalGeneration, M2M100Tokenizer
from orpheus.model import Orpheus
from orpheus.inference.inference import InferenceBackend

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
tts_model = None

# --- FastAPI Application ---
app = FastAPI()

def download_and_unzip(zip_name, file_info):
    if not os.path.exists(file_info["extract_path"]):
        zip_path = os.path.join(MODELS_DIR, zip_name)
        print(f"Downloading {zip_name}...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        gdown.download(url=file_info["url"], output=zip_path, quiet=False)
        print(f"Unzipping {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(file_info["extract_path"])
        os.remove(zip_path)
        print(f"Model ready at {file_info['extract_path']}")
    else:
        print(f"Model folder {file_info['extract_path']} already exists. Skipping download.")

@app.on_event("startup")
async def startup_event():
    global asr_model, asr_processor, mt_model, mt_tokenizer, tts_model
    for zip_name, file_info in MODEL_FILES.items():
        download_and_unzip(zip_name, file_info)

    print("--- Loading models into memory... ---")
    
    # 1. Load ASR Model (Whisper)
    print("Loading ASR model...")
    asr_processor = WhisperProcessor.from_pretrained(os.path.join(ASR_MODEL_PATH, 'whisper_fr_inference_v1'))
    asr_model = WhisperForConditionalGeneration.from_pretrained(os.path.join(ASR_MODEL_PATH, 'whisper_fr_inference_v1')).to(DEVICE)
    print("✅ ASR model loaded.")

    # 2. Load MT Model (M2M100)
    print("Loading MT model...")
    mt_tokenizer = M2M100Tokenizer.from_pretrained(MT_MODEL_PATH, src_lang="fr", tgt_lang="bas")
    mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH).to(DEVICE)
    print("✅ MT model loaded.")
    
    # 3. Load TTS Model (Orpheus)
    print("Loading TTS model...")
    orpheus_main_model_path = os.path.join(TTS_MODEL_PATH, "model")
    backend = InferenceBackend(orpheus_main_model_path, torch_compile=False, device=DEVICE)
    tts_model = Orpheus(backend)
    print("✅ TTS model loaded.")

    print("--- Startup complete. All models are ready. ---")

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            # 1. Receive raw audio bytes from the client
            audio_bytes = await websocket.receive_bytes()
            
            # Convert bytes to a float array for Whisper
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 2. ASR: French Audio -> French Text
            input_features = asr_processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = asr_model.generate(input_features.to(DEVICE))
            french_text = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"Transcribed (FR): {french_text}")

            # 3. MT: French Text -> Basaa Text
            encoded_fr = mt_tokenizer(french_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_tokens = mt_model.generate(**encoded_fr, forced_bos_token_id=mt_tokenizer.get_lang_id("bas"))
            basaa_text = mt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            print(f"Translated (Basaa): {basaa_text}")
            
            # 4. TTS: Basaa Text -> Basaa Audio
            with torch.no_grad():
                text_processed = tts_model.process_text(basaa_text)
                result = tts_model.synthesize(text_processed, n_codes=3)
                # Decode the audio codes to a NumPy array
                audio_out_np = tts_model.decode(result['codes'])
            
            # Convert NumPy audio to WAV bytes in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_out_np, tts_model.vocoder.sampling_rate, format='WAV')
            wav_bytes = buffer.getvalue()
            
            # 5. Send synthesized WAV audio back to the client
            await websocket.send_bytes(wav_bytes)
            print("Sent audio response to client.")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason="An internal error occurred.")
