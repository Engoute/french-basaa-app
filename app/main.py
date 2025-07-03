import os
import torch
import zipfile
import gdown
import numpy as np
import io
import soundfile as sf
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import Orpheus directly from the app package
from app.orpheus.model import Orpheus
from app.orpheus.inference.inference import InferenceBackend

# Import other transformers classes
from transformers import WhisperForConditionalGeneration, WhisperProcessor, M2M100ForConditionalGeneration, M2M100Tokenizer

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "/app/models"
ASR_MODEL_PATH = os.path.join(MODELS_DIR, "whisper_fr_inference_v1")
MT_MODEL_PATH = os.path.join(MODELS_DIR, "m2m100_basaa_inference_v1")
TTS_WEIGHTS_PATH = os.path.join(MODELS_DIR, "orpheus_basaa_bundle_16bit_final")

# --- Model Placeholders ---
asr_model, asr_processor, mt_model, mt_tokenizer, tts_model = [None] * 5

def download_model(url, zip_name, final_path):
    """Downloads and unzips a single model if it doesn't exist."""
    if not os.path.exists(os.path.join(final_path, "config.json")):
        zip_path = os.path.join(MODELS_DIR, zip_name)
        print(f"Downloading {zip_name}...")
        gdown.download(url=url, output=zip_path, quiet=False, fuzzy=True)
        
        print(f"Unzipping {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(final_path)
        os.remove(zip_path)
        print(f"Model ready at {final_path}")
    else:
        print(f"Model folder {final_path} already exists. Skipping download.")

@app.on_event("startup")
async def startup_event():
    """On startup, download any missing models and then load everything into the GPU."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Download models
    download_model("https://drive.google.com/file/d/1reOzKsylgFqPVaWZcSG4knPVNrJW-Hca/view?usp=sharing", "whisper.zip", ASR_MODEL_PATH)
    download_model("https://drive.google.com/file/d/15iOWnGQnaVTB5mTKRNRHE2CKAKca3dxN/view?usp=sharing", "m2m100.zip", MT_MODEL_PATH)
    download_model("https://drive.google.com/file/d/1MzgKfF7tvVr8QSTrwQ0bvb95aqfjj4S6/view?usp=sharing", "orpheus.zip", TTS_WEIGHTS_PATH)

    # Load models into memory
    print("--- Loading models into memory... ---")
    
    global asr_processor, asr_model, mt_tokenizer, mt_model, tts_model
    
    # Load ASR Model
    asr_inner_path = os.path.join(ASR_MODEL_PATH, "whisper_large_v3_fr_int8")
    print(f"Loading ASR model from {asr_inner_path}...")
    asr_processor = WhisperProcessor.from_pretrained(asr_inner_path, local_files_only=True)
    asr_model = WhisperForConditionalGeneration.from_pretrained(asr_inner_path, local_files_only=True).to(DEVICE)
    print("✅ ASR model loaded.")

    # Load MT Model
    print(f"Loading MT model from {MT_MODEL_PATH}...")
    mt_tokenizer = M2M100Tokenizer.from_pretrained(MT_MODEL_PATH, src_lang="fr", tgt_lang="bas", local_files_only=True)
    mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, local_files_only=True).to(DEVICE)
    print("✅ MT model loaded.")
    
    # Load TTS Model
    print(f"Loading TTS model from {TTS_WEIGHTS_PATH}...")
    acoustic_model_path = os.path.join(TTS_WEIGHTS_PATH, "acoustic_model")
    vocoder_path = os.path.join(TTS_WEIGHTS_PATH, "vocoder")
    backend = InferenceBackend(acoustic_model_path, torch_compile=False, device=DEVICE)
    tts_model = Orpheus(backend, vocoder_path=vocoder_path)
    print("✅ TTS model loaded.")

    print("--- Startup complete. All models are ready. ---")

# --- FastAPI Application ---
app = FastAPI()

# --- WebSocket Endpoint ---
@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Stage 1: ASR
            input_features = asr_processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = asr_model.generate(input_features.to(DEVICE))
            french_text = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"Transcribed (FR): {french_text}")

            # Stage 2: MT
            encoded_fr = mt_tokenizer(french_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_tokens_mt = mt_model.generate(**encoded_fr, forced_bos_token_id=mt_tokenizer.get_lang_id("bas"))
            basaa_text = mt_tokenizer.batch_decode(generated_tokens_mt, skip_special_tokens=True)[0]
            print(f"Translated (Basaa): {basaa_text}")
            
            # Stage 3: TTS
            prompt = f"{tts_model.tokenizer.bos_token or ''}<|voice|>basaa_speaker<|text|>{basaa_text}{tts_model.tokenizer.eos_token or ''}<|audio|>"
            input_ids = tts_model.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                generated_tokens_tts = tts_model.model.generate(input_ids, max_new_tokens=4000, do_sample=True, pad_token_id=tts_model.tokenizer.pad_token_id, eos_token_id=tts_model.tokenizer.eos_token_id)
            
            llm_audio_token_ids = generated_tokens_tts[0][input_ids.shape[-1]:].tolist()
            raw_codes = [tok - 128266 - ((i % 7) * 4096) for i, tok in enumerate(llm_audio_token_ids)]
            num_frames = len(raw_codes) // 7
            raw_codes = raw_codes[:num_frames * 7]
            codes = [[], [], []]
            for i in range(num_frames):
                frame_start = i * 7
                frame = raw_codes[frame_start:frame_start+7]
                if any(not (0 <= code < 4096) for code in frame): continue
                codes[0].append(frame[0]); codes[1].extend([frame[1], frame[4]]); codes[2].extend([frame[2], frame[3], frame[5], frame[6]])
            codes_for_decode = [torch.tensor(c, dtype=torch.long).unsqueeze(0).to(DEVICE) for c in codes]

            with torch.no_grad(): 
                waveform = tts_model.vocoder.decode(codes_for_decode)
            
            buffer = io.BytesIO()
            sf.write(buffer, waveform.squeeze(0).cpu().numpy(), 24000, format='WAV')
            wav_bytes = buffer.getvalue()
            
            await websocket.send_bytes(wav_bytes)
            print("Sent audio response to client.")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason="An internal error occurred.")
