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
            shutil.move(str(child), final_dir)

    os.remove(zip_path)
    print(f"Model ready at {final_dir}")

def load_all_models():
    global asr_model, asr_processor, mt_model, mt_tokenizer, tts_acoustic_model, tts_tokenizer, tts_vocoder
    try:
        # Download and unzip all models first
        safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
        safe_unzip(MODELS_DIR / "m2m100.zip", MT_MODEL_PATH, MODEL_URLS["m2m100.zip"])
        safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

        # Then load them
        if asr_model is None:
            print("--- LAZY LOADING: ASR Model ---")
            asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print("✅ ASR model loaded.")

        if mt_model is None:
            print("--- LAZY LOADING: MT Model ---")
            mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
            mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")
            print("✅ MT model loaded.")

        if tts_acoustic_model is None:
            print("--- LAZY LOADING: TTS Models ---")
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Application started. Models will be loaded on first request.")

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if any(m is None for m in [asr_model, mt_model, tts_acoustic_model]):
        if not load_all_models():
            await websocket.close(code=1011, reason="Models failed to load on server.")
            return

    print("Client connected and models are ready.")
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            inputs = asr_processor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(asr_model.device).to(torch.float16)
            with torch.no_grad():
                generated_ids = asr_model.generate(input_features)
            french_text = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"ASR Result: {french_text}")

            mt_tokenizer.src_lang = "fr"
            encoded_fr = mt_tokenizer(french_text, return_tensors="pt").to(mt_model.device)
            with torch.no_grad():
                generated_tokens_mt = mt_model.generate(**encoded_fr, forced_bos_token_id=mt_tokenizer.get_lang_id("bas"))
            basaa_text = mt_tokenizer.batch_decode(generated_tokens_mt, skip_special_tokens=True)[0]
            print(f"MT Result: {basaa_text}")

            prompt = f"{tts_tokenizer.bos_token or ''}<|voice|>basaa_speaker<|text|>{basaa_text}{tts_tokenizer.eos_token or ''}<|audio|>"
            input_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                generated_tokens_tts = tts_acoustic_model.generate(
                    input_ids, max_new_tokens=4000, do_sample=True, pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id
                )
            
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
                waveform = tts_vocoder.decode(codes_for_decode)
            
            output_buffer = io.BytesIO()
            sf.write(output_buffer, waveform.squeeze(0).cpu().numpy(), 24000, format="wav")
            wav_bytes = output_buffer.getvalue()
            
            await websocket.send_bytes(wav_bytes)
            print("Sent audio response to client.")
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        traceback.print_exc()
        await websocket.close(code=1011, reason="An error occurred during translation.")
