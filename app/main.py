import os
import torch
import traceback
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

@app.websocket("/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Lazy load models on the first connection
    if any(m is None for m in [asr_model, mt_model, tts_acoustic_model]):
        if not load_all_models():
            await websocket.close(code=1011, reason="Models failed to load on server.")
            return

    print("Client connected and models are ready.")
    try:
        while True:
            # 1. Receive audio
            audio_bytes = await websocket.receive_bytes()
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # --- 2. ASR ---
            inputs = asr_processor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(asr_model.device).to(torch.float16)
            with torch.no_grad():
                generated_ids = asr_model.generate(input_features)
            french_text = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"ASR Result: {french_text}")

            # --- 3. MT ---
            mt_tokenizer.src_lang = "fr"
            encoded_fr = mt_tokenizer(french_text, return_tensors="pt").to(mt_model.device)
            with torch.no_grad():
                generated_tokens_mt = mt_model.generate(**encoded_fr, forced_bos_token_id=mt_tokenizer.get_lang_id("bas"))
            basaa_text = mt_tokenizer.batch_decode(generated_tokens_mt, skip_special_tokens=True)[0]
            print(f"MT Result: {basaa_text}")

            # --- 4. TTS ---
            prompt = f"{tts_tokenizer.bos_token or ''}<|voice|>basaa_speaker<|text|>{basaa_text}{tts_tokenizer.eos_token or ''}<|audio|>"
            input_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                generated_tokens_tts = tts_acoustic_model.generate(
                    input_ids, max_new_tokens=4000, temperature=0.6, top_p=0.95,
                    repetition_penalty=1.1, do_sample=True, pad_token_id=tts_tokenizer.pad_token_id,
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
            waveform_to_save = waveform.squeeze(0).cpu()
            sf.write(output_buffer, waveform_to_save, 24000, format="wav")
            wav_bytes = output_buffer.getvalue()
            
            # --- 5. Send audio back ---
            await websocket.send_bytes(wav_bytes)
            print("Sent audio response to client.")
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        traceback.print_exc()
        await websocket.close(code=1011, reason="An error occurred during translation.")
