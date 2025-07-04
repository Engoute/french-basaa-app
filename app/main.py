# ───────────────────────── main.py ──────────────────────────
import os, io, shutil, zipfile, tempfile, traceback
from pathlib import Path

import torch, gdown, numpy as np, soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoModelForSpeechSeq2Seq,
    M2M100ForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)
from snac import SNAC

# ── Configuration ────────────────────────────────────────────
MODELS_DIR      = Path(os.getenv("MODELS_DIR", "/app/models"))
ASR_MODEL_PATH  = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_PATH   = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_PATH  = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URLS = {
    "whisper.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/whisper.zip",
    "m2m100.zip":  "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/m2m100.zip",
    "orpheus.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/orpheus.zip",
}

# ── Globals ─────────────────────────────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ── Helpers ─────────────────────────────────────────────────
def safe_unzip(zip_path: Path, final_dir: Path, url: str):
    """Download (if missing) → unzip → move to final_dir (needs config.json)."""
    if final_dir.joinpath("config.json").exists():
        print(f"{final_dir} already exists. Skip download/unzip.")
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading for {final_dir.name} …")
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        print(f"Unzipping {zip_path.name} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_root)
        cfg = list(tmp_root.rglob("config.json"))
        if not cfg:
            raise RuntimeError("config.json not found in archive")
        model_root = cfg[0].parent
        for item in model_root.iterdir():
            shutil.move(str(item), final_dir)
    zip_path.unlink()
    print(f"Model ready at {final_dir}")

def wav_to_pcm16(wav_bytes: bytes) -> np.ndarray:
    """
    Accept either raw 16-bit little-endian PCM or a whole RIFF/WAVE file.
    Returns int16 numpy array *mono* @ 16 kHz.
    """
    if wav_bytes[:4] == b"RIFF":                        # WAVE header present
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
        if sr != 16_000:
            raise ValueError(f"expected 16 kHz, got {sr}")
        if audio.ndim > 1:                              # stereo → mono
            audio = audio.mean(axis=1).astype(np.int16)
        return audio
    # assume already raw 16-bit PCM
    return np.frombuffer(wav_bytes, np.int16)

def load_all_models():
    """Ensure zips exist → unzip → load all three model stacks."""
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder
    try:
        safe_unzip(MODELS_DIR/"whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
        safe_unzip(MODELS_DIR/"m2m100.zip",  MT_MODEL_PATH,  MODEL_URLS["m2m100.zip"])
        safe_unzip(MODELS_DIR/"orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

        if asr_model is None:
            print("— Loading ASR —")
            asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                ASR_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
            )
            print("✅ ASR loaded")

        if mt_model is None:
            print("— Loading MT —")
            mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
            mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")
            print("✅ MT loaded")

        if tts_acoustic_model is None:
            print("— Loading TTS —")
            acoustic = TTS_MODEL_PATH / "acoustic_model"
            vocoder  = TTS_MODEL_PATH / "vocoder"
            tts_tokenizer = AutoTokenizer.from_pretrained(acoustic, local_files_only=True)
            tts_acoustic_model = AutoModelForCausalLM.from_pretrained(acoustic, torch_dtype="auto").to(DEVICE).eval()
            tts_vocoder = SNAC.from_pretrained(vocoder, local_files_only=True).to(DEVICE).eval()
            print("✅ TTS loaded")

        print("— All models loaded —")
        return True
    except Exception as e:
        print(f"!!! Model load failed: {e}")
        traceback.print_exc()
        return False

# ── FastAPI lifecycle ───────────────────────────────────────
@app.on_event("startup")
async def _startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("App started. Models will load on first request.")

# ── WebSocket endpoint ──────────────────────────────────────
@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()

    if any(m is None for m in [asr_model, mt_model, tts_acoustic_model]):
        if not load_all_models():
            await ws.close(code=1011, reason="Model load failed")
            return

    print("Client connected, models ready")
    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            audio_np = wav_to_pcm16(audio_bytes).astype(np.float32) / 32768.0

            # ASR → French
            feats = asr_processor(audio_np, sampling_rate=16_000, return_tensors="pt").input_features
            feats = feats.to(asr_model.device).half()
            with torch.no_grad():
                fr_ids = asr_model.generate(feats)
            fr_text = asr_processor.batch_decode(fr_ids, skip_special_tokens=True)[0].strip()
            print("ASR:", fr_text)

            # MT → Basaa
            mt_tokenizer.src_lang = "fr"
            enc_fr = mt_tokenizer(fr_text, return_tensors="pt").to(mt_model.device)
            with torch.no_grad():
                bas_ids = mt_model.generate(**enc_fr, forced_bos_token_id=mt_tokenizer.get_lang_id("lg"))
            bas_text = mt_tokenizer.batch_decode(bas_ids, skip_special_tokens=True)[0]
            print("MT :", bas_text)

            # TTS → waveform
            prompt = f"{tts_tokenizer.bos_token or ''}<|voice|>basaa_speaker<|text|>{bas_text}{tts_tokenizer.eos_token or ''}<|audio|>"
            in_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                gen = tts_acoustic_model.generate(
                    in_ids, max_new_tokens=4000, do_sample=True,
                    pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id,
                )

            llm_tok = gen[0][in_ids.shape[-1]:].tolist()
            raw = [t - 128266 - ((i % 7) * 4096) for i, t in enumerate(llm_tok)]
            frames = len(raw) // 7
            raw = raw[: frames * 7]
            codes = [[], [], []]
            for i in range(frames):
                f = raw[i*7:(i+1)*7]
                if any(not (0 <= c < 4096) for c in f):   # skip corrupted frame
                    continue
                codes[0].append(f[0])
                codes[1].extend([f[1], f[4]])
                codes[2].extend([f[2], f[3], f[5], f[6]])

            codes = [torch.tensor(c).unsqueeze(0).to(DEVICE) for c in codes]
            with torch.no_grad():
                wav = tts_vocoder.decode(codes)

            buf = io.BytesIO()
            sf.write(buf, wav.squeeze(0).cpu().numpy(), 24_000,
                     format="WAV", subtype="PCM_16")          # ← explicit
            await ws.send_bytes(buf.getvalue())
            print("Audio sent")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Translation error:", e)
        traceback.print_exc()
        await ws.close(code=1011, reason="Server error")
