# app/main.py  – models resident in VRAM, pushes JSON + WAV
import os, io, zipfile, shutil, tempfile, traceback, json       # ← added json
from pathlib import Path

import gdown
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoModelForSpeechSeq2Seq,
    M2M100ForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)
from snac import SNAC

# ───────────────────────── CONFIG ───────────────────────────────
MODELS_DIR     = Path(os.getenv("MODELS_DIR", "/app/models"))
ASR_MODEL_PATH = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_PATH  = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_PATH = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URLS = {
    "whisper.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/whisper.zip",
    "m2m100.zip":  "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/m2m100.zip",
    "orpheus.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/orpheus.zip",
}

# ─────────────────── GLOBAL SINGLETONS ──────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ────────────────────── HELPERS ─────────────────────────────────
def safe_unzip(zip_path: Path, final_dir: Path, url: str):
    """Download → unzip → move model root into `final_dir` (idempotent)."""
    if final_dir.joinpath("config.json").exists():
        print(f"{final_dir} already present – skip download")
        return

    final_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)

    with tempfile.TemporaryDirectory() as tmp_root:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_root)
        model_root = next(Path(tmp_root).rglob("config.json")).parent
        for item in model_root.iterdir():
            shutil.move(str(item), final_dir)
    os.remove(zip_path)
    print(f"✓ extracted to {final_dir}")

def wav_to_pcm16(blob: bytes) -> np.ndarray:
    """Return int16 ndarray regardless of WAV/raw input."""
    if blob[:4] == b"RIFF":                               # WAV header
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError(f"WAV must be 16 kHz, got {sr}")
        return data
    return np.frombuffer(blob, np.int16)

def load_models() -> bool:
    """Download (if missing) and load **once** into GPU."""
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder
    try:
        # ── ensure files exist ───────────────────────────────
        safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
        safe_unzip(MODELS_DIR / "m2m100.zip",  MT_MODEL_PATH,  MODEL_URLS["m2m100.zip"])
        safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

        # ── ASR ──────────────────────────────────────────────
        asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        print("✓ ASR ready")

        # ── MT ───────────────────────────────────────────────
        mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
        mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")
        print("✓ MT ready")

        # ── TTS ──────────────────────────────────────────────
        ac = TTS_MODEL_PATH / "acoustic_model"
        vc = TTS_MODEL_PATH / "vocoder"
        tts_tokenizer = AutoTokenizer.from_pretrained(ac, local_files_only=True)
        tts_acoustic_model = AutoModelForCausalLM.from_pretrained(ac, torch_dtype="auto").to(DEVICE).eval()
        tts_vocoder = SNAC.from_pretrained(vc, local_files_only=True).to(DEVICE).eval()
        print("✓ TTS ready")

        return True
    except Exception as e:
        print("❌ Model-load failed:", e)
        traceback.print_exc()
        return False

# ────────────────── FASTAPI LIFECYCLE ───────────────────────────
@app.on_event("startup")
async def _startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("🚀 Pod booting – loading models into GPU…")
    if not load_models():
        raise RuntimeError("Failed to load models on startup")
    print("🌟 All models resident in VRAM – ready!")

# ─────────────────── WEBSOCKET ENDPOINT ─────────────────────────
@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            try:
                audio_blob = await ws.receive_bytes()
            except WebSocketDisconnect:
                break

            pcm16 = wav_to_pcm16(audio_blob).astype(np.float32) / 32768.0

            # ---------- ASR ----------
            inp = asr_processor(pcm16, sampling_rate=16_000, return_tensors="pt")
            feats = inp.input_features.to(asr_model.device).half()
            with torch.no_grad():
                gen_ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            print("FR:", fr)

            # ---------- MT ----------
            mt_tokenizer.src_lang = "fr"
            enc = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            with torch.no_grad():
                bos = mt_tokenizer.get_lang_id("lg")     # Basaa
                trans_ids = mt_model.generate(**enc, forced_bos_token_id=bos)
            bas = mt_tokenizer.batch_decode(trans_ids, skip_special_tokens=True)[0]
            print("LG:", bas)

            # ---------- push texts first (JSON) ----------
            await ws.send_text(json.dumps({"fr": fr, "lg": bas}, ensure_ascii=False))

            # ---------- TTS ----------
            prompt = (
                f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker<|text|>"
                f"{bas}{tts_tokenizer.eos_token}<|audio|>"
            )
            in_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                out = tts_acoustic_model.generate(
                    in_ids,
                    max_new_tokens=4000,
                    do_sample=True,
                    pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id,
                )

            llm_tok = out[0][in_ids.shape[-1]:].tolist()
            raw = [t - 128266 - ((i % 7) * 4096) for i, t in enumerate(llm_tok)]
            raw = raw[: (len(raw) // 7) * 7]            # trim partial frame

            # ---- LONG-dtype fix & empty-frame guard ----
            tracks = [[], [], []]
            for i in range(0, len(raw), 7):
                f = raw[i : i + 7]
                if any(not 0 <= c < 4096 for c in f):
                    continue
                tracks[0].append(f[0])
                tracks[1].extend([f[1], f[4]])
                tracks[2].extend([f[2], f[3], f[5], f[6]])

            if not tracks[0]:           # nothing valid → skip audio
                continue

            codes_for_decode = [
                torch.tensor(track, dtype=torch.long, device=DEVICE).unsqueeze(0)
                for track in tracks
            ]
            # --------------------------------------------

            with torch.no_grad():
                wav = tts_vocoder.decode(codes_for_decode).cpu().numpy().squeeze()

            # ---------- SEND ----------
            buf = io.BytesIO()
            sf.write(buf, wav, 24_000, format="WAV", subtype="PCM_16")
            await ws.send_bytes(buf.getvalue())
            print("🠔  sent audio")

    except Exception as e:
        print("❌ Runtime error:", e)
        traceback.print_exc()
        await ws.close(code=1011, reason="Server exception")
