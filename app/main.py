# app/main.py  – GPU-resident models + keep-alive & chunk streaming
import asyncio, io, json, os, shutil, tempfile, traceback, zipfile
from pathlib import Path

import gdown, numpy as np, soundfile as sf, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (AutoModelForCausalLM, AutoModelForSpeechSeq2Seq,
                          AutoProcessor, AutoTokenizer,
                          M2M100ForConditionalGeneration)
from snac import SNAC

# ────────────────  CONFIG  ──────────────────────────────────────
MODELS_DIR     = Path(os.getenv("MODELS_DIR", "/app/models"))
ASR_MODEL_PATH = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_PATH  = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_PATH = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

URLS = {
    "whisper.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/whisper.zip",
    "m2m100.zip":  "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/m2m100.zip",
    "orpheus.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/orpheus.zip",
}

# ────────────────  GLOBALS  ─────────────────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None
app = FastAPI()

# ────────────────  HELPERS  ─────────────────────────────────────
def _safe_unzip(z: Path, dst: Path, url: str) -> None:
    if dst.joinpath("config.json").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(z), quiet=False, fuzzy=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        root = next(Path(tmp).rglob("config.json")).parent
        for f in root.iterdir():
            shutil.move(str(f), dst)
    z.unlink()

def _wav_or_raw(blob: bytes) -> np.ndarray:
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError("expected 16 kHz WAV")
        return data
    return np.frombuffer(blob, np.int16)

def _load_models() -> None:
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder

    _safe_unzip(MODELS_DIR/"whisper.zip", ASR_MODEL_PATH, URLS["whisper.zip"])
    _safe_unzip(MODELS_DIR/"m2m100.zip",  MT_MODEL_PATH,  URLS["m2m100.zip"])
    _safe_unzip(MODELS_DIR/"orpheus.zip", TTS_MODEL_PATH, URLS["orpheus.zip"])

    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_PATH, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map="auto")

    mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
    mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")

    ac, vc = TTS_MODEL_PATH / "acoustic_model", TTS_MODEL_PATH / "vocoder"
    tts_tokenizer      = AutoTokenizer.from_pretrained(ac, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(ac, torch_dtype="auto").to(DEVICE).eval()
    tts_vocoder        = SNAC.from_pretrained(vc, local_files_only=True).to(DEVICE).eval()

# ────────────────  STARTUP  ─────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    MODELS_DIR.mkdir(Parents=True, exist_ok=True)
    print("🔄  loading models …")
    _load_models()
    print("✅  GPU ready")

# ────────────────  WEBSOCKET  ───────────────────────────────────
PING_EVERY = 20          # seconds

async def _ping(ws: WebSocket) -> None:
    """background ping-pong keep-alive"""
    try:
        while True:
            await asyncio.sleep(PING_EVERY)
            await ws.send_json({"ping": True})
    except Exception:
        pass    # socket closed – ping task ends

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()

    ping_task = asyncio.create_task(_ping(ws))
    try:
        while True:
            try:
                blob = await ws.receive_bytes()
            except WebSocketDisconnect:
                break

            # ── ASR ───────────────────────────────────────────────
            pcm = _wav_or_raw(blob).astype(np.float32) / 32768.0
            feats = asr_processor(pcm, sampling_rate=16_000,
                                  return_tensors="pt").input_features
            feats = feats.to(asr_model.device).half()
            with torch.no_grad():
                ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

            # ── MT ────────────────────────────────────────────────
            mt_tokenizer.src_lang = "fr"
            enc = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            bos = mt_tokenizer.get_lang_id("lg")
            with torch.no_grad():
                out = mt_model.generate(**enc, forced_bos_token_id=bos)
            lg = mt_tokenizer.batch_decode(out, skip_special_tokens=True)[0]

            # ── push JSON immediately ─────────────────────────────
            await ws.send_text(json.dumps({"fr": fr, "lg": lg}))

            # ── TTS ───────────────────────────────────────────────
            prompt = (f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker<|text|>"
                      f"{lg}{tts_tokenizer.eos_token}<|audio|>")
            in_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                gen = tts_acoustic_model.generate(
                    in_ids, max_new_tokens=4096, do_sample=True,
                    pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id)

            toks = gen[0][in_ids.size(-1):].tolist()
            raw  = [t - 128_266 - ((i % 7) * 4096) for i, t in enumerate(toks)]
            raw  = raw[: (len(raw)//7)*7]
            code = [[], [], []]
            for i in range(0, len(raw), 7):
                f = raw[i:i+7]
                if any(not 0 <= c < 4096 for c in f):
                    continue
                code[0].append(f[0])
                code[1].extend([f[1], f[4]])
                code[2].extend([f[2], f[3], f[5], f[6]])
            if not code[0]:
                continue

            codes = [torch.tensor(c, dtype=torch.long, device=DEVICE).unsqueeze(0)
                     for c in code]
            with torch.no_grad():
                wav = tts_vocoder.decode(codes).cpu().numpy().squeeze()

            buf = io.BytesIO()
            sf.write(buf, wav, 24_000, format="WAV", subtype="PCM_16")
            await ws.send_bytes(buf.getvalue())

    except Exception as exc:
        print("❌  server error:", exc)
        traceback.print_exc()
    finally:
        ping_task.cancel()
        await ws.close()
