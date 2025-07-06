# app/main.py  –  streaming + keep-alive, fully offline
import asyncio, io, json, os, shutil, tempfile, traceback, zipfile
from pathlib import Path
from typing import Optional

import gdown, numpy as np, soundfile as sf, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoModelForCausalLM,
    AutoProcessor, AutoTokenizer, M2M100ForConditionalGeneration,
)
from snac import SNAC

# ────────────────────────── shim for old snac wheels ────────────
try:
    from snac.configuration_snac import SnacConfig          # type: ignore
except ModuleNotFoundError:
    class SnacConfig(dict):
        def __getattr__(self, k): return self[k]

# ────────────────────────── config & paths ──────────────────────
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

# ────────── globals (initialised once) ──────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ───────────────────────── helpers ──────────────────────────────
def safe_unzip(zip_path: Path, final_dir: Path, url: str) -> None:
    """Download `url` → `zip_path` (once) and unzip into `final_dir`."""
    if final_dir.exists() and any(final_dir.iterdir()):
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for itm in Path(tmp).iterdir():
            shutil.move(str(itm), final_dir)
    os.remove(zip_path)

def resolve_model_dir(root: Path) -> Path:
    """Return `root` if it has a config.json, else the first sub-dir that does."""
    if (root / "config.json").exists():
        return root
    try:
        return next(root.rglob("config.json")).parent
    except StopIteration:
        raise FileNotFoundError(f"No config.json inside {root}")

def wav_to_pcm16(blob: bytes) -> np.ndarray:
    """Accept 16-kHz WAV or raw PCM-16 and return np.int16 PCM."""
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError("expect 16 kHz wav")
        return data
    return np.frombuffer(blob, np.int16)

# ─── offline SNAC loader (no Hub, quiet, strict=False) ──────────
def load_snac_local(model_dir: Path, device="cpu") -> SNAC:
    cfg_file = model_dir / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(cfg_file)

    cfg   = SnacConfig(**json.loads(cfg_file.read_text()))
    voc   = SNAC(cfg)

    ckpt: Optional[Path] = getattr(cfg, "checkpoint", None)
    if ckpt:
        ckpt = model_dir / ckpt
    else:
        ckpt = next(model_dir.glob("*.bin*"), None)
    if not ckpt or not ckpt.exists():
        raise FileNotFoundError(f"weights not found under {model_dir}")

    state = torch.load(ckpt, map_location=device)

    import contextlib, io as _io, sys
    with contextlib.redirect_stdout(_io.StringIO()), \
         contextlib.redirect_stderr(_io.StringIO()):
        voc.load_state_dict(state, strict=False)

    print("✅  SNAC vocoder loaded (strict=False, messages suppressed)")
    return voc.to(device).eval()

# ────────── model bootstrap (runs once) ─────────────────────────
def load_models() -> None:
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder

    safe_unzip(MODELS_DIR/"whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
    safe_unzip(MODELS_DIR/"m2m100.zip",  MT_MODEL_PATH,  MODEL_URLS["m2m100.zip"])
    safe_unzip(MODELS_DIR/"orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

    # Whisper – ASR
    asr_dir       = resolve_model_dir(ASR_MODEL_PATH)
    asr_processor = AutoProcessor.from_pretrained(asr_dir, local_files_only=True)
    asr_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    # M2M-100 – MT
    mt_dir       = resolve_model_dir(MT_MODEL_PATH)
    mt_tokenizer = AutoTokenizer.from_pretrained(mt_dir, local_files_only=True)
    mt_model     = M2M100ForConditionalGeneration.from_pretrained(mt_dir, device_map="auto")

    # Orpheus – TTS
    ac_root = resolve_model_dir(TTS_MODEL_PATH / "acoustic_model")
    vc_root = resolve_model_dir(TTS_MODEL_PATH / "vocoder")

    tts_tokenizer      = AutoTokenizer.from_pretrained(ac_root, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(ac_root, torch_dtype="auto"
                           ).to(DEVICE).eval()
    tts_vocoder        = load_snac_local(vc_root, DEVICE)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ────────── startup hook ────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    print("🌟 models pinned in VRAM, server ready")

# ────────── streaming websocket ─────────────────────────────────
FRAME_MS          = 100
TRIGGER_DURATION  = 0.9
SILENCE_TIMEOUT   = 0.30
PING_INTERVAL     = 15

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_buffer: list[np.ndarray] = []
    last_rx = asyncio.get_event_loop().time()

    async def ping_loop():
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send_bytes(b"\0")
            except Exception:
                break
    ping_task = asyncio.create_task(ping_loop())

    try:
        while True:
            try:
                blob = await ws.receive_bytes()
            except WebSocketDisconnect:
                break

            now = asyncio.get_event_loop().time()
            if len(blob) <= 1:   # keep-alive ping
                continue

            pcm_buffer.append(wav_to_pcm16(blob))
            last_rx = now

            # enough speech?
            if sum(len(c) for c in pcm_buffer) < TRIGGER_DURATION * 16_000:
                continue

            await asyncio.sleep(SILENCE_TIMEOUT)
            if asyncio.get_event_loop().time() - last_rx <= SILENCE_TIMEOUT:
                continue

            # ─── ASR ──────────────────────────────────────────
            pcm16 = np.concatenate(pcm_buffer).astype(np.float32) / 32768.0
            pcm_buffer.clear()

            feats = asr_processor(pcm16, sampling_rate=16_000,
                                  return_tensors="pt").input_features
            feats = feats.to(asr_model.device).half()
            with torch.no_grad():
                gen_ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            # ─── MT ───────────────────────────────────────────
            mt_tokenizer.src_lang = "fr"
            enc = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            bos = mt_tokenizer.get_lang_id("lg")
            with torch.no_grad():
                trans_ids = mt_model.generate(**enc, forced_bos_token_id=bos)
            bas = mt_tokenizer.batch_decode(trans_ids, skip_special_tokens=True)[0]

            await ws.send_text(json.dumps({"fr": fr, "lg": bas}))

            # ─── TTS ─────────────────────────────────────────
            prompt = (f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker<|text|>"
                      f"{bas}{tts_tokenizer.eos_token}<|audio|>")
            in_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                out = tts_acoustic_model.generate(
                    in_ids, max_new_tokens=4000, pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id)

            tail = out[0][in_ids.shape[-1]:].tolist()
            raw  = [t - 128_266 - ((i % 7) * 4096) for i, t in enumerate(tail)]
            raw  = raw[: (len(raw) // 7) * 7]

            tracks = [[], [], []]
            for i in range(0, len(raw), 7):
                f = raw[i:i+7]
                if any(not 0 <= c < 4096 for c in f):
                    continue
                tracks[0].append(f[0])
                tracks[1].extend([f[1], f[4]])
                tracks[2].extend([f[2], f[3], f[5], f[6]])

            if tracks[0]:
                codes = [torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(0)
                         for t in tracks]
                with torch.no_grad():
                    wav = tts_vocoder.decode(codes).cpu().numpy().squeeze()

                buf = io.BytesIO()
                sf.write(buf, wav, 24_000, format="WAV", subtype="PCM_16")
                await ws.send_bytes(buf.getvalue())
    except Exception as e:
        print("❌ runtime:", e); traceback.print_exc()
        await ws.close(code=1011, reason="server error")
    finally:
        ping_task.cancel()
