# app/main.py  –  latency‑optimised  ●  2025‑07‑10
import asyncio, io, json, os, shutil, tempfile, zipfile, time
from pathlib import Path
from typing import List

import gdown, numpy as np, soundfile as sf, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed          # for safe send
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoModelForCausalLM,
    AutoProcessor, AutoTokenizer, M2M100ForConditionalGeneration,
)

# ═══ SNAC loader (pinned 48‑ch) ═══════════════════════════════
from snac import SNAC
try:
    from snac.configuration_snac import SnacConfig          # legacy wheel
except ModuleNotFoundError:
    class SnacConfig(dict):                                 # type: ignore
        def __getattr__(self, k): return self[k]

# ─── paths ────────────────────────────────────────────────────
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

# ─── globals ──────────────────────────────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ─── helpers ──────────────────────────────────────────────────
def safe_unzip(zip_path: Path, dst_dir: Path, url: str) -> None:
    if dst_dir.exists() and any(dst_dir.iterdir()):
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for itm in Path(tmp).iterdir():
            shutil.move(str(itm), dst_dir)
    os.remove(zip_path)

def resolve_dir(root: Path) -> Path:
    return root if (root / "config.json").exists() else next(root.rglob("config.json")).parent

def wav_to_int16(blob: bytes) -> np.ndarray:
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError("expected 16 kHz WAV")
        return data
    return np.frombuffer(blob, np.int16)

def load_snac(model_dir: Path, device="cpu") -> SNAC:
    cfg = SnacConfig(**json.loads((model_dir / "config.json").read_text()))
    voc = SNAC(**cfg).to(device)
    raw = torch.load(model_dir / "pytorch_model.bin", map_location=device)
    good = {k: v for k, v in raw.items()
            if k in voc.state_dict() and v.shape == voc.state_dict()[k].shape}
    voc.load_state_dict(good, strict=False)
    voc.eval()
    print("✅  SNAC loaded")
    return voc

# ─── optimisation helpers ────────────────────────────────────
def whisper_chunks(pcm_f32: np.ndarray, sr=16_000,
                   win_s=25, hop_s=23):
    step = hop_s * sr
    win  =  win_s * sr
    for off in range(0, len(pcm_f32), step):
        yield pcm_f32[off : off + win]

def fast_generate_whisper(feats):
    return asr_model.generate(
        feats,
        do_sample=False,
        num_beams=1,
        max_length=448,
    )

def tts_token_budget(text: str, ceiling=9000):
    est = int(len(text.split()) * 9 * 1.4)
    return min(max(256, est), ceiling)

# ─── model bootstrap ─────────────────────────────────────────
def _try_compile(model, name: str):
    try:
        return torch.compile(model, mode="reduce-overhead", fullgraph=True)
    except Exception as ex:
        print(f"⚠️  torch.compile failed for {name}: {ex}  (continuing without)")
        return model

def load_models() -> None:
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder

    safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
    safe_unzip(MODELS_DIR / "m2m100.zip",  MT_MODEL_PATH,  MODEL_URLS["m2m100.zip"])
    safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

    # Whisper
    asr_dir       = resolve_dir(ASR_MODEL_PATH)
    asr_processor = AutoProcessor.from_pretrained(asr_dir, local_files_only=True)
    asr_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_dir, torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True
    ).eval()
    asr_model     = _try_compile(asr_model, "Whisper")

    # MT
    mt_dir       = resolve_dir(MT_MODEL_PATH)
    mt_tokenizer = AutoTokenizer.from_pretrained(mt_dir, local_files_only=True)
    mt_model     = M2M100ForConditionalGeneration.from_pretrained(
        mt_dir, torch_dtype=torch.float16, device_map={"": 0}
    ).eval()
    mt_model     = _try_compile(mt_model, "M2M‑100")

    # TTS acoustic + vocoder
    ac_root = resolve_dir(TTS_MODEL_PATH / "acoustic_model")
    vc_root = resolve_dir(TTS_MODEL_PATH / "vocoder")

    tts_tokenizer      = AutoTokenizer.from_pretrained(ac_root, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(
        ac_root, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tts_acoustic_model = _try_compile(tts_acoustic_model, "Orpheus‑LM")

    tts_vocoder        = load_snac(vc_root, DEVICE)
    tts_vocoder.decode = _try_compile(tts_vocoder.decode, "SNAC‑decode")

    # Global GPU knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.sdp_kernel(enable_flash=True,
                                   enable_math=False,
                                   enable_mem_efficient=False)

@app.on_event("startup")
async def _startup() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    print("🌟 models in VRAM, server ready")

# ══════════════════════════════════════════════════════════════
#  Web‑Socket 1 :  /translate
# ══════════════════════════════════════════════════════════════
PING_INTERVAL = 15  # seconds

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_chunks: List[np.ndarray] = []

    async def pinger():
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try: await ws.send_bytes(b"\0")
            except Exception: break
    asyncio.create_task(pinger())

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if "text" in msg and msg["text"]:
                if msg["text"] == "DONE": break
                continue
            if msg.get("bytes") is None: continue
            pcm_chunks.append(wav_to_int16(msg["bytes"]))
    except WebSocketDisconnect:
        return

    # ───── ASR ─────
    if not pcm_chunks:
        await ws.close(code=4000, reason="no audio")
        return
    t0 = time.perf_counter()

    pcm_f32 = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
    # One pass is faster if <25 s
    if len(pcm_f32) < 25 * 16_000:
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            feats = asr_processor(pcm_f32, sampling_rate=16_000,
                                  return_tensors="pt").input_features.to(DEVICE)
            ids = fast_generate_whisper(feats)
        fr = asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    else:
        parts = []
        for ch in whisper_chunks(pcm_f32):
            feats = asr_processor(ch, sampling_rate=16_000,
                                  return_tensors="pt").input_features.to(DEVICE)
            with torch.inference_mode():
                ids = fast_generate_whisper(feats)
            parts.append(asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        fr = " ".join(parts).strip()

    # ───── MT ─────
    mt_tokenizer.src_lang = "fr"
    enc = mt_tokenizer(fr, return_tensors="pt").to(DEVICE)
    bos = mt_tokenizer.get_lang_id("lg")
    with torch.inference_mode():
        out_ids = mt_model.generate(
            **enc, forced_bos_token_id=bos, max_new_tokens=512, num_beams=2)
    bas = mt_tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

    await ws.send_text(json.dumps({"fr": fr, "lg": bas}))

    # ───── TTS ─────
    prompt = (f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker"
              f"<|text|>{bas}{tts_tokenizer.eos_token}<|audio|>")
    in_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # acoustic model off‑loaded to thread
    gen = await asyncio.to_thread(
        tts_acoustic_model.generate,
        in_ids,
        max_new_tokens=tts_token_budget(bas),
        pad_token_id=tts_tokenizer.pad_token_id,
        eos_token_id=tts_tokenizer.eos_token_id,
    )

    tail = gen[0][in_ids.shape[-1]:].tolist()
    raw = [t - 128_266 - ((i % 7) * 4096) for i, t in enumerate(tail)]
    raw = raw[: (len(raw) // 7) * 7]

    tracks = [[], [], []]
    for i in range(0, len(raw), 7):
        f = raw[i : i + 7]
        if any(not 0 <= c < 4096 for c in f): continue
        tracks[0].append(f[0])
        tracks[1].extend([f[1], f[4]])
        tracks[2].extend([f[2], f[3], f[5], f[6]])

    if tracks[0]:
        codes = [torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(0) for t in tracks]

        wav = await asyncio.to_thread(
            lambda: tts_vocoder.decode(codes).detach().cpu().numpy().squeeze()
        )

        buf = io.BytesIO()
        sf.write(buf, wav, 24_000, format="WAV", subtype="PCM_16")
        try:
            await ws.send_bytes(memoryview(buf.getbuffer()))
        except (WebSocketDisconnect, ConnectionClosed):
            return

    await ws.close()
    print(f"⏱  voice pipeline {time.perf_counter()-t0:.2f}s")

# ══════════════════════════════════════════════════════════════
#  Web‑Socket 2 :  /translate_text
# ══════════════════════════════════════════════════════════════
@app.websocket("/translate_text")
async def translate_text(ws: WebSocket):
    await ws.accept()
    try:
        fr = await ws.receive_text()
        mt_tokenizer.src_lang = "fr"
        enc = mt_tokenizer(fr, return_tensors="pt").to(DEVICE)
        bos = mt_tokenizer.get_lang_id("lg")
        with torch.inference_mode():
            ids = mt_model.generate(**enc, forced_bos_token_id=bos)
        bas = mt_tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        await ws.send_text(bas)
    except WebSocketDisconnect:
        pass
    finally:
        await ws.close()
