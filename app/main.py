# app/main.py  –  GPU-resident models • chunk-streaming • keep-alive ping
import asyncio, io, json, os, shutil, tempfile, traceback, zipfile
import contextlib, sys
from pathlib import Path
from typing import Optional

import gdown, numpy as np, soundfile as sf, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
)
from snac import SNAC

# ─────────────────────────────────────────────────────────────── shim
try:
    from snac.configuration_snac import SnacConfig            # bleeding-edge
except ModuleNotFoundError:                                   # fallback
    class SnacConfig(dict):
        """minimal dict-based stand-in – lets SNAC(**cfg) work offline"""
        def __getattr__(self, k):  # dot-access
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e
# ─────────────────────────────────────────────────────────────── paths
MODELS_DIR     = Path(os.getenv("MODELS_DIR", "/workspace/models"))
ASR_MODEL_PATH = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_PATH  = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_PATH = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URLS = {
    "whisper.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/whisper.zip",
    "m2m100.zip":  "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/m2m100.zip",
    "orpheus.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/orpheus.zip",
}

# globals – filled at startup
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ───────────────────────── helpers ──────────────────────────────────────────
def safe_unzip(zip_path: Path, target: Path, url: str) -> None:
    """Download `url` once and unzip to `target` – skipped when target non-empty."""
    if target.exists() and any(target.iterdir()):
        return
    target.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for p in Path(tmp).iterdir():                # file OR dir
            shutil.move(str(p), target)
    os.remove(zip_path)

def resolve_model_dir(root: Path) -> Path:
    """Return `root` if it has config.json else the first sub-dir that has one."""
    if (root / "config.json").exists():
        return root
    try:
        return next(root.rglob("config.json")).parent
    except StopIteration:
        raise FileNotFoundError(f"No config.json under {root}")

def wav_to_pcm16(blob: bytes) -> np.ndarray:
    """Accept 16-kHz WAV or raw PCM16 → np.int16 PCM array."""
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError("WAV must be 16 kHz")
        return data
    return np.frombuffer(blob, np.int16)

# ───────────────────────── SNAC loader ──────────────────────────────────────
def load_snac_local(model_dir: Path, device: str = "cpu") -> SNAC:
    """
    Fully-offline SNAC loader.
    • tries strict load first
    • on mismatch: retries strict=False, prints ONE concise line,
      then writes a `.fixed` marker so next boots are silent
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SNAC config not found: {cfg_path}")

    marker = model_dir / ".fixed"
    cfg    = SnacConfig(**json.loads(cfg_path.read_text()))
    voc    = SNAC(cfg)

    # checkpoint file
    ckpt: Optional[Path] = getattr(cfg, "checkpoint", None)
    ckpt = model_dir / ckpt if ckpt else next(model_dir.glob("*.bin*"), None)
    if not ckpt or not ckpt.exists():
        raise FileNotFoundError(f"SNAC weights not found in {model_dir}")

    state = torch.load(ckpt, map_location=device)
    try:
        voc.load_state_dict(state, strict=True)
    except RuntimeError:                      # size mismatch
        voc.load_state_dict(state, strict=False)
        if not marker.exists():               # only say it once
            print("⚠️  SNAC: weights-≠-config → loaded with strict=False")
            marker.touch()
    return voc.to(device).eval()

# ───────────────────────── model bootstrap ──────────────────────────────────


def load_snac_local(model_dir: Path, device: str = "cpu") -> SNAC:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SNAC config not found: {cfg_path}")

    cfg = SnacConfig(**json.loads(cfg_path.read_text()))
    voc = SNAC(cfg)

    # find checkpoint
    ckpt = model_dir / getattr(cfg, "checkpoint", "")
    if not ckpt.exists():
        ckpt = next(model_dir.glob("*.bin*"), None)
    if not ckpt:
        raise FileNotFoundError(f"SNAC weights not found in {model_dir}")

    state = torch.load(ckpt, map_location=device)

    # suppress PyTorch’s noisy printout
    with contextlib.redirect_stdout(io.StringIO()):
        voc.load_state_dict(state, strict=False)

    # optional one-liner so you *know* it used the lenient path
    print("⚠️  SNAC loaded with strict=False")

    return voc.to(device).eval()

    # 4) Orpheus TTS
    ac_root = (TTS_MODEL_PATH / "acoustic_model") if (TTS_MODEL_PATH / "acoustic_model").exists() else TTS_MODEL_PATH
    vc_root = TTS_MODEL_PATH / "vocoder"

    tts_tokenizer      = AutoTokenizer.from_pretrained(ac_root, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(ac_root, torch_dtype="auto").to(DEVICE).eval()
    tts_vocoder        = load_snac_local(vc_root, DEVICE)

    # 5) tiny perf knobs
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ───────────────────────── FastAPI lifecycle ────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    print("✅ Models loaded — server ready")

# ───────────────────────── websocket endpoint ───────────────────────────────
PING_EVERY = 25  # seconds

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_buffer = bytearray()

    async def keep_alive():
        while True:
            await asyncio.sleep(PING_EVERY)
            try:
                await ws.send_bytes(b"\x00")   # 1-byte ping
            except Exception:
                break
    asyncio.create_task(keep_alive())

    try:
        while True:
            # ── receive audio or detect silence ──────────────────────────
            try:
                chunk = await asyncio.wait_for(ws.receive_bytes(), timeout=PING_EVERY + 5)
                pcm_buffer.extend(chunk)
                continue
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            if not pcm_buffer:
                continue

            # ── ASR ─────────────────────────────────────────────────────
            pcm16 = wav_to_pcm16(bytes(pcm_buffer)).astype(np.float32) / 32768.0
            feats = asr_processor(pcm16, sampling_rate=16_000,
                                  return_tensors="pt").input_features.to(asr_model.device).half()
            with torch.inference_mode():
                txt_ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(txt_ids, skip_special_tokens=True)[0].strip()

            # ── MT ──────────────────────────────────────────────────────
            mt_tokenizer.src_lang = "fr"
            enc = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            bos = mt_tokenizer.get_lang_id("lg")
            with torch.inference_mode():
                trg_ids = mt_model.generate(**enc, forced_bos_token_id=bos)
            lg = mt_tokenizer.batch_decode(trg_ids, skip_special_tokens=True)[0]

            await ws.send_text(json.dumps({"fr": fr, "lg": lg}))

            # ── TTS ─────────────────────────────────────────────────────
            prompt   = f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker<|text|>{lg}{tts_tokenizer.eos_token}<|audio|>"
            token_in = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

            with torch.inference_mode():
                llm = tts_acoustic_model.generate(
                    token_in,
                    max_new_tokens=4000,
                    pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id,
                )

                diff   = [t - 128266 - ((i % 7) * 4096) for i, t in enumerate(llm[0][token_in.shape[-1]:])]
                diff   = diff[: (len(diff) // 7) * 7]
                tracks = [[], [], []]
                for i in range(0, len(diff), 7):
                    f = diff[i:i+7]
                    if any(not 0 <= c < 4096 for c in f):
                        continue
                    tracks[0].append(f[0])
                    tracks[1].extend([f[1], f[4]])
                    tracks[2].extend([f[2], f[3], f[5], f[6]])
                codes = [torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(0) for t in tracks]

                wav = (
                    tts_vocoder.decode(codes)      # (1, C, T)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .T                             # (T, C)
                )

            buf = io.BytesIO()
            sf.write(buf, wav, 16_000, format="WAV", subtype="PCM_16")
            await ws.send_bytes(buf.getvalue())
            pcm_buffer.clear()

    except Exception as e:
        print("❌", e)
        traceback.print_exc()
        if not ws.client_state.name.startswith("CLOS"):
            await ws.close(code=1011, reason="server error")
