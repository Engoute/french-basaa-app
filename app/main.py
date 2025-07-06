# app/main.py  – GPU-resident models • chunk-streaming • keep-alive ping
import asyncio, io, json, os, shutil, tempfile, traceback, zipfile
from pathlib import Path

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

# ────────────────────────── optional bleeding-edge import ─────────────────────────
try:
    from snac.configuration_snac import SnacConfig  # modern snac
except ModuleNotFoundError:                         # fallback shim
    class SnacConfig(dict):
        """Enough to satisfy SNAC(cfg) without touching the Hub."""
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e
# ──────────────────────────────────────────────────────────────────────────────────

# ───────── Config ────────────────────────────────────────────────────────────────
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

# ───────── Globals ───────────────────────────────────────────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ───────── Helpers ───────────────────────────────────────────────────────────────
def safe_unzip(zip_path: Path, target: Path, url: str) -> None:
    """Download & extract `url` once; do nothing if `target` already populated."""
    if target.exists() and any(target.iterdir()):
        return
    target.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for p in Path(tmp).iterdir():        # preserve internal structure
            shutil.move(str(p), target)

    os.remove(zip_path)

def find_model_dir(base: Path, marker_files: list[str]) -> Path:
    """
    Return first directory under `base` that contains ALL `marker_files`.
    Allows us to ignore one-level nesting in zips.
    """
    for root, _, files in os.walk(base):
        if all(m in files for m in marker_files):
            return Path(root)
    raise FileNotFoundError(f"No dir with {marker_files} beneath {base}")

def wav_to_pcm16(blob: bytes) -> np.ndarray:
    """Accept 16-kHz WAV or raw PCM16 and return np.int16 PCM."""
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError("WAV must be 16 kHz")
        return data
    return np.frombuffer(blob, np.int16)

def load_snac_local(model_dir: Path, device: str = "cpu") -> SNAC:
    """
    Offline loader that mimics `SNAC.from_pretrained` but *never* touches the Hub.
    – If `"checkpoint"` not present ⇢ falls back to first weight file found.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SNAC config not found: {cfg_path}")

    cfg = SnacConfig(**json.loads(cfg_path.read_text()))

    # ── resolve checkpoint ──────────────────────────────────────
    ckpt_name = getattr(cfg, "checkpoint", None)
    if ckpt_name is None:
        # common default file names
        for cand in ("pytorch_model.bin", "model.bin", "model.safetensors"):
            if (model_dir / cand).exists():
                ckpt_name = cand
                break
        else:
            # grab whatever *.bin / *.safetensors is first
            bins = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
            if not bins:
                raise FileNotFoundError(f"No weight file in {model_dir}")
            ckpt_name = bins[0].name
        cfg["checkpoint"] = ckpt_name  # persist for future calls

    vocoder = SNAC(cfg)
    vocoder.load_state_dict(torch.load(model_dir / ckpt_name, map_location=device))
    return vocoder.to(device).eval()

# ───────── Model loader ──────────────────────────────────────────────────────────
def load_models() -> None:
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder

    # ── download / unzip once ───────────────────────────────────
    safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
    safe_unzip(MODELS_DIR / "m2m100.zip",  MT_MODEL_PATH,  MODEL_URLS["m2m100.zip"])
    safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

    # ── Whisper (ASR) ───────────────────────────────────────────
    asr_root      = find_model_dir(ASR_MODEL_PATH, ["preprocessor_config.json", "config.json"])
    asr_processor = AutoProcessor.from_pretrained(asr_root, local_files_only=True)
    asr_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_root, torch_dtype=torch.float16, device_map="auto"
    )

    # ── M2M-100 (MT) ────────────────────────────────────────────
    mt_root     = find_model_dir(MT_MODEL_PATH, ["config.json"])
    mt_tokenizer = AutoTokenizer.from_pretrained(mt_root, local_files_only=True)
    mt_model     = M2M100ForConditionalGeneration.from_pretrained(
        mt_root, device_map="auto"
    )

    # ── Orpheus (TTS) ───────────────────────────────────────────
    ac_root = TTS_MODEL_PATH / "acoustic_model"
    if not ac_root.exists():                       # flattened layout
        ac_root = TTS_MODEL_PATH
    vc_root = TTS_MODEL_PATH / "vocoder"

    tts_tokenizer      = AutoTokenizer.from_pretrained(ac_root, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(ac_root, torch_dtype="auto").to(DEVICE).eval()
    tts_vocoder        = load_snac_local(vc_root, DEVICE)

    # ── Performance knobs ───────────────────────────────────────
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ────────── FastAPI startup ──────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    print("✅ Models loaded — server ready")

# ────────── WebSocket endpoint ───────────────────────────────────────────────────
PING_EVERY = 25  # seconds

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_buffer = bytearray()

    async def keep_alive():
        while True:
            await asyncio.sleep(PING_EVERY)
            try:
                await ws.send_bytes(b"\x00")  # 1-byte ping
            except Exception:
                break
    asyncio.create_task(keep_alive())

    try:
        while True:
            # ── receive mic chunk or silence / close ────────────
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

            # ── ASR ──────────────────────────────────────────────
            pcm16 = wav_to_pcm16(bytes(pcm_buffer)).astype(np.float32) / 32768.0
            feats = asr_processor(
                pcm16, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(asr_model.device).half()
            with torch.inference_mode():
                txt_ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(txt_ids, skip_special_tokens=True)[0].strip()

            # ── MT ───────────────────────────────────────────────
            mt_tokenizer.src_lang = "fr"
            enc = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            bos = mt_tokenizer.get_lang_id("lg")
            with torch.inference_mode():
                trg_ids = mt_model.generate(**enc, forced_bos_token_id=bos)
            lg = mt_tokenizer.batch_decode(trg_ids, skip_special_tokens=True)[0]

            # ── send texts immediately (UI) ─────────────────────
            await ws.send_text(json.dumps({"fr": fr, "lg": lg}))

            # ── TTS ─────────────────────────────────────────────
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
                    f = diff[i : i + 7]
                    if any(not 0 <= c < 4096 for c in f):
                        continue
                    tracks[0].append(f[0])
                    tracks[1].extend([f[1], f[4]])
                    tracks[2].extend([f[2], f[3], f[5], f[6]])
                codes = [torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(0) for t in tracks]

                wav = (
                    tts_vocoder.decode(codes)        # (1, C, T)
                    .squeeze(0)                      # (C, T)
                    .detach()
                    .cpu()
                    .numpy()
                    .T                               # (T, C)
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
