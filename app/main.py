# app/main.py  – GPU models • chunk-streaming • keep-alive ping
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

# ------------------------------------------------------------------
# optional import: exists only in bleeding-edge snac
try:
    from snac.configuration_snac import SnacConfig  # type: ignore
except ModuleNotFoundError:
    class SnacConfig(dict):                         # tiny shim
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e
# ------------------------------------------------------------------

# ─────────── Config ──────────────────────────────────────────────
MODELS_DIR     = Path(os.getenv("MODELS_DIR", "/app/models"))
ASR_MODEL_DIR  = MODELS_DIR / "whisper_fr_inference_v1"
MT_MODEL_DIR   = MODELS_DIR / "m2m100_basaa_inference_v1"
TTS_MODEL_DIR  = MODELS_DIR / "orpheus_basaa_bundle_16bit_final"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URLS = {
    "whisper.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/whisper.zip",
    "m2m100.zip":  "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/m2m100.zip",
    "orpheus.zip": "https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/orpheus.zip",
}

# ─────────── Globals ─────────────────────────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ─────────── Helpers ────────────────────────────────────────────
def safe_unzip(zip_path: Path, target: Path, url: str) -> None:
    """
    Download `url` once and unzip it into `target`.
    If `target` already contains files, skip.
    """
    if target.exists() and any(target.iterdir()):
        return
    target.mkdir(parents=True, exist_ok=True)

    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # move everything inside the tmp dir to target
        for p in Path(tmp).iterdir():
            shutil.move(str(p), target)

    zip_path.unlink(missing_ok=True)


def find_model_root(base: Path, required_files: list[str]) -> Path:
    """
    Recursively search `base` for a directory that contains *all* `required_files`.
    Returns that directory (may be `base` itself).
    """
    for root, dirs, files in os.walk(base):
        if all(f in files for f in required_files):
            return Path(root)
    raise FileNotFoundError(
        f"Could not locate {required_files} under {base} (archive layout unexpected)."
    )


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
    Offline loader that mimics `SNAC.from_pretrained` without the Hub.
    Works even when `config.json` lacks the 'checkpoint' field.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SNAC config not found: {cfg_path}")

    cfg_dict = json.loads(cfg_path.read_text())

    # ── locate checkpoint file ─────────────────────────────────
    ckpt_name = cfg_dict.get("checkpoint")               # optional
    ckpt_file = (model_dir / ckpt_name) if ckpt_name else None
    if not ckpt_file or not ckpt_file.exists():
        # pick the first plausible weight file
        candidates = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(f"No *.bin or *.safetensors found in {model_dir}")
        ckpt_file = candidates[0]
    # make sure config knows the final filename (SNAC expects it)
    cfg_dict["checkpoint"] = ckpt_file.name

    cfg     = SnacConfig(**cfg_dict)
    vocoder = SNAC(cfg)
    vocoder.load_state_dict(torch.load(ckpt_file, map_location=device))
    return vocoder.to(device).eval()



def load_models() -> None:
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder

    # ── download / unzip (only first run) ───────────────────────
    safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_DIR, MODEL_URLS["whisper.zip"])
    safe_unzip(MODELS_DIR / "m2m100.zip",  MT_MODEL_DIR,  MODEL_URLS["m2m100.zip"])
    safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_DIR, MODEL_URLS["orpheus.zip"])

    # ── figure out real roots (handle extra wrapper folders) ────
    whisper_root = find_model_root(ASR_MODEL_DIR, ["config.json", "preprocessor_config.json"])
    mt_root      = find_model_root(MT_MODEL_DIR,  ["config.json"])
    # acoustic / vocoder remain the same but may be nested too
    ac_root      = find_model_root(TTS_MODEL_DIR, ["config.json"])
    vc_root      = find_model_root(TTS_MODEL_DIR / "vocoder", ["config.json"])

    # ── Whisper (ASR) ───────────────────────────────────────────
    asr_processor = AutoProcessor.from_pretrained(whisper_root, local_files_only=True)
    asr_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_root, torch_dtype=torch.float16, device_map="auto"
    )

    # ── M2M-100 (MT) ────────────────────────────────────────────
    mt_tokenizer = AutoTokenizer.from_pretrained(mt_root, local_files_only=True)
    mt_model     = M2M100ForConditionalGeneration.from_pretrained(
        mt_root, device_map="auto"
    )

    # ── Orpheus (TTS) ───────────────────────────────────────────
    tts_tokenizer      = AutoTokenizer.from_pretrained(ac_root, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(
        ac_root, torch_dtype="auto"
    ).to(DEVICE).eval()
    tts_vocoder        = load_snac_local(vc_root, DEVICE)

    # ── Performance knobs ───────────────────────────────────────
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ─────────── Startup ────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    print("✅ Models loaded — server ready")

# ─────────── WebSocket — same as before ─────────────────────────
PING_EVERY = 25  # seconds

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_buffer = bytearray()

    async def keep_alive():
        while True:
            await asyncio.sleep(PING_EVERY)
            try:
                await ws.send_bytes(b"\x00")
            except Exception:
                break
    asyncio.create_task(keep_alive())

    try:
        while True:
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

            # ── ASR ──────────────────────────────────────────
            pcm16  = wav_to_pcm16(bytes(pcm_buffer)).astype(np.float32) / 32768.0
            feats  = asr_processor(pcm16, sampling_rate=16_000, return_tensors="pt")\
                        .input_features.to(asr_model.device).half()
            with torch.inference_mode():
                ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

            # ── MT ───────────────────────────────────────────
            mt_tokenizer.src_lang = "fr"
            enc  = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            bos  = mt_tokenizer.get_lang_id("lg")
            with torch.inference_mode():
                out = mt_model.generate(**enc, forced_bos_token_id=bos)
            lg = mt_tokenizer.batch_decode(out, skip_special_tokens=True)[0]

            await ws.send_text(json.dumps({"fr": fr, "lg": lg}))

            # ── TTS ──────────────────────────────────────────
            prompt   = f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker<|text|>{lg}" \
                       f"{tts_tokenizer.eos_token}<|audio|>"
            token_in = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

            with torch.inference_mode():
                llm = tts_acoustic_model.generate(
                    token_in,
                    max_new_tokens=4000,
                    pad_token_id=tts_tokenizer.pad_token_id,
                    eos_token_id=tts_tokenizer.eos_token_id,
                )
                diff = [t - 128266 - ((i % 7) * 4096)
                        for i, t in enumerate(llm[0][token_in.shape[-1]:])]
                diff = diff[: (len(diff) // 7) * 7]
                tracks = [[], [], []]
                for i in range(0, len(diff), 7):
                    f = diff[i:i+7]
                    if any(not 0 <= c < 4096 for c in f):
                        continue
                    tracks[0].append(f[0])
                    tracks[1].extend([f[1], f[4]])
                    tracks[2].extend([f[2], f[3], f[5], f[6]])
                codes = [torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(0)
                         for t in tracks]
                wav = tts_vocoder.decode(codes).squeeze(0).cpu().numpy().T

            buf = io.BytesIO()
            sf.write(buf, wav, 16_000, format="WAV", subtype="PCM_16")
            await ws.send_bytes(buf.getvalue())
            pcm_buffer.clear()

    except Exception as e:
        print("❌", e)
        traceback.print_exc()
        if not ws.client_state.name.startswith("CLOS"):
            await ws.close(code=1011, reason="server error")
