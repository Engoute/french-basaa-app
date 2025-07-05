# app/main.py  – GPU-resident models • chunk-streaming • keep-alive ping
import asyncio, io, json, os, shutil, tempfile, traceback, zipfile
from pathlib import Path

import gdown, numpy as np, soundfile as sf, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (AutoModelForSpeechSeq2Seq, AutoModelForCausalLM,
                          AutoProcessor, AutoTokenizer,
                          M2M100ForConditionalGeneration)
from snac import SNAC

# ─────────── Config ─────────────────────────────────────────────
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

# ─────────── Globals (loaded once) ──────────────────────────────
asr_model = asr_processor = mt_model = mt_tokenizer = None
tts_acoustic_model = tts_tokenizer = tts_vocoder = None

app = FastAPI()

# ─────────── Helpers ────────────────────────────────────────────
def safe_unzip(zip_path: Path, target: Path, url: str) -> None:
    if target.joinpath("config.json").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        root = next(Path(tmp).rglob("config.json")).parent
        for p in root.iterdir():
            shutil.move(str(p), target)
    os.remove(zip_path)

def wav_to_pcm16(blob: bytes) -> np.ndarray:
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="int16")
        if sr != 16_000:
            raise ValueError("WAV must be 16 kHz")
        return data
    return np.frombuffer(blob, np.int16)

def load_models():
    global asr_model, asr_processor, mt_model, mt_tokenizer
    global tts_acoustic_model, tts_tokenizer, tts_vocoder

    safe_unzip(MODELS_DIR/"whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper.zip"])
    safe_unzip(MODELS_DIR/"m2m100.zip",  MT_MODEL_PATH,  MODEL_URLS["m2m100.zip"])
    safe_unzip(MODELS_DIR/"orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus.zip"])

    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
    asr_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

    mt_tokenizer  = AutoTokenizer.from_pretrained(MT_MODEL_PATH, local_files_only=True)
    mt_model      = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_PATH, device_map="auto")

    ac, vc = TTS_MODEL_PATH / "acoustic_model", TTS_MODEL_PATH / "vocoder"
    tts_tokenizer      = AutoTokenizer.from_pretrained(ac, local_files_only=True)
    tts_acoustic_model = AutoModelForCausalLM.from_pretrained(ac, torch_dtype="auto").to(DEVICE).eval()
    tts_vocoder        = SNAC.from_pretrained(vc, local_files_only=True).to(DEVICE).eval()

# ─────────── Startup ────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    print("✅ Models loaded — server ready")

# ─────────── WebSocket ──────────────────────────────────────────
PING_EVERY = 25            # seconds

@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_buffer = bytearray()
    last_recv  = asyncio.get_event_loop().time()

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
            # ── receive mic chunk or close ───────────────────────
            try:
                chunk = await asyncio.wait_for(ws.receive_bytes(), timeout=PING_EVERY+5)
                pcm_buffer.extend(chunk)
                last_recv = asyncio.get_event_loop().time()
                continue                         # wait for more chunks
            except asyncio.TimeoutError:
                # no new audio for a while → treat buffer as utterance
                pass
            except WebSocketDisconnect:
                break

            if not pcm_buffer:
                continue                         # nothing yet

            # ── ASR ──────────────────────────────────────────────
            pcm16 = wav_to_pcm16(bytes(pcm_buffer)).astype(np.float32) / 32768.0
            feats = asr_processor(pcm16, sampling_rate=16_000,
                                  return_tensors="pt").input_features.to(asr_model.device).half()
            with torch.no_grad():
                txt_ids = asr_model.generate(feats)
            fr = asr_processor.batch_decode(txt_ids, skip_special_tokens=True)[0].strip()

            # ── MT ───────────────────────────────────────────────
            mt_tokenizer.src_lang = "fr"
            enc = mt_tokenizer(fr, return_tensors="pt").to(mt_model.device)
            bos = mt_tokenizer.get_lang_id("lg")
            with torch.no_grad():
                trg_ids = mt_model.generate(**enc, forced_bos_token_id=bos)
            lg = mt_tokenizer.batch_decode(trg_ids, skip_special_tokens=True)[0]

            # ── send texts immediately (UI) ─────────────────────
            await ws.send_text(json.dumps({"fr": fr, "lg": lg}))

            # ── TTS ─────────────────────────────────────────────
            prompt = f"{tts_tokenizer.bos_token}<|voice|>basaa_speaker<|text|>{lg}{tts_tokenizer.eos_token}<|audio|>"
            token_in = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                llm = tts_acoustic_model.generate(token_in, max_new_tokens=4000,
                                                   pad_token_id=tts_tokenizer.pad_token_id,
                                                   eos_token_id=tts_tokenizer.eos_token_id)

            diff   = [t-128266-((i%7)*4096) for i,t in enumerate(llm[0][token_in.shape[-1]:])]
            diff   = diff[: (len(diff)//7)*7]
            tracks = [[],[],[]]
            for i in range(0,len(diff),7):
                f = diff[i:i+7]
                if any(not 0<=c<4096 for c in f): continue
                tracks[0].append(f[0]); tracks[1].extend([f[1],f[4]]); tracks[2].extend([f[2],f[3],f[5],f[6]])
            codes  = [torch.tensor(t,dtype=torch.long,device=DEVICE).unsqueeze(0) for t in tracks]
            wav = (
                tts_vocoder.decode(codes)  # (B, C, T)
                    .detach()              # ① break the grad graph
                    .cpu()
                    .numpy()
                    .squeeze()
            )    

            buf = io.BytesIO(); sf.write(buf, wav, 16_000, format="WAV", subtype="PCM_16")
            await ws.send_bytes(buf.getvalue())

            pcm_buffer.clear()                   # ready for next utterance

    except Exception as e:
        print("❌", e); traceback.print_exc()
        if not ws.client_state.name.startswith("CLOS"):      # avoid double-close
            await ws.close(code=1011, reason="server error")
