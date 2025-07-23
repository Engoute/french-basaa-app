# app/main.py – Refined for streaming inference with dedicated engines
import asyncio, io, json, os, shutil, tempfile, zipfile, time
from pathlib import Path
from typing import AsyncGenerator

import gdown, numpy as np, soundfile as sf, torch
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, M2M100Tokenizer
from vllm import AsyncLLMEngine, SamplingParams
from snac import SNAC

# ══════════════════════════════════════════════════════════════
#  Configuration & Paths
# ══════════════════════════════════════════════════════════════
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: Paths now point to directories containing optimized model formats
# as recommended by the research report (CTranslate2, ONNX, and vLLM-compatible).
ASR_MODEL_PATH = MODELS_DIR / "whisper-small-fr-ct2"
MT_MODEL_PATH = MODELS_DIR / "m2m100-fr-basaa-onnx"
TTS_MODEL_PATH = MODELS_DIR / "orpheus-basaa-vllm"

MODEL_URLS = {
    "whisper_ct2.zip": "URL_TO_FASTER_WHISPER_MODEL.zip", # Placeholder URL
    "m2m100_onnx.zip": "URL_TO_M2M100_ONNX_MODEL.zip",     # Placeholder URL
    "orpheus_vllm.zip": "URL_TO_ORPHEUS_VLLM_MODEL.zip",   # Placeholder URL
}

# ─── Globals for service instances ────────────────────────────
asr_service = None
mt_service = None
tts_service = None

app = FastAPI()

# ─── helpers ──────────────────────────────────────────────────
def safe_unzip(zip_path: Path, dst_dir: Path, url: str) -> None:
    if dst_dir.exists() and any(dst_dir.iterdir()):
        print(f"✅ Models already exist in {dst_dir}")
        return
    print(f"Downloading and unzipping models from {url} to {dst_dir}...")
    dst_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dst_dir)
    os.remove(zip_path)
    print("✅ Download complete.")

def wav_to_f32(blob: bytes) -> np.ndarray:
    if blob[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(blob), dtype="float32")
        if sr != 16_000:
            raise ValueError("Expected 16 kHz WAV audio.")
        return data
    # Assume raw int16 bytes if not a WAV file
    return np.frombuffer(blob, np.int16).astype(np.float32) / 32768.0

# ══════════════════════════════════════════════════════════════
#  Inference Services (as per research report recommendations)
# ══════════════════════════════════════════════════════════════

class ASRService:
    """ASR service using faster-whisper (CTranslate2)."""
    def __init__(self, model_dir: str):
        print("Loading ASR model (faster-whisper)...")
        self.model = WhisperModel(model_dir, device=DEVICE, compute_type="float16")
        print("✅ ASR Service ready.")

    async def transcribe_stream(self, pcm_f32: np.ndarray) -> AsyncGenerator[str, None]:
        # faster-whisper's stream yields segments as they are transcribed.
        segments, _info = self.model.transcribe(pcm_f32, language="fr", beam_size=2)
        for segment in segments:
            yield segment.text

class TranslationService:
    """Machine Translation service using ONNX Runtime."""
    def __init__(self, model_dir: Path):
        print("Loading MT model (ONNX Runtime)...")
        onnx_path = str(next(model_dir.rglob("*.onnx")))
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
        print("✅ Translation Service ready.")

    def translate(self, text: str) -> str:
        self.tokenizer.src_lang = "fr"
        encoded_fr = self.tokenizer(text, return_tensors="np")
        generated_tokens = self.session.run(
            None,
            {"input_ids": encoded_fr["input_ids"], "attention_mask": encoded_fr["attention_mask"]},
        )[0]
        basaa_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            forced_bos_token_id=self.tokenizer.get_lang_id("lg"),
        )[0]
        return basaa_text

class OrpheusService:
    """TTS service using vLLM for the acoustic model and a SNAC vocoder."""
    def __init__(self, model_dir: Path):
        print("Loading TTS model (vLLM) and SNAC vocoder...")
        acoustic_dir = model_dir / "acoustic_model"
        vocoder_dir = model_dir / "vocoder"

        self.engine = AsyncLLMEngine.from_engine_args(
            model=str(acoustic_dir),
            tokenizer=str(acoustic_dir),
            tensor_parallel_size=1,
            dtype="float16",
            max_model_len=4096,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(acoustic_dir)
        self.vocoder = SNAC.from_pretrained(str(vocoder_dir)).to(DEVICE)
        self.sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)
        print("✅ Orpheus TTS Service ready.")

    async def generate_stream(self, prompt: str) -> AsyncGenerator[bytes, None]:
        """Generates audio chunks by streaming tokens from vLLM."""
        full_prompt = (f"{self.tokenizer.bos_token}<|voice|>basaa_speaker"
                       f"<|text|>{prompt}{self.tokenizer.eos_token}<|audio|>")
        
        results_generator = self.engine.generate(full_prompt, self.sampling_params, request_id=f"tts-{time.time()}")
        
        token_buffer = []
        async for result in results_generator:
            new_tokens = result.outputs[0].token_ids
            # Replicate the logic from the research report to decode chunks
            token_buffer.extend(new_tokens[len(token_buffer):])
            
            # Decode audio when buffer is a multiple of 7 (SNAC codebook size)
            # and large enough to be efficient.
            num_to_decode = (len(token_buffer) // 7) * 7
            if num_to_decode > 0:
                codes_to_process = token_buffer[:num_to_decode]
                token_buffer = token_buffer[num_to_decode:]

                # Process codes and decode with SNAC
                raw = [t - 128_266 - ((i % 7) * 4096) for i, t in enumerate(codes_to_process)]
                tracks = [[], [], []]
                for i in range(0, len(raw), 7):
                    f0, f1, f2, f3, f4, f5, f6 = raw[i: i + 7]
                    tracks[0].append(f0 & 0xFFF)
                    tracks[1].extend([f1 & 0xFFF, f4 & 0xFFF])
                    tracks[2].extend([f2 & 0xFFF, f3 & 0xFFF, f5 & 0xFFF, f6 & 0xFFF])

                codes = [torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(0) for t in tracks]
                
                with torch.inference_mode():
                    wav = self.vocoder.decode(codes).detach().cpu().numpy().squeeze()
                
                buf = io.BytesIO()
                sf.write(buf, wav, 24_000, format="WAV", subtype="PCM_16")
                yield buf.getvalue()

# ─── Model Bootstrap ──────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global asr_service, mt_service, tts_service
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # NOTE: You must provide your own model URLs
    # safe_unzip(MODELS_DIR / "whisper.zip", ASR_MODEL_PATH, MODEL_URLS["whisper_ct2.zip"])
    # safe_unzip(MODELS_DIR / "m2m100.zip", MT_MODEL_PATH, MODEL_URLS["m2m100_onnx.zip"])
    # safe_unzip(MODELS_DIR / "orpheus.zip", TTS_MODEL_PATH, MODEL_URLS["orpheus_vllm.zip"])

    asr_service = ASRService(str(ASR_MODEL_PATH))
    mt_service = TranslationService(MT_MODEL_PATH)
    tts_service = OrpheusService(TTS_MODEL_PATH)
    
    # Enable CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print("🌟 All services loaded and server is ready.")

# ══════════════════════════════════════════════════════════════
#  Streaming WebSocket Endpoint (/translate)
# ══════════════════════════════════════════════════════════════
@app.websocket("/translate")
async def translate(ws: WebSocket):
    await ws.accept()
    pcm_chunks = []
    
    # 1. Receive all audio from the client until "DONE"
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                pcm_chunks.append(wav_to_f32(msg["bytes"]))
            elif msg.get("text") == "DONE":
                break
    except WebSocketDisconnect:
        print("Client disconnected during audio reception.")
        return

    if not pcm_chunks:
        await ws.close(code=4000, reason="No audio received.")
        return

    start_time = time.perf_counter()
    full_pcm = np.concatenate(pcm_chunks)
    
    # 2. ASR -> MT -> TTS streaming pipeline
    try:
        # ASR yields transcribed French text segments
        async for fr_text in asr_service.transcribe_stream(full_pcm):
            # Translate each segment to Basaa
            basaa_text = mt_service.translate(fr_text)
            
            # Send the transcribed/translated text pair to the client
            await ws.send_text(json.dumps({"fr": fr_text, "lg": basaa_text}))

            # Stream generated audio chunks for the Basaa text
            async for audio_chunk in tts_service.generate_stream(basaa_text):
                await ws.send_bytes(audio_chunk)
                
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected during streaming.")
    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")
    finally:
        await ws.close()
        end_time = time.perf_counter()
        print(f"⏱️ Pipeline finished in {end_time - start_time:.2f}s")
