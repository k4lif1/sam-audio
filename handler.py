"""
RunPod Serverless Handler for SAM-Audio.

This handler provides a serverless API for the SAM-Audio model,
allowing audio separation based on text descriptions.
"""

import os
import sys
import time
import base64
import tempfile
from io import BytesIO
from datetime import datetime

import requests
import runpod
import torch
import torchaudio

# Configure logging to stdout for RunPod visibility
def log(message: str):
    """Print timestamped log message to stdout."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)

log("="*60)
log("SAM-Audio Handler Starting")
log(f"Python version: {sys.version}")
log(f"PyTorch version: {torch.__version__}")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"CUDA device: {torch.cuda.get_device_name(0)}")
    log(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
log("="*60)

# Global model reference
model = None
processor = None


def load_model():
    """Load model and processor on cold start."""
    global model, processor

    model_size = os.environ.get("MODEL_SIZE", "large")
    model_name = f"facebook/sam-audio-{model_size}"

    log(f"Starting model load: {model_name}")
    log(f"HuggingFace cache dir: {os.environ.get('HF_HOME', 'default')}")
    
    start_time = time.time()
    
    log("Loading SAMAudio model from HuggingFace...")
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    log("Downloading/loading model weights (this may take several minutes on first run)...")
    model = SAMAudio.from_pretrained(model_name)
    model_load_time = time.time() - start_time
    log(f"Model weights loaded in {model_load_time:.1f}s")
    
    log("Loading processor...")
    processor = SAMAudioProcessor.from_pretrained(model_name)
    processor_load_time = time.time() - start_time - model_load_time
    log(f"Processor loaded in {processor_load_time:.1f}s")
    
    log("Moving model to CUDA...")
    model = model.eval().cuda()
    
    total_time = time.time() - start_time
    log(f"Model ready! Total load time: {total_time:.1f}s")
    log("="*60)


def download_audio(url: str) -> str:
    """Download audio from URL to temp file."""
    log(f"Downloading audio from: {url[:100]}...")
    start_time = time.time()
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    suffix = ".wav"
    if ".mp3" in url.lower():
        suffix = ".mp3"
    elif ".flac" in url.lower():
        suffix = ".flac"
    elif ".ogg" in url.lower():
        suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(response.content)
        download_time = time.time() - start_time
        log(f"Audio downloaded: {len(response.content)/1024:.1f} KB in {download_time:.1f}s -> {f.name}")
        return f.name


def handler(event):
    """
    RunPod serverless handler for SAM-Audio.

    Input:
        - audio_url (str): URL to the audio file
        - audio_base64 (str): Base64 encoded audio (alternative to URL)
        - description (str): Text description of the sound to isolate
        - predict_spans (bool): Enable automatic span prediction (default: True)
        - reranking_candidates (int): Number of reranking candidates (default: 4)
        - anchors (list): Optional time ranges [["+", start, end], ...]

    Output:
        - target_audio (str): Base64 encoded separated target audio (WAV)
        - residual_audio (str): Base64 encoded residual audio (WAV)
        - sample_rate (int): Audio sample rate
    """
    global model, processor

    log("="*60)
    log("New request received")
    request_start = time.time()

    if model is None:
        log("Model not loaded, loading now...")
        load_model()
    else:
        log("Model already loaded, processing request...")

    input_data = event.get("input", {})

    # Get audio input
    audio_url = input_data.get("audio_url")
    audio_base64 = input_data.get("audio_base64")
    description = input_data.get("description", "")

    log(f"Description: '{description}'")
    log(f"Audio URL provided: {bool(audio_url)}")
    log(f"Audio base64 provided: {bool(audio_base64)}")

    # Get processing options with defaults from environment
    predict_spans = input_data.get(
        "predict_spans",
        os.environ.get("PREDICT_SPANS", "true").lower() == "true"
    )
    reranking_candidates = input_data.get(
        "reranking_candidates",
        int(os.environ.get("RERANKING_CANDIDATES", "4"))
    )
    anchors = input_data.get("anchors")
    
    log(f"Options: predict_spans={predict_spans}, reranking_candidates={reranking_candidates}")

    # Validation
    if not audio_url and not audio_base64:
        log("ERROR: No audio input provided")
        return {"error": "Either 'audio_url' or 'audio_base64' is required"}

    if not description:
        log("ERROR: No description provided")
        return {"error": "'description' is required"}

    temp_files = []

    try:
        # Load audio
        if audio_url:
            audio_path = download_audio(audio_url)
            temp_files.append(audio_path)
        else:
            log("Decoding base64 audio...")
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                audio_path = f.name
                temp_files.append(audio_path)
                log(f"Audio saved to: {audio_path}")

        # Prepare batch
        log("Preparing audio batch...")
        batch_kwargs = {
            "audios": [audio_path],
            "descriptions": [description],
        }

        if anchors:
            batch_kwargs["anchors"] = [anchors]
            log(f"Using anchors: {anchors}")

        batch = processor(**batch_kwargs).to("cuda")
        log("Batch prepared and moved to CUDA")

        # Run separation
        log("Running audio separation...")
        separation_start = time.time()
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates
            )
        separation_time = time.time() - separation_start
        log(f"Separation complete in {separation_time:.1f}s")

        sample_rate = processor.audio_sampling_rate

        # Encode outputs as base64
        log("Encoding output audio to base64...")
        def audio_to_base64(audio_tensor):
            buffer = BytesIO()
            torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format="wav")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

        target_b64 = audio_to_base64(result.target)
        residual_b64 = audio_to_base64(result.residual)
        
        total_time = time.time() - request_start
        log(f"Request complete! Total time: {total_time:.1f}s")
        log(f"Output sizes: target={len(target_b64)/1024:.1f}KB, residual={len(residual_b64)/1024:.1f}KB")
        log("="*60)

        return {
            "target_audio": target_b64,
            "residual_audio": residual_b64,
            "sample_rate": sample_rate
        }

    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return {"error": str(e)}

    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                os.unlink(f)
                log(f"Cleaned up temp file: {f}")
            except Exception:
                pass


# Initialize model on cold start
log("Initializing model on cold start...")
load_model()

log("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
