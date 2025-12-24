"""
RunPod Serverless Handler for SAM-Audio.
"""

import sys
print("="*60, flush=True)
print("SAM-Audio Handler - Early startup", flush=True)
print(f"Python: {sys.version}", flush=True)

try:
    import os
    import time
    import base64
    import tempfile
    from io import BytesIO
    from datetime import datetime
    print("Basic imports OK", flush=True)
    
    import requests
    print("requests OK", flush=True)
    
    import runpod
    print("runpod OK", flush=True)
    
    import torch
    print(f"torch OK - CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    import torchaudio
    print("torchaudio OK", flush=True)
    
except Exception as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All imports successful!", flush=True)
print("="*60, flush=True)


def log(message: str):
    """Print timestamped log message."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


# Global model reference
model = None
processor = None


def load_model():
    """Load model and processor on cold start."""
    global model, processor

    model_size = os.environ.get("MODEL_SIZE", "large")
    model_name = f"facebook/sam-audio-{model_size}"

    log(f"Loading model: {model_name}")
    log(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    log(f"HUGGING_FACE_HUB_TOKEN set: {bool(os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")
    
    start_time = time.time()
    
    try:
        log("Importing sam_audio...")
        from sam_audio import SAMAudio, SAMAudioProcessor
        log("sam_audio imported OK")
        
        log("Loading model from HuggingFace (may take 10-20 min on first run)...")
        model = SAMAudio.from_pretrained(model_name)
        log(f"Model loaded in {time.time() - start_time:.1f}s")
        
        log("Loading processor...")
        processor = SAMAudioProcessor.from_pretrained(model_name)
        log("Processor loaded")
        
        log("Moving model to CUDA...")
        model = model.eval().cuda()
        
        log(f"Model ready! Total time: {time.time() - start_time:.1f}s")
        
    except Exception as e:
        log(f"MODEL LOAD ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def download_audio(url: str) -> str:
    """Download audio from URL to temp file."""
    log(f"Downloading: {url[:80]}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    suffix = ".wav"
    if ".mp3" in url.lower():
        suffix = ".mp3"
    elif ".flac" in url.lower():
        suffix = ".flac"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(response.content)
        log(f"Downloaded {len(response.content)/1024:.1f} KB")
        return f.name


def handler(event):
    """RunPod serverless handler for SAM-Audio."""
    global model, processor

    log("Request received")

    if model is None:
        load_model()

    input_data = event.get("input", {})
    audio_url = input_data.get("audio_url")
    audio_base64 = input_data.get("audio_base64")
    description = input_data.get("description", "")

    predict_spans = input_data.get(
        "predict_spans",
        os.environ.get("PREDICT_SPANS", "true").lower() == "true"
    )
    reranking_candidates = input_data.get(
        "reranking_candidates",
        int(os.environ.get("RERANKING_CANDIDATES", "4"))
    )
    anchors = input_data.get("anchors")

    log(f"Description: {description}")
    log(f"predict_spans={predict_spans}, reranking={reranking_candidates}")

    if not audio_url and not audio_base64:
        return {"error": "Either 'audio_url' or 'audio_base64' is required"}
    if not description:
        return {"error": "'description' is required"}

    temp_files = []

    try:
        if audio_url:
            audio_path = download_audio(audio_url)
            temp_files.append(audio_path)
        else:
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                audio_path = f.name
                temp_files.append(audio_path)

        log("Processing audio...")
        batch_kwargs = {"audios": [audio_path], "descriptions": [description]}
        if anchors:
            batch_kwargs["anchors"] = [anchors]

        batch = processor(**batch_kwargs).to("cuda")

        log("Running separation...")
        start = time.time()
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates
            )
        log(f"Separation done in {time.time() - start:.1f}s")

        sample_rate = processor.audio_sampling_rate

        def audio_to_base64(audio_tensor):
            buffer = BytesIO()
            torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format="wav")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

        log("Encoding output...")
        return {
            "target_audio": audio_to_base64(result.target),
            "residual_audio": audio_to_base64(result.residual),
            "sample_rate": sample_rate
        }

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass


# Cold start
log("Starting cold start model load...")
load_model()

log("Starting RunPod handler...")
runpod.serverless.start({"handler": handler})
