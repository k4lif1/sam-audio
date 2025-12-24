"""
RunPod Serverless Handler for SAM-Audio.

This handler provides a serverless API for the SAM-Audio model,
allowing audio separation based on text descriptions.
"""

import os
import base64
import tempfile
from io import BytesIO

import requests
import runpod
import torch
import torchaudio

from sam_audio import SAMAudio, SAMAudioProcessor

# Global model reference
model = None
processor = None


def load_model():
    """Load model and processor on cold start."""
    global model, processor

    model_size = os.environ.get("MODEL_SIZE", "large")
    model_name = f"facebook/sam-audio-{model_size}"

    print(f"Loading model: {model_name}")

    model = SAMAudio.from_pretrained(model_name)
    processor = SAMAudioProcessor.from_pretrained(model_name)
    model = model.eval().cuda()

    print("Model loaded successfully")


def download_audio(url: str) -> str:
    """Download audio from URL to temp file."""
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

    if model is None:
        load_model()

    input_data = event.get("input", {})

    # Get audio input
    audio_url = input_data.get("audio_url")
    audio_base64 = input_data.get("audio_base64")
    description = input_data.get("description", "")

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

    # Validation
    if not audio_url and not audio_base64:
        return {"error": "Either 'audio_url' or 'audio_base64' is required"}

    if not description:
        return {"error": "'description' is required"}

    temp_files = []

    try:
        # Load audio
        if audio_url:
            audio_path = download_audio(audio_url)
            temp_files.append(audio_path)
        else:
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                audio_path = f.name
                temp_files.append(audio_path)

        # Prepare batch
        batch_kwargs = {
            "audios": [audio_path],
            "descriptions": [description],
        }

        if anchors:
            batch_kwargs["anchors"] = [anchors]

        batch = processor(**batch_kwargs).to("cuda")

        # Run separation
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates
            )

        sample_rate = processor.audio_sampling_rate

        # Encode outputs as base64
        def audio_to_base64(audio_tensor):
            buffer = BytesIO()
            torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format="wav")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "target_audio": audio_to_base64(result.target),
            "residual_audio": audio_to_base64(result.residual),
            "sample_rate": sample_rate
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass


# Initialize model on cold start
load_model()

runpod.serverless.start({"handler": handler})
