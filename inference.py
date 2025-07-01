from __future__ import annotations

"""inference.py
Reusable transcription utilities for KinyaWhisper.

Usage (CLI):
    python inference.py --audio path/to/file.mp3

This module also exposes the function:
    transcribe_audio(file_path: str) -> str
which is imported by the Flask server (app.py).
"""

import argparse
import functools
import os
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

_MODEL_DIR = "benax-rw/KinyaWhisper"  # HF Hub or local path
_device = torch.device("cpu")  # Explicitly use CPU to keep container slim


def _load_model_and_processor() -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load and cache Whisper model/processor to avoid reloading for every call."""

    @functools.lru_cache(maxsize=1)
    def _loader():
        model = WhisperForConditionalGeneration.from_pretrained(_MODEL_DIR)
        processor = WhisperProcessor.from_pretrained(_MODEL_DIR)
        model.eval()
        return model, processor

    return _loader()


def transcribe_audio(file_path: str) -> str:
    """Transcribe a single audio file and return the text.

    Parameters
    ----------
    file_path : str
        Path to the audio file (.wav, .mp3, etc.)
    Returns
    -------
    str
        Transcribed text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    model, processor = _load_model_and_processor()

    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if needed
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file using KinyaWhisper")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav/.mp3)")
    args = parser.parse_args()

    text = transcribe_audio(args.audio)
    print(text)


if __name__ == "__main__":
    _cli()