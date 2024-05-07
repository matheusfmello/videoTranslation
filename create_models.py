"""
Create all models in main.py.

You can run this file before executing the main file, as downloading models
may take a while.
"""

import argparse
import logging
import os

from deep_translator import GoogleTranslator
import moviepy.editor as mp
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Pipeline
from TTS.api import TTS

logging.basicConfig(filename="create_models.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger(__name__)


os.environ["COQUI_TOS_AGREED"] = "1" #  Agree Coqui-ai’s xTTS license for using Text-to-speech models. This will download a 1.87GB model.

def create_transcription_model(model_id: str = "openai/whisper-large-v3") -> Pipeline:
    """Return a speech to text pipeline."""
    logger.info(f"Creating Whisper model")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

def create_translation_model(source_language: str="pt", target_language: str="en") -> GoogleTranslator:
    """Create the model used for translation."""
    logger.info(f"Creating translator from {source_language} to {target_language}")
    translator = GoogleTranslator(source=source_language, target=target_language)
    return translator

def create_text_to_speech_model() -> TTS:
    """Return the text-to-speech model."""
    logging.info("Creating Coqui text-to-speech model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    return tts

def main():
    """Create each model sequentially."""
    logging.info("Starting main function")
    create_transcription_model()
    create_translation_model()
    _ = create_text_to_speech_model()

if __name__ == "__main__":

    main()