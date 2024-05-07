import argparse
import logging
import os

from deep_translator import GoogleTranslator
import moviepy.editor as mp
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Pipeline
from TTS.api import TTS

logging.basicConfig(
    filename="main.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

os.environ["COQUI_TOS_AGREED"] = "1" #  Agree Coqui-ai’s xTTS license for using Text-to-speech models. This will download a 1.87GB model.

def convert_mp4_to_wav(video_file: str, wav_file: str) -> str:
    """Write a .wav file from the audio extracted from a video file."""
    logger.info(f"Converting {video_file} to {wav_file}")
    clip = mp.VideoFileClip(video_file)
    clip.audio.write_audiofile(wav_file)

def create_transcription_model(model_id: str = "openai/whisper-large-v3") -> Pipeline:
    """Return a speech to text pipeline."""
    logger.info(f"Creating Whisper model")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def transcript_video_audio(
        pipe: Pipeline,
        input_file: str,
        output_file: str = None,
        source_language: str="portuguese"
        ) -> dict:
    """Create a dictionary with the transcripted speech and its metadata."""
    logger.info(f"Transcribing {input_file} to {output_file}")
    generate_kwargs = dict()
    generate_kwargs['language'] = source_language
    results = pipe(input_file, generate_kwargs)
    with open(output_file, 'w') as transcription_file:
        transcription_file.write(results['text'])

    return results

def create_translation_model(
    source_language: str="pt",
    target_language: str="en"
    ) -> GoogleTranslator:
    """Create the model used for translation."""
    logger.info(f"Creating translator from {source_language} to {target_language}")
    translator = GoogleTranslator(source=source_language, target=target_language)
    return translator

def translate_text(
    text_to_translate: str,
    translator: GoogleTranslator
) -> str:
    """Translate some text using google translator."""
    logger.info("Translating text file")
    if len(text_to_translate) >= 5000:
      chunks = [text_to_translate[i:i+4999] for i in range(0, len(text_to_translate), 4999)]

      translated_text = ""
      for chunk in chunks:
          translated_chunk = translate_text(chunk, translator)
          translated_text += translated_chunk
    else:
      translated_text = translator.translate(text_to_translate)

    return translated_text

def create_text_to_speech_model() -> TTS:
    """Return the text-to-speech model."""
    logging.info("Creating Coqui text-to-speech model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    return tts

def text_to_speech(
    text: str,
    tts_model: TTS,
    audio_output_file: str,
    audio_reference_file: str = None,
    language: str = "en"
) -> None:
    """Make an audio file from a text file. The voice can be cloned from a reference speaker file."""
    logging.info("Transforming text-to-speech.")
    tts_model.tts_to_file(text, speaker_wav=audio_reference_file, language=language, file_path=audio_output_file)

def resize_audio_length(audio_file: str, input_video: str) -> None:
    """Resize the audio length to match the original video."""
    logging.info("Resizing audio length")
    audio = AudioSegment.from_wav(audio_file)
    video = mp.VideoFileClip(input_video)
    speed_factor = audio.duration_seconds / video.duration
    audio_fixed = audio._spawn(audio.raw_data, overrides={'frame_rate':int(audio.frame_rate * speed_factor)})
    audio_fixed.export(audio_file)

def insert_audio_in_video(input_video: str, output_video: str, output_audio: str, fps: int=24) -> None:
    """Insert the .wav audio in the .mp4 video."""
    logging.info(f"Creating {input_video} voiceover.")
    video = mp.VideoFileClip(input_video)
    video = video.set_audio(mp.AudioFileClip(output_audio))
    video.write_videofile(output_video, fps=fps)
    logging.info(f"Translated video {output_video} created.")


def main(args):
    """Translate and dub a video."""
    logger.info("Starting execution")

    input_video = args.input_video
    output_video = args.output_video
    input_name, _ = os.path.splitext(input_video)
    input_audio = os.path.join("audio", input_name + f"_{args.source_language}.wav")
    output_audio = os.path.join("audio", input_name + f"_{args.target_language}.wav")
    source_transcription_file = os.path.join('transcription', input_name + f"_{args.source_language}.txt")
    translated_transcription_file = os.path.join('transcription', input_name + f"_{args.target_language}.txt")

    convert_mp4_to_wav(video_file=input_video, wav_file=input_audio)

    pipe = create_transcription_model()
    transcription_results = transcript_video_audio(
        pipe=pipe,
        input_file=input_audio,
        output_file=source_transcription_file,
        source_language=args.source_language
        )
    source_transcription = transcription_results['text']

    translator = create_translation_model(
        source_language=args.source_language,
        target_language=args.target_language
        )
    translated_text = translate_text(text_to_translate=source_transcription, translator=translator)
    with open(translated_transcription_file, 'w') as translated_file:
        translated_file.write(translated_text)

    tts = create_text_to_speech_model()
    text_to_speech(
        text=translated_text,
        tts_model=tts,
        audio_output_file=output_audio,
        audio_reference_file=input_audio,
        language=args.target_language
        )
    resize_audio_length(audio_file=output_audio, input_video=input_video)
    insert_audio_in_video(
        input_video=input_video,
        output_video=output_video,
        output_audio=output_audio
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Translation and Dubbing')
    parser.add_argument('--input_video', type=str, required=True, help='Input video file')
    parser.add_argument('--output_video', type=str, required=True, help='Output video file')
    parser.add_argument('--source_language', type=str, default="pt", help='Source language of the video')
    parser.add_argument('--target_language', type=str, default="en", help='Target language for translation and dubbing')
    
    parser.epilog = """
    Usage:
    python script.py --input_video input.mp4 --output_video output.mp4
    """
    args = parser.parse_args()
    main(args)
