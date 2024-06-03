# Video Translation

This project aims to provide a quick integration for translating video voices using open-source tools. The audio is extracted from the video into a text file, which is then translated into the desired language, and finally transformed back into audio for inputing on the video.

### Disclaimers :warning:

Due to hardware limitations, the output files were not generated from the executor in the repository. The main function was reproduced in a Google Colab notebook, which allows the usage of GPU resources for running the deep learning models.

Installing Coqui TTS from the main source was resulting in a metadata-generation-failed, so I've downloaded the model from a fork repository.
[Building the TTS package](https://github.com/coqui-ai/TTS/discussions/3705)

### Tools :hammer:

* Whisper: Automatic speech recognition
* deep-translator: Text translation
* Coqui TTS: Voice cloning and text-to-speech

### Usage :writing_hand:

The first step is to create the execution environment.

```
conda create env --your_env_name python=3.10.14
```

Then you can download the required packages by running:

```
pip install -r requirements.txt
```

You can see the execution template below

```
usage: main.py [-h] --input_video INPUT_VIDEO --output_video OUTPUT_VIDEO [--source_language SOURCE_LANGUAGE] [--target_language TARGET_LANGUAGE]
```

As well as an execution example
```
python main.py --input_video case_ai.mp4 --output_video case_ai_en.mp4 --source_language pt --target_language en
```

The first execution may take more time than usual, as the deep learning models have to be downloaded. If you want to download the models before executing the main file, you can run

```
python create_models.py
```

so that when you don't overestimate the traslation and execution time.

#### Running on Colab

Colab natively supports Tensorflow, whose dependencies are incompatible with this project requirements. If you're running this project on Colab, you first need to uninstall tensorflow before installing the dependencies.

```
!pip uninstall tensorflow
!pip install -r requirements.txt
```
