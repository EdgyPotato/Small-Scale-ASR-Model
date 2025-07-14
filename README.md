# Small-Scale-ASR-Model
This small scale transformer architecture of automated speech recognition model is based on Alexander Loubser's **End-to-end automated speech recognition using a character based small scale transformer architecture research paper** with modifications.

## Prerequisites
1. Python 3.12.9
2. Install required library (will include `requirements.txt` soon)
    - **(Optional)** Change datasets if needed, currently only using LibriSpeech datasets.
4. Run `main.py` to start training.
5. Run `testing.py` to test trained model.
    - To use custom audio and transcript text, use `python testing.py -a {AUDIO_PATH} -t "Here is your transcript"`

## Reference
Loubser, A., De Villiers, P., & De Freitas, A. (2024). End-to-end automated speech recognition using a character based small scale transformer architecture. Expert Systems With Applications, 252, 124119. https://doi.org/10.1016/j.eswa.2024.124119
