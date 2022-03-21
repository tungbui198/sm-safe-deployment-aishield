import logging
import os
import sys
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import malaya_speech
from pydub import AudioSegment
from pydub.silence import split_on_silence

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def list_dir(path: Union[str, Path]):
    files = []
    path = Path(path)
    if not path.exists():
        logger.warn(f"{path.as_posix()} doesn't exist.")
    else:
        files = os.listdir(str(path))
    return files


def zero_pad(y, sr):
    if y.shape[0] <= 5*sr:
        pad_width = 5*sr - y.shape[0]
        y = np.pad(y, pad_width=((0, pad_width)), mode='constant')
    return y


def pre_process(fpath: str):
    y, sr = librosa.load(fpath, res_type='kaiser_fast')
    y = librosa.effects.trim(y, top_db=20)[0]
    y_int = malaya_speech.astype.float_to_int(y)
    audio = AudioSegment(
        y_int.tobytes(),
        frame_rate=sr,
        sample_width=y_int.dtype.itemsize,
        channels=1
    )
    audio_chunks = split_on_silence(
        audio,
        min_silence_len=200,
        silence_thresh=-30,
        keep_silence=100,
    )
    y = sum(audio_chunks)
    y = np.array(y.get_array_of_samples())
    y = malaya_speech.astype.int_to_float(y)
    y = zero_pad(y, sr)
    mfcc = librosa.feature.mfcc(y=y[0:5*sr], sr=sr, n_mfcc=13)
    mfcc = mfcc.reshape(-1,)
    return mfcc
