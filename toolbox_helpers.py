# Toolbox Helpers
#
# These are the ONLY primitives available to AI-written toolbox functions.
# This file is written and maintained by you, not by any AI — it's the
# trusted foundation that proposed code is allowed to call instead of
# importing anything itself.
#
# To add a new capability (e.g. something other than sound), add a function
# here, then it becomes something an AI's proposed code can reference by
# name in future proposals.

import wave
import numpy as np
import sounddevice as sd

ASSETS_DIR = "assets"


def play_sound(filename):
    """
    Plays a WAV file from the assets/sounds/ library.
    filename: just the file name, e.g. "cat_meow.wav" — not a full path.
    """
    path = f"{ASSETS_DIR}/sounds/{filename}"
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    audio = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    sd.play(audio, samplerate=framerate)
    sd.wait()