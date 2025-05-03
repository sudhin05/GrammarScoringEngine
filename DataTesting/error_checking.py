import os
import librosa
import logging
from IPython.display import Audio

logging.basicConfig(level=logging.WARNING,filename="flag2.log",filemode='w')
flag = []
aud_dir = "Dataset/audios/test"

for a in sorted(os.listdir(aud_dir)):
    apath = os.path.join(aud_dir,a)
    sr = librosa.get_samplerate(apath)
    dur = librosa.get_duration(path=apath)
    if dur > 120:
        logging.warning(f"Flagged: {a}")
