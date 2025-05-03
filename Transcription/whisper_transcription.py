import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.generation_config.language = "<|en|>"
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")

audio_path = "Dataset/audios/train/audio_1002.wav"
target_sr = 16000
chunk_duration = 25  
stride_duration = 0  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

y, sr = librosa.load(audio_path, sr=target_sr)
total_duration = len(y) / target_sr

chunk_size = chunk_duration * target_sr
stride_size = stride_duration * target_sr
chunks = []
for i in range(0, len(y), chunk_size - stride_size):
    chunk = y[i: i + chunk_size]
    if len(chunk) < 1000:
        break
    chunks.append(chunk)

full_transcript = ""
for idx, chunk in enumerate(chunks):
    inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    full_transcript += " " + text.strip()

print("Transcription:\n")
print(full_transcript.strip())
