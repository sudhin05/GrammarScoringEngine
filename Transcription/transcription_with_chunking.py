"""TRANSCRIPTION WITH CHUNKING THAT USES FASTER WHISPER"""


import os 
import pandas as pd
import librosa
from faster_whisper import WhisperModel

grammar_rubric = {
    1.0: "The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.",
    1.5: "The person's speech shows signs of basic structure but still has notable issues with grammar and syntax.",
    2.0: "The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.",
    2.5: "The person sometimes forms correct sentences but often makes errors that affect understanding.",
    3.0: "The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.",
    3.5: "The person has mostly correct grammar but makes occasional errors that may slightly affect clarity.",
    4.0: "The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.",
    4.5: "The person speaks accurately most of the time, with small grammar issues that rarely affect clarity.",
    5.0: "Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."
}

label_csv = "Dataset/test.csv"
audio_dir = "Dataset/audios/test/"
output_csv = "testing_data.csv"

df = pd.read_csv(label_csv)
model = WhisperModel("large", compute_type="float16")

rows = []

for idx, row in df.iterrows():
    filename = row["filename"]
    # label = float(row["label"])

    path = os.path.join(audio_dir, filename)
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        continue

    try:
        y, sr = librosa.load(path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration > 70:
            rows.append({
                "filename": filename,
                "transcription": "flag"
            })
            print(f"[FLAG] {filename} duration {duration:.2f}s marked as 'flag'")
            continue

        full_transcription = ""

        # Process in 20s chunks
        chunk_length = 20 * sr
        total_chunks = int(len(y) / chunk_length) + 1

        for i in range(total_chunks):
            start = i * chunk_length
            end = min((i + 1) * chunk_length, len(y))
            chunk = y[start:end]

            if len(chunk) < sr:
                continue

            segments, _ = model.transcribe(chunk, language="en", beam_size=5)
            chunk_text = " ".join([seg.text for seg in segments]).strip()
            full_transcription += " " + chunk_text

        # rubric_desc = grammar_rubric.get(label, "Unknown rubric description.")

        rows.append({
            "filename": filename,
            "transcription": full_transcription.strip(),
        })

        print(f"[OK] Processed {filename}")

    except Exception as e:
        print(f"[ERR] Failed {filename}: {e}")

df_out = pd.DataFrame(rows)
df_out.to_csv(output_csv, index=False)
print(f"\nSaved {len(df_out)} entries to {output_csv}")
