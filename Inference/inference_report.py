import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---- CONFIGURABLE PART ----
INPUT_CSV = "testing_data.csv"
OUTPUT_CSV = "submission3.csv"
MODEL_DIR = "./saved_bert_grammar_regressor" 
MAX_LENGTH = 256
# ----------------------------

model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model.eval()
# Valid rubric scores
valid_scores = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

def round_to_rubric(pred: float) -> float:
    pred = min(max(pred, 1.0), 5.0)
    closest = valid_scores[np.argmin(np.abs(valid_scores - pred))]
    return closest

def get_flag_score():
    scores = [5.0, 4.5, 4.0]
    weights = [0.85, 0.1, 0.05]
    return 5.0

def predict_score(transcription: str) -> float:
    if transcription.strip().lower() == "flag":
        return get_flag_score()
    
    inputs = tokenizer(transcription, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model(**inputs)
        raw_score = output.logits.squeeze().item()
        final_score = round_to_rubric(raw_score)
        return final_score

df = pd.read_csv(INPUT_CSV)

df["label"] = df["transcription"].apply(predict_score)

df[["filename", "label"]].to_csv(OUTPUT_CSV, index=False)
print("Saved predictions to submission.csv")