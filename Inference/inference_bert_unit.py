import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

save_dir = "./saved_bert_grammar_regressor"

model = BertForSequenceClassification.from_pretrained(save_dir)
tokenizer = BertTokenizer.from_pretrained(save_dir)
model.eval()

# Valid rubric scores
valid_scores = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

def round_to_rubric(pred: float) -> float:
    pred = min(max(pred, 1.0), 5.0)  
    closest = valid_scores[np.argmin(np.abs(valid_scores - pred))]
    return closest

def predict_score(transcription: str) -> float:
    inputs = tokenizer(
        transcription,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        output = model(**inputs)
        raw_score = output.logits.squeeze().item()
        final_score = round_to_rubric(raw_score)
        return final_score


txt = "My goal is I become an entrepreneur because I'm not working under anyone.  I think to provide work to everyone because my education is not to work from me under anyone try to provide work to everyone and  and develop myself  providing work is very good  to in the world because giving work is a good help to everyone this is my long-term goal to become an entrepreneur thank you"

print("Predicted Score:", predict_score(txt))

