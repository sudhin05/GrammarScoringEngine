from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('all-mpnet-base-v2')

rubric_map = {
    "1.0": "Struggles with sentence structure, errors are frequent.",
    "1.5": "Struggles with sentence structure, but a few sentences are okay.",
    "2.0": "Limited understanding of sentence structure, simple errors.",
    "2.5": "Limited understanding, but manages some decent structure.",
    "3.0": "Decent control of sentence structure, occasional errors.",
    "3.5": "Decent grammar with only small, infrequent issues.",
    "4.0": "Strong control of structure and syntax, minor mistakes.",
    "4.5": "Very strong grammar with near-perfect sentence construction.",
    "5.0": "Excellent control, rarely any noticeable mistakes."
}

rubric_texts = list(rubric_map.values())
rubric_embeddings = model.encode(rubric_texts, convert_to_tensor=True)

transcription = """My favorite place to visit is always beach not just because of the scenery and also because  of the journey. on how will you going to your destination and also the person that  you're coming with the vacation on the beach  I love to go to the beach because there are also local foods that you can try  and I love seeing the sunset and the sunrise especially when"""

transcription_embedding = model.encode(transcription, convert_to_tensor=True)

cos_scores = util.pytorch_cos_sim(transcription_embedding, rubric_embeddings)[0]

best_score_idx = torch.argmax(cos_scores).item()
predicted_score = list(rubric_map.keys())[best_score_idx]

print("📈 Predicted Grammar Score:", predicted_score)
