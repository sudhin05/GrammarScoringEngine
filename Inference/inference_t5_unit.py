import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_dir = "./saved_t5_grammar_model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

model.eval()

# Inference example
transcription = "Uh, let's see, I'm at Eastern Market. Um, the place is crowded. They're selling everything from clothing to jewelry to shoes. Um, there's street vendors here and they're selling hot dogs, hamburgers, sliders, fish sandwiches, Mexican food. Chinese food. It's pretty loud here so I have to like talk over the mic and  in the morning it's a pretty relaxed crowd mostly old people looking for some  great deals and then as it transitions into the evening you get more of the  urban crowd. single moms kids teenagers and then by night you get the rowdy a bunch the people coming home from  the bar um who just want to eat some of this delicious street food i don't know i would say  that the eastern market is a very good market you should come here when you get the chance"

input_text = f"Transcription: {transcription} \nRubric: Rate grammar from 1 to 5. Use .5 if in-between."
input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids

with torch.no_grad():
    outputs = model.generate(input_ids, max_length=5)

predicted_score = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Predicted Grammar Score:", predicted_score)
