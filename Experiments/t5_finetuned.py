import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

df = pd.read_csv("training_data_for_t5.csv")

# Format: Input = transcription + rubric, Target = grammar score
df = df.dropna(subset=["transcription", "rubric_description", "label"])
df["input_text"] = df.apply(lambda x: f"Transcription: {x['transcription']}", axis=1)
df["target_text"] = df["label"].astype(str)

#to HuggingFace Dataset
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

dataset = dataset.train_test_split(test_size=0.1)

model_name = "t5-large"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# apply tokenizer
def preprocess(example):
    model_input = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=10)
    model_input["labels"] = labels["input_ids"]
    return model_input

tokenized_dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./t5_grammar_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs"
)

# Data collator to pad
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

save_dir = "./saved_t5_grammar_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model saved to {save_dir}")
