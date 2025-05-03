import pandas as pd
from datasets import Dataset
from transformers import Trainer, DataCollatorWithPadding
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments


df = pd.read_csv("training_data_for_t5.csv")  

df = df.dropna(subset=["transcription", "label"])

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)


tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
    problem_type="regression"  
)

def preprocess(batch):
    tokens = tokenizer(
        batch["transcription"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokens["labels"] = [float(l) for l in batch["label"]]
    return tokens

tokenized = dataset.map(preprocess, batched=True)


training_args = TrainingArguments(
    output_dir="./bert_grammar_regressor",
    num_train_epochs=50,             
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,            
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    weight_decay=0.01
)


# Data collator to pad 
data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

save_dir = "./saved_bert_grammar_regressor"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")
