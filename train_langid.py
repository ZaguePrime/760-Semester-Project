import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
import joblib

# === Load TSV ===
df = pd.read_csv("data/combined.tsv", sep="\t")

# === Encode language column ===
lang_encoder = LabelEncoder()
df["label"] = lang_encoder.fit_transform(df["language"])
df = df[["text", "label"]]  # keep only required columns

# === Convert to HuggingFace Dataset ===
dataset = Dataset.from_pandas(df)

# === Tokenize ===
model_name = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# === Save encoders ===
tokenizer.save_pretrained("langid_model/tokenizer")
joblib.dump(lang_encoder, "langid_model/lang_encoder.pkl")

# === Model and Trainer ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(lang_encoder.classes_))
training_args = TrainingArguments(
    output_dir="langid_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_dir="langid_model/logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
trainer.save_model("langid_model/model")
