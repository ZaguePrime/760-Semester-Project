import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- CONFIG ---
MODEL_NAME = "Davlan/afro-xlmr-base"
DATA_PATH = "all_languages_train_shuffled.tsv"
SAVE_DIR = "multitask_afrosenti_model"
EPOCHS = 3
BATCH_SIZE = 4

# --- LOAD & ENCODE DATA ---
df = pd.read_csv(DATA_PATH, sep='\t')
lang_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

df['lang_id'] = lang_encoder.fit_transform(df['language'])
df['sentiment_id'] = sentiment_encoder.fit_transform(df['label'])

# --- SAVE ENCODERS ---
os.makedirs(SAVE_DIR, exist_ok=True)
joblib.dump(lang_encoder, f"{SAVE_DIR}/lang_encoder.pkl")
joblib.dump(sentiment_encoder, f"{SAVE_DIR}/sentiment_encoder.pkl")

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- DATASET ---
class MultiTaskDataset(Dataset):
    def __init__(self, texts, lang_labels, sentiment_labels):
        self.texts = texts
        self.lang_labels = lang_labels
        self.sentiment_labels = sentiment_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(self.texts[idx], truncation=True, padding=False, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "lang_label": torch.tensor(self.lang_labels[idx]),
            "sentiment_label": torch.tensor(self.sentiment_labels[idx]),
        }

train_dataset = MultiTaskDataset(
    df['text'].tolist(),
    df['lang_id'].tolist(),
    df['sentiment_id'].tolist()
)

# --- MODEL ---
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_langs, num_sentiments):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.lang_classifier = nn.Linear(hidden_size, num_langs)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiments)

    def forward(self, input_ids, attention_mask, lang_label=None, sentiment_label=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        lang_logits = self.lang_classifier(pooled)
        sentiment_logits = self.sentiment_classifier(pooled)
        return {
            "loss": None,
            "lang_logits": lang_logits,
            "sentiment_logits": sentiment_logits,
            "lang_label": lang_label,
            "sentiment_label": sentiment_label,
        }

# --- CUSTOM TRAINER ---
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lang_labels = inputs.pop("lang_label")
        sentiment_labels = inputs.pop("sentiment_label")
        outputs = model(**inputs)

        lang_loss = nn.CrossEntropyLoss()(outputs["lang_logits"], lang_labels)
        sentiment_loss = nn.CrossEntropyLoss()(outputs["sentiment_logits"], sentiment_labels)
        total_loss = lang_loss + sentiment_loss

        return (total_loss, outputs) if return_outputs else total_loss


# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="no",
    save_strategy="epoch",
    logging_dir=os.path.join(SAVE_DIR, "logs"),
    logging_steps=50,
    save_total_limit=1
)

# --- FINAL SETUP ---
model = MultiTaskModel(
    MODEL_NAME,
    num_langs=len(lang_encoder.classes_),
    num_sentiments=len(sentiment_encoder.classes_)
)

collator = DataCollatorWithPadding(tokenizer)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collator
)

# --- TRAIN ---
trainer.train()

# --- SAVE MODEL + TOKENIZER ---
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Training complete. Model saved to {SAVE_DIR}")
