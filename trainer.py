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
SAVE_DIR = "lang_id_model"
EPOCHS = 3
BATCH_SIZE = 64

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# --- LOAD & ENCODE DATA ---
df = pd.read_csv(DATA_PATH, sep='\t')
lang_encoder = LabelEncoder()
df['lang_id'] = lang_encoder.fit_transform(df['language'])

os.makedirs(SAVE_DIR, exist_ok=True)
joblib.dump(lang_encoder, f"{SAVE_DIR}/lang_encoder.pkl")

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- DATASET ---
class LangIDDataset(Dataset):
    def __init__(self, texts, lang_labels):
        self.texts = texts
        self.lang_labels = lang_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(self.texts[idx], truncation=True, padding=False, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.lang_labels[idx]),
        }

train_dataset = LangIDDataset(
    df['text'].tolist(),
    df['lang_id'].tolist()
)

# --- MODEL ---
class LangIDModel(nn.Module):
    def __init__(self, model_name, num_langs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_langs)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

# --- CUSTOM TRAINER ---
class LangIDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="no",
    save_strategy="epoch",
    logging_dir=os.path.join(SAVE_DIR, "logs"),
    logging_steps=50,
    save_total_limit=1,
    fp16=True,
)

# --- Custom Collator ---

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = super().__call__(features)  # pad input_ids and attention_mask
        batch["labels"] = torch.stack(labels)
        return batch


# --- SETUP MODEL ---
model = LangIDModel(
    MODEL_NAME,
    num_langs=len(lang_encoder.classes_)
).to(device)



collator = CustomDataCollator(tokenizer)

trainer = LangIDTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collator
)

# --- TRAIN ---
trainer.train()

# --- SAVE MODEL + TOKENIZER ---
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Training complete. Model saved to {SAVE_DIR}")
