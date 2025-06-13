import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder

# Configs
LANGID_MODEL_NAME = "Davlan/afro-xlmr-base"
AFRISENTI_MODEL_NAME = "Davlan/afrisenti-twitter-sentiment-afroxlmr-large"
DATA_PATH = "all_languages_train_shuffled.tsv"
BATCH_SIZE = 32
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset with pandas
df = pd.read_csv(DATA_PATH, sep="\t")

# Assume your dataset has columns: "text", "language", "sentiment"
# Adapt column names if needed

# Encode language labels
lang_encoder = LabelEncoder()
df["lang_id"] = lang_encoder.fit_transform(df["language"])

# Encode sentiment labels
sentiment_encoder = LabelEncoder()
df["sentiment_id"] = sentiment_encoder.fit_transform(df["label"])

# Save encoders if needed
os.makedirs("ensemble_model", exist_ok=True)
import joblib
joblib.dump(lang_encoder, "ensemble_model/lang_encoder.pkl")
joblib.dump(sentiment_encoder, "ensemble_model/sentiment_encoder.pkl")

# Tokenizers
langid_tokenizer = AutoTokenizer.from_pretrained(LANGID_MODEL_NAME)
sentiment_tokenizer = AutoTokenizer.from_pretrained(AFRISENTI_MODEL_NAME)

# Dataset class (same as before)
class StackedDataset(Dataset):
    def __init__(self, texts, lang_labels, sentiment_labels, langid_tokenizer, sentiment_tokenizer):
        self.texts = texts
        self.lang_labels = lang_labels
        self.sentiment_labels = sentiment_labels
        self.langid_tokenizer = langid_tokenizer
        self.sentiment_tokenizer = sentiment_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        langid_enc = self.langid_tokenizer(text, truncation=True, padding=False, return_tensors="pt")
        sentiment_enc = self.sentiment_tokenizer(text, truncation=True, padding=False, return_tensors="pt")

        out = {
            "langid_input_ids": langid_enc["input_ids"].squeeze(0),
            "langid_attention_mask": langid_enc["attention_mask"].squeeze(0),
            "sentiment_input_ids": sentiment_enc["input_ids"].squeeze(0),
            "sentiment_attention_mask": sentiment_enc["attention_mask"].squeeze(0),
            "lang_labels": torch.tensor(self.lang_labels[idx], dtype=torch.long),
            "sentiment_labels": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }
        print(out.keys())  # Should print all 6 keys every time
        print('THE FUCKING KEYS')
        print(out)
        return out


# Collator class (same as before)
class StackedDataCollator:
    def __init__(self, langid_tokenizer, sentiment_tokenizer):
        self.langid_collator = DataCollatorWithPadding(langid_tokenizer)
        self.sentiment_collator = DataCollatorWithPadding(sentiment_tokenizer)

    def __call__(self, features):
        print("Batch feature keys sample:", features[0].keys())
        langid_features = [{"input_ids": f["langid_input_ids"], "attention_mask": f["langid_attention_mask"]} for f in features]
        sentiment_features = [{"input_ids": f["sentiment_input_ids"], "attention_mask": f["sentiment_attention_mask"]} for f in features]
        lang_labels = torch.stack([f["lang_labels"] for f in features])
        sentiment_labels = torch.stack([f["sentiment_labels"] for f in features])

        langid_batch = self.langid_collator(langid_features)
        sentiment_batch = self.sentiment_collator(sentiment_features)

        batch = {
            "langid_input_ids": langid_batch["input_ids"],
            "langid_attention_mask": langid_batch["attention_mask"],
            "sentiment_input_ids": sentiment_batch["input_ids"],
            "sentiment_attention_mask": sentiment_batch["attention_mask"],
            "lang_labels": lang_labels,
            "sentiment_labels": sentiment_labels,
        }
        return batch

# Models (same as before)
class LangIDModel(nn.Module):
    def __init__(self, model_name, num_langs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_langs)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled)
        return logits

class StackedSentimentModel(nn.Module):
    def __init__(self, afrisenti_name, langid_model, num_langs, num_sentiment_classes):
        super().__init__()
        self.langid_model = langid_model
        self.sentiment_encoder = AutoModel.from_pretrained(afrisenti_name)
        hidden_size_sentiment = self.sentiment_encoder.config.hidden_size
        self.langid_num_classes = num_langs
        self.classifier = nn.Linear(hidden_size_sentiment + num_langs, num_sentiment_classes)

    def forward(self, sentiment_input_ids, sentiment_attention_mask,
                langid_input_ids, langid_attention_mask):
        langid_logits = self.langid_model(langid_input_ids, langid_attention_mask)
        langid_probs = torch.softmax(langid_logits, dim=1)
        sentiment_outputs = self.sentiment_encoder(input_ids=sentiment_input_ids, attention_mask=sentiment_attention_mask)
        sentiment_cls = sentiment_outputs.last_hidden_state[:, 0]
        combined = torch.cat([sentiment_cls, langid_probs], dim=1)
        logits = self.classifier(combined)
        return logits

from transformers import Trainer

class StackedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("sentiment_labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# Prepare dataset from real data
texts = df["text"].tolist()
print(f'------------------\n{len(texts)}')
lang_labels = df["lang_id"].tolist()
print(f'------------------\n{len(lang_labels)}')
sentiment_labels = df["sentiment_id"].tolist()
print(f'------------------\n{len(sentiment_labels)}')

# exit()

dataset = StackedDataset(texts, lang_labels, sentiment_labels, langid_tokenizer, sentiment_tokenizer)
collator = StackedDataCollator(langid_tokenizer, sentiment_tokenizer)

num_langs = len(lang_encoder.classes_)
num_sentiment_classes = len(sentiment_encoder.classes_)

# Load your trained LangIDModel weights here if you have them
langid_model = LangIDModel(LANGID_MODEL_NAME, num_langs)
# langid_model.load_state_dict(torch.load("path_to_langid_weights.pt"))
langid_model.to(DEVICE)
langid_model.eval()  # freeze during ensemble training if desired

ensemble_model = StackedSentimentModel(AFRISENTI_MODEL_NAME, langid_model, num_langs, num_sentiment_classes)
ensemble_model.to(DEVICE)

training_args = TrainingArguments(
    output_dir="ensemble_model",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    fp16=True,
    label_names=["lang_labels", "sentiment_labels"],
)

trainer = StackedTrainer(
    model=ensemble_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()