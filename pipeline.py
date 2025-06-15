import os
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset

# Paths
LANG_MODEL_DIR = "langid_model"
SENT_MODEL_DIR = "sentiment_model"
os.makedirs(LANG_MODEL_DIR, exist_ok=True)
os.makedirs(SENT_MODEL_DIR, exist_ok=True)

# === 1. Load and Split Dataset ===
df = pd.read_csv("all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

# Shared train/test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# === 2. Language ID Model ===
lang_train_df = train_df[["text", "language"]].copy()
lang_test_df = test_df[["text", "language"]].copy()

lang_encoder = LabelEncoder()
lang_train_df["label"] = lang_encoder.fit_transform(lang_train_df["language"])
lang_test_df["label"] = lang_encoder.transform(lang_test_df["language"])

joblib.dump(lang_encoder, os.path.join(LANG_MODEL_DIR, "lang_encoder.pkl"))

lang_train_ds = Dataset.from_pandas(lang_train_df)
lang_test_ds = Dataset.from_pandas(lang_test_df)

lang_model_name = "Davlan/afro-xlmr-base"
lang_tokenizer = AutoTokenizer.from_pretrained(lang_model_name)

lang_train_ds = lang_train_ds.map(lambda x: lang_tokenizer(x["text"], truncation=True), batched=True)
lang_test_ds = lang_test_ds.map(lambda x: lang_tokenizer(x["text"], truncation=True), batched=True)

lang_model = AutoModelForSequenceClassification.from_pretrained(
    lang_model_name, num_labels=len(lang_encoder.classes_)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

lang_args = TrainingArguments(
    output_dir=LANG_MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    num_train_epochs=1,
    load_best_model_at_end=True,
    logging_dir=os.path.join(LANG_MODEL_DIR, "logs")
)

lang_trainer = Trainer(
    model=lang_model,
    args=lang_args,
    train_dataset=lang_train_ds,
    eval_dataset=lang_test_ds,
    tokenizer=lang_tokenizer,
    data_collator=DataCollatorWithPadding(lang_tokenizer),
    compute_metrics=compute_metrics,
)

lang_trainer.train()
lang_trainer.save_model(os.path.join(LANG_MODEL_DIR, "model"))
lang_tokenizer.save_pretrained(os.path.join(LANG_MODEL_DIR, "tokenizer"))

# === 3. Sentiment Model ===
sent_train_df = train_df.copy()
sent_test_df = test_df.copy()

# Save original sentiment encoder
sent_encoder = LabelEncoder()
sent_train_df["label"] = sent_encoder.fit_transform(sent_train_df["label"])
sent_test_df["label"] = sent_encoder.transform(sent_test_df["label"])
joblib.dump(sent_encoder, os.path.join(SENT_MODEL_DIR, "sentiment_encoder.pkl"))

# Prepend language to text
sent_train_df["text"] = sent_train_df["language"] + " [SEP] " + sent_train_df["text"]
sent_test_df["text"] = sent_test_df["language"] + " [SEP] " + sent_test_df["text"]

sent_train_ds = Dataset.from_pandas(sent_train_df)
sent_test_ds = Dataset.from_pandas(sent_test_df)

sent_tokenizer = AutoTokenizer.from_pretrained(lang_model_name)

sent_train_ds = sent_train_ds.map(lambda x: sent_tokenizer(x["text"], truncation=True), batched=True)
sent_test_ds = sent_test_ds.map(lambda x: sent_tokenizer(x["text"], truncation=True), batched=True)

sent_model = AutoModelForSequenceClassification.from_pretrained(
    lang_model_name, num_labels=len(sent_encoder.classes_)
)

sent_args = TrainingArguments(
    output_dir=SENT_MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    num_train_epochs=3,
    load_best_model_at_end=True,
    logging_dir=os.path.join(SENT_MODEL_DIR, "logs")
)

sent_trainer = Trainer(
    model=sent_model,
    args=sent_args,
    train_dataset=sent_train_ds,
    eval_dataset=sent_test_ds,
    tokenizer=sent_tokenizer,
    data_collator=DataCollatorWithPadding(sent_tokenizer),
    compute_metrics=compute_metrics,
)

sent_trainer.train()
sent_trainer.save_model(os.path.join(SENT_MODEL_DIR, "model"))
sent_tokenizer.save_pretrained(os.path.join(SENT_MODEL_DIR, "tokenizer"))


# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang_model.to(device)
sent_model.to(device)

# === Pipeline Evaluation ===
print("=== Pipeline Evaluation ===")
lang_preds, sentiment_preds, true_sentiments, true_languages = [], [], [], []

for i in range(len(test_df)):
    # Get the original raw text (without any [SEP] modifications)
    raw_text = test_df.iloc[i]["text"]
    true_sentiment_label = sent_encoder.transform([test_df.iloc[i]["label"]])[0]  # Encode the sentiment label
    true_language = test_df.iloc[i]["language"]

    # === Step 1: Language ID Prediction ===
    lang_inputs = lang_tokenizer(raw_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        lang_logits = lang_model(**lang_inputs).logits
    predicted_lang_id = lang_logits.argmax(dim=-1).item()
    predicted_language = lang_encoder.inverse_transform([predicted_lang_id])[0]

    # === Step 2: Format Input for Sentiment Model ===
    new_input_text = f"{predicted_language} [SEP] {raw_text}"
    sent_inputs = sent_tokenizer(new_input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        sentiment_logits = sent_model(**sent_inputs).logits
    predicted_sentiment_id = sentiment_logits.argmax(dim=-1).item()

    # === Collect Results ===
    lang_preds.append(predicted_lang_id)
    sentiment_preds.append(predicted_sentiment_id)
    true_sentiments.append(true_sentiment_label)
    true_languages.append(lang_encoder.transform([true_language])[0])  # Convert to encoded form

# === Evaluation Metrics ===
print("\n=== Language ID Results ===")
lang_accuracy = accuracy_score(true_languages, lang_preds)
print(f"Language ID Accuracy: {lang_accuracy:.4f}")
print("\nLanguage ID Classification Report:")
print(classification_report(true_languages, lang_preds, 
                          target_names=lang_encoder.classes_, zero_division=0))

print("\n=== Sentiment Analysis Results ===")
sent_accuracy = accuracy_score(true_sentiments, sentiment_preds)
print(f"Sentiment Analysis Accuracy: {sent_accuracy:.4f}")
print("\nSentiment Classification Report:")
print(classification_report(true_sentiments, sentiment_preds, 
                          target_names=sent_encoder.classes_, zero_division=0))

print(f"\n=== Overall Pipeline Performance ===")
print(f"Language ID Accuracy: {lang_accuracy:.4f}")
print(f"Sentiment Analysis Accuracy: {sent_accuracy:.4f}")