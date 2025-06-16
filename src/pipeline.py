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
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Paths
BASE_DIR = "../pipeline_model"
os.makedirs(BASE_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv("../language_datasets/all_languages_train_shuffled.tsv", sep="\t")
df = df[["text", "label", "language"]].dropna()

# Setup train-test split (80/20)
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df["label"]  # Stratify by sentiment label to maintain class balance
)

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Global encoders
lang_encoder = LabelEncoder()
lang_encoder.fit(df["language"])
sent_encoder = LabelEncoder()
sent_encoder.fit(df["label"])

# Save encoders for later use
joblib.dump(lang_encoder, os.path.join(BASE_DIR, "lang_encoder.pkl"))
joblib.dump(sent_encoder, os.path.join(BASE_DIR, "sent_encoder.pkl"))

# Model base
model_name = "Davlan/afro-xlmr-base"

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

print("\n==== Training Language Identification Model ====")

# Language DataPrep
lang_train_df = train_df[["text", "language"]].copy()
lang_test_df = test_df[["text", "language"]].copy()
lang_train_df["label"] = lang_encoder.transform(lang_train_df["language"])
lang_test_df["label"] = lang_encoder.transform(lang_test_df["language"])

lang_tokenizer = AutoTokenizer.from_pretrained(model_name)
lang_train_ds = Dataset.from_pandas(lang_train_df)
lang_test_ds = Dataset.from_pandas(lang_test_df)
lang_train_ds = lang_train_ds.map(lambda x: lang_tokenizer(x["text"], truncation=True), batched=True)
lang_test_ds = lang_test_ds.map(lambda x: lang_tokenizer(x["text"], truncation=True), batched=True)

lang_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(lang_encoder.classes_)
)
lang_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "lang_model"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    logging_dir=os.path.join(BASE_DIR, "lang_logs"),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2
)
lang_trainer = Trainer(
    model=lang_model,
    args=lang_args,
    train_dataset=lang_train_ds,
    eval_dataset=lang_test_ds,
    tokenizer=lang_tokenizer,
    data_collator=DataCollatorWithPadding(lang_tokenizer),
    compute_metrics=compute_metrics
)
lang_trainer.train()

print("\n==== Training Sentiment Analysis Model ====")

# Sentiment DataPrep
sent_train_df = train_df.copy()
sent_test_df = test_df.copy()
sent_train_df["label"] = sent_encoder.transform(sent_train_df["label"])
sent_test_df["label"] = sent_encoder.transform(sent_test_df["label"])
# Prepend language information to text
sent_train_df["text"] = sent_train_df["language"] + " [SEP] " + sent_train_df["text"]
sent_test_df["text"] = sent_test_df["language"] + " [SEP] " + sent_test_df["text"]

sent_tokenizer = AutoTokenizer.from_pretrained(model_name)
sent_train_ds = Dataset.from_pandas(sent_train_df)
sent_test_ds = Dataset.from_pandas(sent_test_df)
sent_train_ds = sent_train_ds.map(lambda x: sent_tokenizer(x["text"], truncation=True), batched=True)
sent_test_ds = sent_test_ds.map(lambda x: sent_tokenizer(x["text"], truncation=True), batched=True)

sent_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(sent_encoder.classes_)
)
sent_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "sent_model"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir=os.path.join(BASE_DIR, "sent_logs"),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2
)
sent_trainer = Trainer(
    model=sent_model,
    args=sent_args,
    train_dataset=sent_train_ds,
    eval_dataset=sent_test_ds,
    tokenizer=sent_tokenizer,
    data_collator=DataCollatorWithPadding(sent_tokenizer),
    compute_metrics=compute_metrics
)
sent_trainer.train()

print("\n==== Evaluating Full Pipeline ====")

# Evaluate pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang_model.to(device)
sent_model.to(device)

lang_preds, sent_preds, lang_true, sent_true = [], [], [], []

for i in range(len(test_df)):
    raw_text = test_df.iloc[i]["text"]
    true_lang = test_df.iloc[i]["language"]
    true_sent = test_df.iloc[i]["label"]

    # Step 1: Language identification
    lang_input = lang_tokenizer(raw_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        lang_logits = lang_model(**lang_input).logits
    pred_lang_id = lang_logits.argmax(dim=-1).item()
    pred_lang = lang_encoder.inverse_transform([pred_lang_id])[0]

    # Step 2: Sentiment analysis with predicted language
    new_text = f"{pred_lang} [SEP] {raw_text}"
    sent_input = sent_tokenizer(new_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        sent_logits = sent_model(**sent_input).logits
    pred_sent_id = sent_logits.argmax(dim=-1).item()

    # Store predictions and true labels
    lang_preds.append(pred_lang_id)
    lang_true.append(lang_encoder.transform([true_lang])[0])
    sent_preds.append(pred_sent_id)
    sent_true.append(sent_encoder.transform([true_sent])[0])

    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(test_df)} test samples")

# === Final Results ===
print("\n=== Final Pipeline Results ===")
print(f"Test set size: {len(test_df)} samples")

print("\n-- Language Identification Performance --")
lang_accuracy = accuracy_score(lang_true, lang_preds)
print(f"Accuracy: {lang_accuracy:.4f}")
print("\nDetailed Report:")
print(classification_report(lang_true, lang_preds, target_names=lang_encoder.classes_, zero_division=0))

print("\n-- Sentiment Analysis Performance --")
sent_accuracy = accuracy_score(sent_true, sent_preds)
print(f"Accuracy: {sent_accuracy:.4f}")
print("\nDetailed Report:")
print(classification_report(sent_true, sent_preds, target_names=sent_encoder.classes_, zero_division=0))

print(f"\n-- Overall Pipeline Performance --")
print(f"Language ID Accuracy: {lang_accuracy:.4f}")
print(f"Sentiment Analysis Accuracy: {sent_accuracy:.4f}")

# Save results
results = {
    'lang_accuracy': lang_accuracy,
    'sent_accuracy': sent_accuracy,
    'lang_preds': lang_preds,
    'lang_true': lang_true,
    'sent_preds': sent_preds,
    'sent_true': sent_true
}
joblib.dump(results, os.path.join(BASE_DIR, "pipeline_results.pkl"))
print(f"\nResults saved to {BASE_DIR}/pipeline_results.pkl")



print("\n-- Plotting ROC-AUC for Sentiment Classes --")

# Get number of classes
n_classes = len(sent_encoder.classes_)

# Binarize true labels
sent_true_bin = label_binarize(sent_true, classes=range(n_classes))

# Convert logits list to array
sent_logits_array = []
with torch.no_grad():
    for i in range(len(test_df)):
        input_text = f"{lang_encoder.inverse_transform([lang_preds[i]])[0]} [SEP] {test_df.iloc[i]['text']}"
        inputs = sent_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
        logits = sent_model(**inputs).logits
        sent_logits_array.append(logits.detach().cpu().numpy().flatten())
sent_logits_array = np.array(sent_logits_array)


# Plot ROC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(sent_true_bin[:, i], sent_logits_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{sent_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Sentiment ROC Curve per Class')
plt.legend(loc="lower right")
plt.grid(True)

roc_path = os.path.join(BASE_DIR, "sentiment_roc_per_class.png")
plt.savefig(roc_path)
print(f"ROC curves saved to: {roc_path}")