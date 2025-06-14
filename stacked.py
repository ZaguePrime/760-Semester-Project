import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import json

# Configs
LANGID_MODEL_NAME = "Davlan/afro-xlmr-base"
AFRISENTI_MODEL_NAME = "Davlan/afro-xlmr-base"
DATA_PATH = "all_languages_train_shuffled.tsv"
MODE = 1  # Set to 1 for regular train/val split, 2 for K-fold cross validation
BATCH_SIZE = 32
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset with pandas
df = pd.read_csv(DATA_PATH, sep="\t")

# Encode language labels
lang_encoder = LabelEncoder()
df["lang_id"] = lang_encoder.fit_transform(df["language"])

# Encode sentiment labels
sentiment_encoder = LabelEncoder()
df["sentiment_id"] = sentiment_encoder.fit_transform(df["label"])

# Save encoders and label mappings
os.makedirs("ensemble_model", exist_ok=True)
joblib.dump(lang_encoder, "ensemble_model/lang_encoder.pkl")
joblib.dump(sentiment_encoder, "ensemble_model/sentiment_encoder.pkl")

# Save label mappings for easier interpretation
lang_labels = {i: label for i, label in enumerate(lang_encoder.classes_)}
sentiment_labels = {i: label for i, label in enumerate(sentiment_encoder.classes_)}

# Create config dictionary for saving
config = {
    'num_langs': len(lang_encoder.classes_),
    'num_sentiment_classes': len(sentiment_encoder.classes_),
    'langid_model_name': LANGID_MODEL_NAME,
    'afrisenti_model_name': AFRISENTI_MODEL_NAME
}

with open("ensemble_model/lang_labels.json", "w") as f:
    json.dump(lang_labels, f)
with open("ensemble_model/sentiment_labels.json", "w") as f:
    json.dump(sentiment_labels, f)
with open("ensemble_model/config.json", "w") as f:
    json.dump(config, f)

# Tokenizers
langid_tokenizer = AutoTokenizer.from_pretrained(LANGID_MODEL_NAME)
sentiment_tokenizer = AutoTokenizer.from_pretrained(AFRISENTI_MODEL_NAME)

# Save tokenizers
langid_tokenizer.save_pretrained("ensemble_model/langid_tokenizer")
sentiment_tokenizer.save_pretrained("ensemble_model/sentiment_tokenizer")

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
        return {
            "langid_input_ids": langid_enc["input_ids"].squeeze(0),
            "langid_attention_mask": langid_enc["attention_mask"].squeeze(0),
            "sentiment_input_ids": sentiment_enc["input_ids"].squeeze(0),
            "sentiment_attention_mask": sentiment_enc["attention_mask"].squeeze(0),
            "lang_labels": torch.tensor(self.lang_labels[idx], dtype=torch.long),
            "sentiment_labels": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }

class StackedDataCollator:
    def __init__(self, langid_tokenizer, sentiment_tokenizer):
        self.langid_collator = DataCollatorWithPadding(langid_tokenizer)
        self.sentiment_collator = DataCollatorWithPadding(sentiment_tokenizer)

    def __call__(self, features):
        langid_features = [{"input_ids": f["langid_input_ids"], "attention_mask": f["langid_attention_mask"]} for f in features]
        sentiment_features = [{"input_ids": f["sentiment_input_ids"], "attention_mask": f["sentiment_attention_mask"]} for f in features]
        lang_labels = torch.stack([f["lang_labels"] for f in features])
        sentiment_labels = torch.stack([f["sentiment_labels"] for f in features])

        langid_batch = self.langid_collator(langid_features)
        sentiment_batch = self.sentiment_collator(sentiment_features)

        return {
            "langid_input_ids": langid_batch["input_ids"],
            "langid_attention_mask": langid_batch["attention_mask"],
            "sentiment_input_ids": sentiment_batch["input_ids"],
            "sentiment_attention_mask": sentiment_batch["attention_mask"],
            "lang_labels": lang_labels,
            "sentiment_labels": sentiment_labels,
        }

class LangIDModel(nn.Module):
    def __init__(self, model_name, num_langs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_langs)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled)
        return logits

class StackedSentimentModel(nn.Module):
    def __init__(self, afrisenti_name, langid_model, num_langs, num_sentiment_classes):
        super().__init__()
        self.langid_model = langid_model
        self.sentiment_encoder = AutoModel.from_pretrained(afrisenti_name)
        hidden_size_sentiment = self.sentiment_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size_sentiment + num_langs, num_sentiment_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentiment_input_ids, sentiment_attention_mask,
                langid_input_ids, langid_attention_mask, **kwargs):
        # Get language predictions
        langid_logits = self.langid_model(langid_input_ids, langid_attention_mask)
        langid_probs = torch.softmax(langid_logits, dim=1)
        
        # Get sentiment features
        sentiment_outputs = self.sentiment_encoder(input_ids=sentiment_input_ids, attention_mask=sentiment_attention_mask)
        sentiment_cls = self.dropout(sentiment_outputs.last_hidden_state[:, 0])
        
        # Combine features
        combined = torch.cat([sentiment_cls, langid_probs], dim=1)
        sentiment_logits = self.classifier(combined)
        
        return {"sentiment_logits": sentiment_logits, "langid_logits": langid_logits}

class StackedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sentiment_labels = inputs.pop("sentiment_labels")
        lang_labels = inputs.pop("lang_labels")

        outputs = model(**inputs)
        sentiment_logits = outputs["sentiment_logits"]
        langid_logits = outputs["langid_logits"]

        loss_fct = nn.CrossEntropyLoss()
        sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
        lang_loss = loss_fct(langid_logits, lang_labels)

        # Weighted combination of losses
        loss = sentiment_loss + 0.3 * lang_loss  # Reduced weight for lang_loss
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Unpack predictions
    sentiment_logits, langid_logits = predictions
    sentiment_preds = np.argmax(sentiment_logits, axis=1)
    lang_preds = np.argmax(langid_logits, axis=1)

    # Unpack labels
    sentiment_labels, lang_labels = labels

    sentiment_acc = accuracy_score(sentiment_labels, sentiment_preds)
    lang_acc = accuracy_score(lang_labels, lang_preds)

    return {
        "sentiment_accuracy": sentiment_acc,
        "language_accuracy": lang_acc,
        "combined_accuracy": (sentiment_acc + lang_acc) / 2
    }

# Split dataset
texts = df["text"].tolist()
lang_labels_list = df["lang_id"].tolist()
sentiment_labels_list = df["sentiment_id"].tolist()

print(f"Running in MODE {MODE}")

if MODE == 1:
    print("Using regular train/validation split...")
    train_texts, val_texts, train_lang, val_lang, train_sent, val_sent = train_test_split(
        texts, lang_labels_list, sentiment_labels_list, test_size=0.2, random_state=42, stratify=sentiment_labels_list
    )

    train_dataset = StackedDataset(train_texts, train_lang, train_sent, langid_tokenizer, sentiment_tokenizer)
    val_dataset = StackedDataset(val_texts, val_lang, val_sent, langid_tokenizer, sentiment_tokenizer)

    collator = StackedDataCollator(langid_tokenizer, sentiment_tokenizer)

    langid_model = LangIDModel(LANGID_MODEL_NAME, len(lang_encoder.classes_)).to(DEVICE)
    ensemble_model = StackedSentimentModel(AFRISENTI_MODEL_NAME, langid_model, len(lang_encoder.classes_), len(sentiment_encoder.classes_)).to(DEVICE)

    training_args = TrainingArguments(
        output_dir="ensemble_model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="combined_accuracy",
        greater_is_better=True,
        fp16=True,
        label_names=["lang_labels", "sentiment_labels"],
        save_total_limit=2,
    )

    trainer = StackedTrainer(
        model=ensemble_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save models
    torch.save(ensemble_model.state_dict(), "ensemble_model/ensemble_model.pt")
    torch.save(langid_model.state_dict(), "ensemble_model/langid_model.pt")
    torch.save({
        'ensemble_model_state_dict': ensemble_model.state_dict(),
        'langid_model_state_dict': langid_model.state_dict(),
        'config': config
    }, "ensemble_model/complete_model.pt")
    
    print("MODE 1 training completed!")

elif MODE == 2:
    print("Using K-Fold Cross Validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    fold_results = []
    
    for train_idx, val_idx in kfold.split(texts):
        print(f"\n--- Fold {fold} ---")
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_lang = [lang_labels_list[i] for i in train_idx]
        val_lang = [lang_labels_list[i] for i in val_idx]
        train_sent = [sentiment_labels_list[i] for i in train_idx]
        val_sent = [sentiment_labels_list[i] for i in val_idx]

        train_dataset = StackedDataset(train_texts, train_lang, train_sent, langid_tokenizer, sentiment_tokenizer)
        val_dataset = StackedDataset(val_texts, val_lang, val_sent, langid_tokenizer, sentiment_tokenizer)

        collator = StackedDataCollator(langid_tokenizer, sentiment_tokenizer)

        langid_model = LangIDModel(LANGID_MODEL_NAME, len(lang_encoder.classes_)).to(DEVICE)
        ensemble_model = StackedSentimentModel(AFRISENTI_MODEL_NAME, langid_model, len(lang_encoder.classes_), len(sentiment_encoder.classes_)).to(DEVICE)

        training_args = TrainingArguments(
            output_dir=f"ensemble_model/fold_{fold}",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="combined_accuracy",
            greater_is_better=True,
            fp16=True,
            label_names=["lang_labels", "sentiment_labels"],
            save_total_limit=2,
        )

        trainer = StackedTrainer(
            model=ensemble_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        result = trainer.train()
        
        # Save fold results
        fold_results.append({
            'fold': fold,
            'final_metrics': trainer.evaluate()
        })

        # Save models for this fold
        torch.save(ensemble_model.state_dict(), f"ensemble_model/fold_{fold}/ensemble_model.pt")
        torch.save(langid_model.state_dict(), f"ensemble_model/fold_{fold}/langid_model.pt")
        torch.save({
            'ensemble_model_state_dict': ensemble_model.state_dict(),
            'langid_model_state_dict': langid_model.state_dict(),
            'config': config,
            'fold': fold
        }, f"ensemble_model/fold_{fold}/complete_model.pt")
        
        fold += 1
    
    # Save cross-validation results
    with open("ensemble_model/cv_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    
    # Calculate and print average metrics
    avg_sentiment_acc = np.mean([result['final_metrics']['eval_sentiment_accuracy'] for result in fold_results])
    avg_lang_acc = np.mean([result['final_metrics']['eval_language_accuracy'] for result in fold_results])
    avg_combined_acc = np.mean([result['final_metrics']['eval_combined_accuracy'] for result in fold_results])
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Average Sentiment Accuracy: {avg_sentiment_acc:.4f}")
    print(f"Average Language Accuracy: {avg_lang_acc:.4f}")
    print(f"Average Combined Accuracy: {avg_combined_acc:.4f}")
    
    print("MODE 2 (K-Fold) training completed!")

else:
    print(f"Invalid MODE: {MODE}. Please set MODE to 1 (regular split) or 2 (K-fold CV)")

print("Training completed!")