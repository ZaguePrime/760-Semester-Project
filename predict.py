import torch
import joblib
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel
import os

# --- CONFIG ---
MODEL_NAME = "Davlan/afro-xlmr-base"
SAVE_DIR = "lang_id_model"
TEXT = "Waliomua kinyama mtoto wasakwa gt"  # Change this to whatever text you want to test

# --- MODEL DEFINITION ---
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

# --- LOAD TOKENIZER + LABEL ENCODER ---
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
lang_encoder = joblib.load(os.path.join(SAVE_DIR, "lang_encoder.pkl"))

# --- LOAD MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LangIDModel(MODEL_NAME, num_langs=len(lang_encoder.classes_))
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "pytorch_model.bin"), map_location=device))
model.to(device)
model.eval()

# --- PREDICT FUNCTION ---
def predict_language(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()
        return lang_encoder.inverse_transform([pred_id])[0]

# --- RUN PREDICTION ---
predicted_lang = predict_language(TEXT)
print(f"🧠 Predicted Language: {predicted_lang}")
