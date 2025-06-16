import os
import sys
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# configuration
BASE_DIR = "../pipeline_model"
SEPARATOR = " </s> "
MODEL_NAME = "Davlan/afro-xlmr-base"

LANG_MODEL_DIR = os.path.join(BASE_DIR, "lang_model")
SENT_MODEL_DIR = os.path.join(BASE_DIR, "sent_model")
LANG_ENCODER_PATH = os.path.join(BASE_DIR, "lang_encoder.pkl")
SENT_ENCODER_PATH = os.path.join(BASE_DIR, "sent_encoder.pkl")

if not all(os.path.exists(p) for p in [LANG_MODEL_DIR, SENT_MODEL_DIR, LANG_ENCODER_PATH, SENT_ENCODER_PATH]):
    print("Model not trained. Please train the model first.")
    sys.exit(1)

lang_encoder = joblib.load(LANG_ENCODER_PATH)
sent_encoder = joblib.load(SENT_ENCODER_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sent_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

lang_model = AutoModelForSequenceClassification.from_pretrained(LANG_MODEL_DIR).to(device)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_DIR).to(device)

# predict function
def predict_pipeline(texts):
    lang_preds, sent_preds = [], []

    for raw_text in texts:
        #Language prediction
        lang_input = lang_tokenizer(raw_text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            lang_logits = lang_model(**lang_input).logits
        pred_lang_id = torch.argmax(lang_logits, dim=-1).item()
        pred_lang = lang_encoder.inverse_transform([pred_lang_id])[0]
        lang_preds.append(pred_lang)

        # Sentiment prediction
        mod_text = f"{pred_lang}{SEPARATOR}{raw_text}"
        sent_input = sent_tokenizer(mod_text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            sent_logits = sent_model(**sent_input).logits
        pred_sent_id = torch.argmax(sent_logits, dim=-1).item()
        pred_sent = sent_encoder.inverse_transform([pred_sent_id])[0]
        sent_preds.append(pred_sent)

    return pd.DataFrame({
        "text": texts,
        "predicted_language": lang_preds,
        "predicted_sentiment": sent_preds
    })

# =============================================
# add text to predict here
# note: this will be slow to run as the models have to first be load everytime the script gets run
# =============================================
if __name__ == "__main__":
    sample_texts = [
        "በራስ ለማስተዳደር በፉከራ የሚመጣ ነገር የለም፤ በቀረርቶም የሚሆን ነገር የለም! ተግቶ መስራት ብቻ ነው መፍቴያችን፡፡ ...... ኢዜማ የሁገራችንን ችግር ይፈታሉ ብሎ ባዘጋጃቸው 45 ፖሊሲዎ…",
        "@user Nwanna, ọ dịka ị ba nyẹ gọ ụka akụlụ aka?",
        "@user Akụkọ Mike ejagha"
    ]
    
    results_df = predict_pipeline(sample_texts)
    print(results_df.to_string(index=False))
