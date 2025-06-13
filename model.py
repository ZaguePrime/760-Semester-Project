import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
LANGUAGE_MODEL_NAME = "Davlan/afro-xlmr-base"  # Your trained model
SENTIMENT_MODEL_NAME = "davlan/afrisenti-twitter-sentiment-afroxlmr-large"
DATA_PATH = "all_languages_train_shuffled.tsv"
SAVE_DIR = "stacking_ensemble_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS = 5
BATCH_SIZE = 32

print(f"🚀 Using device: {DEVICE}")

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH, sep='\t')
print(f"📊 Loaded {len(df)} samples")

# --- LANGUAGE IDENTIFICATION MODEL (Level 0) ---
class LanguageIdentifierModel(nn.Module):
    """Your existing multitask model adapted for language identification"""
    def __init__(self, model_name, num_langs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.lang_classifier = nn.Linear(hidden_size, num_langs)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        lang_logits = self.lang_classifier(pooled)
        return torch.softmax(lang_logits, dim=-1)

# --- SENTIMENT MODEL WRAPPER (Level 0) ---
class SentimentModelWrapper:
    """Wrapper for pre-trained sentiment analysis model"""
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get sentiment probabilities for a list of texts"""
        probabilities = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                probabilities.append(probs)
        
        return np.array(probabilities)

# --- FEATURE EXTRACTION DATASET ---
class FeatureExtractionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
        }

# --- STACKING ENSEMBLE CLASS ---
class StackingEnsemble:
    def __init__(self, language_model_path=None, sentiment_model_name=SENTIMENT_MODEL_NAME):
        self.language_model_path = language_model_path
        self.sentiment_model_name = sentiment_model_name
        self.lang_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
        
        # Initialize models
        self.language_model = None
        self.sentiment_model = SentimentModelWrapper(sentiment_model_name)
        
        # Meta-learner (Level 1 model)
        self.meta_learner = None
        
        # Encoders
        self.lang_encoder = None
        self.sentiment_encoder = None
        
    def load_language_model(self):
        """Load your trained language identification model"""
        if self.language_model_path and os.path.exists(self.language_model_path):
            # Load encoders
            self.lang_encoder = joblib.load(f"{self.language_model_path}/lang_encoder.pkl")
            
            # Initialize and load model
            self.language_model = LanguageIdentifierModel(
                LANGUAGE_MODEL_NAME, 
                len(self.lang_encoder.classes_)
            ).to(DEVICE)
            
            # Load state dict
            state_dict = torch.load(f"{self.language_model_path}/pytorch_model.bin", 
                                  map_location=DEVICE)
            self.language_model.load_state_dict(state_dict)
            self.language_model.eval()
            print("✅ Language model loaded successfully")
        else:
            print("⚠️ Language model not found. Will train a simple classifier.")
            self._train_simple_language_classifier()
    
    def _train_simple_language_classifier(self):
        """Train a simple language classifier if pre-trained model is not available"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        
        self.lang_encoder = LabelEncoder()
        y_lang = self.lang_encoder.fit_transform(df['language'])
        
        self.simple_lang_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('nb', MultinomialNB())
        ])
        
        self.simple_lang_classifier.fit(df['text'], y_lang)
        print("✅ Simple language classifier trained")
    
    def extract_language_features(self, texts: List[str]) -> np.ndarray:
        """Extract language identification features"""
        if self.language_model is not None:
            # Use neural language model
            dataset = FeatureExtractionDataset(texts, self.lang_tokenizer)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            features = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    
                    lang_probs = self.language_model(input_ids, attention_mask)
                    features.append(lang_probs.cpu().numpy())
            
            return np.vstack(features)
        else:
            # Use simple classifier
            lang_probs = self.simple_lang_classifier.predict_proba(texts)
            return lang_probs
    
    def extract_sentiment_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentiment features using pre-trained model"""
        return self.sentiment_model.predict_proba(texts)
    
    def create_meta_features(self, texts: List[str]) -> np.ndarray:
        """Create combined feature matrix for meta-learner"""
        print("🔍 Extracting language features...")
        lang_features = self.extract_language_features(texts)
        
        print("💭 Extracting sentiment features...")
        sentiment_features = self.extract_sentiment_features(texts)
        
        # Combine features
        meta_features = np.hstack([lang_features, sentiment_features])
        print(f"✅ Created meta-features with shape: {meta_features.shape}")
        
        return meta_features
    
    def train_with_cv(self, texts: List[str], labels: List[str], 
                     meta_learner_type='logistic'):
        """Train the stacking ensemble using cross-validation"""
        
        # Prepare labels
        self.sentiment_encoder = LabelEncoder()
        y = self.sentiment_encoder.fit_transform(labels)
        
        # Initialize meta-learner
        if meta_learner_type == 'logistic':
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_learner_type == 'rf':
            self.meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("meta_learner_type must be 'logistic' or 'rf'")
        
        # Cross-validation for meta-features generation
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        meta_features_list = []
        meta_labels_list = []
        
        print(f"🔄 Starting {K_FOLDS}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
            print(f"📊 Processing fold {fold + 1}/{K_FOLDS}")
            
            # Get validation data
            val_texts = [texts[i] for i in val_idx]
            val_labels = [y[i] for i in val_idx]
            
            # Create meta-features for validation set
            val_meta_features = self.create_meta_features(val_texts)
            
            meta_features_list.append(val_meta_features)
            meta_labels_list.extend(val_labels)
        
        # Combine all meta-features
        all_meta_features = np.vstack(meta_features_list)
        all_meta_labels = np.array(meta_labels_list)
        
        # Train meta-learner
        print("🎯 Training meta-learner...")
        self.meta_learner.fit(all_meta_features, all_meta_labels)
        
        print("✅ Stacking ensemble training completed!")
        
        return all_meta_features, all_meta_labels
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions using the stacking ensemble"""
        meta_features = self.create_meta_features(texts)
        predictions = self.meta_learner.predict(meta_features)
        return self.sentiment_encoder.inverse_transform(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        meta_features = self.create_meta_features(texts)
        return self.meta_learner.predict_proba(meta_features)
    
    def save_model(self, save_dir: str):
        """Save the entire ensemble"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save meta-learner
        joblib.dump(self.meta_learner, f"{save_dir}/meta_learner.pkl")
        
        # Save encoders
        if self.sentiment_encoder:
            joblib.dump(self.sentiment_encoder, f"{save_dir}/sentiment_encoder.pkl")
        
        if hasattr(self, 'simple_lang_classifier'):
            joblib.dump(self.simple_lang_classifier, f"{save_dir}/simple_lang_classifier.pkl")
            joblib.dump(self.lang_encoder, f"{save_dir}/lang_encoder.pkl")
        
        print(f"✅ Ensemble saved to {save_dir}")
    
    def load_model(self, save_dir: str):
        """Load the entire ensemble"""
        self.meta_learner = joblib.load(f"{save_dir}/meta_learner.pkl")
        self.sentiment_encoder = joblib.load(f"{save_dir}/sentiment_encoder.pkl")
        
        if os.path.exists(f"{save_dir}/simple_lang_classifier.pkl"):
            self.simple_lang_classifier = joblib.load(f"{save_dir}/simple_lang_classifier.pkl")
            self.lang_encoder = joblib.load(f"{save_dir}/lang_encoder.pkl")
        
        print(f"✅ Ensemble loaded from {save_dir}")

# --- MAIN TRAINING FUNCTION ---
def train_stacking_ensemble():
    """Main function to train the stacking ensemble"""
    
    # Initialize ensemble
    ensemble = StackingEnsemble(language_model_path="multitask_afrosenti_model")
    
    # Load language model
    ensemble.load_language_model()
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Train ensemble
    meta_features, meta_labels = ensemble.train_with_cv(
        texts, labels, meta_learner_type='logistic'
    )
    
    # Evaluate on training data (for demonstration)
    train_predictions = ensemble.predict(texts[:1000])  # Sample for speed
    train_accuracy = accuracy_score(labels[:1000], train_predictions)
    
    print(f"🎯 Training accuracy (sample): {train_accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(labels[:1000], train_predictions))
    
    # Save ensemble
    ensemble.save_model(SAVE_DIR)
    
    return ensemble

# --- EVALUATION FUNCTION ---
def evaluate_models_comparison(texts: List[str], labels: List[str]):
    """Compare different approaches"""
    
    print("🔍 Evaluating different approaches...")
    
    # 1. Baseline: Direct sentiment model
    print("\n1️⃣ Baseline: Direct sentiment analysis")
    baseline_model = SentimentModelWrapper(SENTIMENT_MODEL_NAME)
    baseline_preds = []
    
    for text in texts[:100]:  # Sample for speed
        probs = baseline_model.predict_proba([text])
        pred = np.argmax(probs[0])
        baseline_preds.append(pred)
    
    # 2. Stacking ensemble
    print("\n2️⃣ Stacking ensemble")
    ensemble = StackingEnsemble()
    ensemble.load_model(SAVE_DIR)
    ensemble_preds = ensemble.predict(texts[:100])
    
    # Convert predictions to same format for comparison
    sentiment_encoder = LabelEncoder()
    sentiment_encoder.fit(labels)
    
    baseline_labels = sentiment_encoder.inverse_transform(baseline_preds)
    
    print(f"Baseline accuracy: {accuracy_score(labels[:100], baseline_labels):.4f}")
    print(f"Ensemble accuracy: {accuracy_score(labels[:100], ensemble_preds):.4f}")

if __name__ == "__main__":
    # Train the stacking ensemble
    ensemble = train_stacking_ensemble()
    
    # Optional: Run comparison
    # evaluate_models_comparison(df['text'].tolist(), df['label'].tolist())