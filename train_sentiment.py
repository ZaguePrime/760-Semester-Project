import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# --- Load Data ---
df = pd.read_csv('all_languages_train_shuffled.tsv', sep='\t')
df = df[['text', 'label', 'language']].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Train/Test Split ---
X = df[['text', 'language']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- ColumnTransformer ---
preprocessor = ColumnTransformer(transformers=[
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'text'),
    ('lang_enc', OneHotEncoder(handle_unknown='ignore'), 'language')
])

# --- Pipeline ---
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultinomialNB())
])

# --- Train ---
print("Training model with language as feature...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- Evaluate ---
y_pred = model_pipeline.predict(X_test)
print("\nSentiment Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- Save ---
os.makedirs("final", exist_ok=True)
joblib.dump(model_pipeline, "final/sentiment_nb_with_lang.joblib")
print("\nModel saved to 'final/sentiment_nb_with_lang.joblib'")
