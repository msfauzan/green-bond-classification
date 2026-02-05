"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ML TRAINER - Green Bond Classification                              ║
║           Bank Indonesia - DSta-DSMF                                         ║
║                                                                              ║
║  Models: Logistic Regression, SVM, Decision Tree, Random Forest              ║
║  Features: TF-IDF + Keyword Scores + Document Stats                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

import PyPDF2
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = r'd:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification'
DATASET_PATH = os.path.join(BASE_DIR, 'ML_Dataset', 'labeled_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'ML_Models')

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ' '.join([(p.extract_text() or '') for p in reader.pages])
            return text
    except:
        return ""


def prepare_features(df, fit_vectorizer=True, vectorizer=None):
    """
    Prepare feature matrix from dataset.
    
    Returns:
        X: Feature matrix (TF-IDF + numeric features)
        y: Labels
        vectorizer: Fitted TF-IDF vectorizer
    """
    print("📊 Extracting features...")
    
    # Extract text from PDFs
    texts = []
    for path in df['filepath']:
        text = extract_text_from_pdf(path)
        texts.append(text)
    
    # TF-IDF features
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words=None  # Keep Indonesian stopwords
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    
    # Numeric features (keyword scores, doc stats)
    numeric_features = df[['green_score', 'sustain_score', 'linked_score', 'pages', 'words']].values
    
    # Normalize numeric features
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)
    
    # Combine TF-IDF + numeric features
    X = np.hstack([tfidf_matrix.toarray(), numeric_scaled])
    
    # Labels
    y = df['label'].values
    
    print(f"   Features shape: {X.shape}")
    print(f"   Labels: {np.unique(y)}")
    
    return X, y, vectorizer


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_models(X, y):
    """Train multiple models and compare performance."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        ),
        'SVM (Linear)': SVC(
            kernel='linear', 
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', 
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
    }
    
    results = []
    best_model = None
    best_score = 0
    
    print("\n" + "="*70)
    print("🤖 TRAINING MODELS")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n▶ {name}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        print(f"   CV Score: {cv_mean:.3f} (+/- {cv_std:.3f})")
        
        results.append({
            'model': name,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        })
        
        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = (name, model)
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Best model details
    print(f"\n🏆 Best Model: {best_model[0]} (F1: {best_score:.3f})")
    
    # Classification report for best model
    y_pred_best = best_model[1].predict(X_test)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_best))
    
    return results_df, best_model, X_train, X_test, y_train, y_test


def save_model(model, vectorizer, model_name):
    """Save trained model and vectorizer."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f'{model_name}_{timestamp}.joblib')
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save vectorizer
    vec_path = os.path.join(MODEL_DIR, f'vectorizer_{timestamp}.joblib')
    joblib.dump(vectorizer, vec_path)
    print(f"✅ Vectorizer saved: {vec_path}")
    
    # Save metadata
    meta = {
        'model_name': model_name,
        'timestamp': timestamp,
        'model_path': model_path,
        'vectorizer_path': vec_path,
    }
    meta_path = os.path.join(MODEL_DIR, f'metadata_{timestamp}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return model_path, vec_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           ML TRAINER - Green Bond Classification                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("   Please run generate_labels.py first!")
        return
    
    df = pd.read_csv(DATASET_PATH)
    print(f"📂 Loaded {len(df)} records from {DATASET_PATH}")
    
    # Show label distribution
    print("\n📊 Label Distribution:")
    for label, count in df['label'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    # Check minimum samples
    min_samples = df['label'].value_counts().min()
    if min_samples < 5:
        print(f"\n⚠️ Warning: Some classes have < 5 samples. Results may be unreliable.")
    
    # Prepare features
    X, y, vectorizer = prepare_features(df)
    
    # Train models
    results_df, best_model, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
    
    # Save best model
    print("\n" + "="*70)
    print("💾 SAVING BEST MODEL")
    print("="*70)
    model_name = best_model[0].lower().replace(' ', '_').replace('(', '').replace(')', '')
    save_model(best_model[1], vectorizer, model_name)
    
    # Save results
    results_path = os.path.join(MODEL_DIR, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✅ Results saved: {results_path}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
