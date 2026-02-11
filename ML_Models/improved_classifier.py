"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               IMPROVED GREEN BOND CLASSIFIER                                  ║
║               Bank Indonesia - DSta-DSMF                                      ║
║                                                                               ║
║  This module implements a hybrid ML approach combining:                       ║
║  1. TF-IDF with n-grams for phrase detection                                  ║
║  2. Keyword scoring as engineered features                                    ║
║  3. Document structure features                                               ║
║  4. Ensemble of multiple models                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# POJK 18/2023 Keywords - SPECIFIC ONLY (no generic terms)
GREEN_KEYWORDS = [
    ('ebus lingkungan', 15), ('efek bersifat utang berwawasan lingkungan', 15),
    ('sukuk berwawasan lingkungan', 15), ('green bond', 12), ('green sukuk', 12),
    ('obligasi hijau', 10), ('sukuk hijau', 10), ('obligasi berwawasan lingkungan', 10),
    ('pojk 18/2023', 8), ('pojk 18 tahun 2023', 8), ('pojk no. 18/2023', 8),
    ('kubl', 10), ('kegiatan usaha berwawasan lingkungan', 10),
    ('green bond framework', 8), ('kerangka kerja obligasi hijau', 8),
]

SUSTAINABILITY_KEYWORDS = [
    ('ebus keberlanjutan', 15), ('efek bersifat utang keberlanjutan', 15),
    ('sukuk keberlanjutan', 15), ('sustainability bond', 12), ('sustainability sukuk', 12),
    ('obligasi keberlanjutan', 10), ('obligasi berkelanjutan', 10),
    ('ebus sosial', 12), ('efek bersifat utang berwawasan sosial', 12),
    ('social bond', 10), ('social sukuk', 10), ('obligasi sosial', 10),
    ('kubs', 10), ('kegiatan usaha berwawasan sosial', 10),
    ('sustainability bond framework', 8),
]

SUSTAINABILITY_LINKED_KEYWORDS = [
    ('ebus terkait keberlanjutan', 15), ('efek bersifat utang terkait keberlanjutan', 15),
    ('sustainability linked bond', 12), ('sustainability-linked bond', 12),
    ('sustainability linked sukuk', 12), ('sustainability-linked sukuk', 12),
    ('obligasi terkait keberlanjutan', 10),
    ('iku keberlanjutan', 10), ('indikator kinerja utama keberlanjutan', 10),
    ('target kinerja keberlanjutan', 10), ('sustainability performance target', 8),
    ('step-up coupon', 5), ('step-down coupon', 5), ('coupon step-up', 5),
]

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calc_keyword_score(text: str, keywords: list) -> tuple:
    """Calculate keyword score with position weighting"""
    text_lower = text.lower()
    total = 0
    found = []
    
    # Check if keyword appears in first 20% of document (title/intro)
    first_section = text_lower[:len(text_lower)//5]
    
    for kw, weight in keywords:
        count = text_lower.count(kw.lower())
        if count > 0:
            # Bonus if in title/intro section
            in_intro = kw.lower() in first_section
            position_bonus = 1.5 if in_intro else 1.0
            
            # Diminishing returns for repeated keywords
            effective_count = min(count, 5) + (max(0, count - 5) * 0.2)
            score = weight * effective_count * position_bonus
            
            total += score
            found.append({'keyword': kw, 'count': count, 'weight': weight, 'in_intro': in_intro})
    
    return total, found

def extract_features(text: str) -> dict:
    """Extract engineered features from text"""
    text_lower = text.lower()
    
    # Keyword scores
    green_score, green_kw = calc_keyword_score(text, GREEN_KEYWORDS)
    sustain_score, sustain_kw = calc_keyword_score(text, SUSTAINABILITY_KEYWORDS)
    linked_score, linked_kw = calc_keyword_score(text, SUSTAINABILITY_LINKED_KEYWORDS)
    
    # Document statistics
    word_count = len(text.split())
    char_count = len(text)
    
    # Specific pattern detection
    has_green_framework = 1 if 'green bond framework' in text_lower or 'kerangka kerja obligasi hijau' in text_lower else 0
    has_sustain_framework = 1 if 'sustainability bond framework' in text_lower or 'sustainability-linked bond framework' in text_lower else 0
    has_pojk_18 = 1 if 'pojk 18/2023' in text_lower or 'pojk 18 tahun 2023' in text_lower else 0
    has_use_of_proceeds_green = 1 if ('penggunaan dana' in text_lower and ('hijau' in text_lower or 'green' in text_lower)) else 0
    has_iku = 1 if 'iku keberlanjutan' in text_lower or 'indikator kinerja utama' in text_lower else 0
    has_tkk = 1 if 'target kinerja keberlanjutan' in text_lower else 0
    
    # Title page detection (more reliable)
    first_page = text_lower[:3000]
    title_has_green = 1 if any(kw in first_page for kw in ['green bond', 'obligasi hijau', 'sukuk hijau']) else 0
    title_has_sustain = 1 if any(kw in first_page for kw in ['sustainability bond', 'obligasi keberlanjutan', 'social bond']) else 0
    title_has_linked = 1 if any(kw in first_page for kw in ['sustainability linked', 'sustainability-linked']) else 0
    
    return {
        'green_score': green_score,
        'sustain_score': sustain_score,
        'linked_score': linked_score,
        'word_count': word_count,
        'char_count': char_count,
        'has_green_framework': has_green_framework,
        'has_sustain_framework': has_sustain_framework,
        'has_pojk_18': has_pojk_18,
        'has_use_of_proceeds_green': has_use_of_proceeds_green,
        'has_iku': has_iku,
        'has_tkk': has_tkk,
        'title_has_green': title_has_green,
        'title_has_sustain': title_has_sustain,
        'title_has_linked': title_has_linked,
        'green_keywords_found': len(green_kw),
        'sustain_keywords_found': len(sustain_kw),
        'linked_keywords_found': len(linked_kw),
    }

# =============================================================================
# MODEL TRAINING
# =============================================================================

class ImprovedGreenBondClassifier:
    """
    Improved classifier using:
    1. TF-IDF with n-grams
    2. Engineered features
    3. SMOTE for class imbalance
    4. Ensemble voting
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words=None  # Keep Indonesian stopwords for now
        )
        self.scaler = StandardScaler()
        self.model = None
        self.labels = ['green_bond', 'obligasi_biasa', 'sustainability_bond', 'sustainability_linked_bond']
    
    def _prepare_features(self, texts, fit=False):
        """Prepare combined TF-IDF and engineered features"""
        # TF-IDF features
        if fit:
            tfidf_features = self.vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.vectorizer.transform(texts)
        
        # Engineered features
        eng_features = []
        for text in texts:
            feat = extract_features(text)
            eng_features.append(list(feat.values()))
        
        eng_features = np.array(eng_features)
        
        # Scale engineered features
        if fit:
            eng_features = self.scaler.fit_transform(eng_features)
        else:
            eng_features = self.scaler.transform(eng_features)
        
        # Combine features
        combined = hstack([tfidf_features, csr_matrix(eng_features)])
        
        return combined
    
    def train(self, texts, labels, use_smote=True):
        """Train the ensemble classifier"""
        print("=" * 60)
        print("TRAINING IMPROVED GREEN BOND CLASSIFIER")
        print("=" * 60)
        
        # Prepare features
        print("\n📊 Preparing features...")
        X = self._prepare_features(texts, fit=True)
        y = np.array(labels)
        
        print(f"   Total samples: {len(y)}")
        print(f"   Feature dimensions: {X.shape}")
        print(f"   Class distribution:")
        for label in self.labels:
            count = sum(y == label)
            print(f"      - {label}: {count} ({count/len(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("\n⚖️ Applying SMOTE for class balancing...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, min(sum(y_train == l) for l in self.labels) - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"   Before SMOTE: {len(y_train)} samples")
                print(f"   After SMOTE: {len(y_train_balanced)} samples")
            except Exception as e:
                print(f"   ⚠️ SMOTE failed: {e}")
                print("   Using original imbalanced data...")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Create ensemble
        print("\n🤖 Training ensemble model...")
        
        # Individual classifiers
        clf_lr = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            C=1.0,
            random_state=42
        )
        
        clf_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        clf_gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # Voting ensemble
        self.model = VotingClassifier(
            estimators=[
                ('lr', clf_lr),
                ('rf', clf_rf),
                ('gb', clf_gb)
            ],
            voting='soft',  # Use probabilities
            weights=[2, 1, 1]  # Give more weight to Logistic Regression
        )
        
        # Train
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        print("\n📈 Evaluation Results:")
        y_pred = self.model.predict(X_test)
        
        print(f"\n   Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"   Test F1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
        
        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.labels, zero_division=0))
        
        print("\n   Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=self.labels)
        print(f"   Labels: {self.labels}")
        print(cm)
        
        # Cross-validation
        print("\n🔄 Cross-validation (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf_lr, X, y, cv=cv, scoring='f1_macro')
        print(f"   CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, text: str) -> dict:
        """Predict with confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        X = self._prepare_features([text], fit=False)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get engineered features for explanation
        features = extract_features(text)
        
        return {
            'label': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(self.labels, probabilities)
            },
            'features': features
        }
    
    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'labels': self.labels
        }, path)
        print(f"✅ Model saved to: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        classifier = cls()
        classifier.model = data['model']
        classifier.vectorizer = data['vectorizer']
        classifier.scaler = data['scaler']
        classifier.labels = data['labels']
        return classifier

# =============================================================================
# HYBRID CLASSIFICATION STRATEGY
# =============================================================================

def hybrid_classify(text: str, ml_classifier: ImprovedGreenBondClassifier = None) -> dict:
    """
    Hybrid classification strategy:
    1. First check rule-based with HIGH confidence (explicit keywords)
    2. If no clear signal, use ML
    3. Combine both for final decision
    """
    # Extract features
    features = extract_features(text)
    
    # Rule-based classification (HIGH CONFIDENCE ONLY)
    rule_result = rule_based_classify_strict(features)
    
    # If rule-based is highly confident (score >= 15 with title match)
    if rule_result['high_confidence']:
        return {
            'label': rule_result['label'],
            'confidence': rule_result['confidence'],
            'method': 'rule-based (high confidence)',
            'features': features,
            'rule_result': rule_result,
            'ml_result': None
        }
    
    # Use ML if available
    if ml_classifier is not None:
        ml_result = ml_classifier.predict(text)
        
        # If ML is confident and agrees with rule-based tendency
        if ml_result['confidence'] >= 0.7:
            return {
                'label': ml_result['label'],
                'confidence': ml_result['confidence'],
                'method': 'machine-learning',
                'features': features,
                'rule_result': rule_result,
                'ml_result': ml_result
            }
        
        # If ML is less confident, combine with rule-based
        final_label = determine_final_label(rule_result, ml_result, features)
        return {
            'label': final_label['label'],
            'confidence': final_label['confidence'],
            'method': 'hybrid',
            'features': features,
            'rule_result': rule_result,
            'ml_result': ml_result
        }
    
    # Fallback to rule-based
    return {
        'label': rule_result['label'],
        'confidence': rule_result['confidence'],
        'method': 'rule-based',
        'features': features,
        'rule_result': rule_result,
        'ml_result': None
    }

def rule_based_classify_strict(features: dict) -> dict:
    """Strict rule-based classification - only high confidence matches"""
    green_score = features['green_score']
    sustain_score = features['sustain_score']
    linked_score = features['linked_score']
    
    title_green = features['title_has_green']
    title_sustain = features['title_has_sustain']
    title_linked = features['title_has_linked']
    
    # High confidence: explicit in title + score >= 15
    high_confidence = False
    
    # Check Sustainability-Linked first (most specific)
    if (linked_score >= 15 and title_linked) or linked_score >= 25:
        label = 'sustainability_linked_bond'
        confidence = min(0.95, 0.7 + linked_score/100)
        high_confidence = title_linked or linked_score >= 25
    # Check Sustainability Bond
    elif (sustain_score >= 15 and title_sustain) or sustain_score >= 25:
        label = 'sustainability_bond'
        confidence = min(0.95, 0.7 + sustain_score/100)
        high_confidence = title_sustain or sustain_score >= 25
    # Check Green Bond
    elif (green_score >= 15 and title_green) or green_score >= 25:
        label = 'green_bond'
        confidence = min(0.95, 0.7 + green_score/100)
        high_confidence = title_green or green_score >= 25
    # Medium confidence (score >= 10 but no title match)
    elif linked_score >= 10 and linked_score > max(green_score, sustain_score):
        label = 'sustainability_linked_bond'
        confidence = 0.6
        high_confidence = False
    elif sustain_score >= 10 and sustain_score > green_score:
        label = 'sustainability_bond'
        confidence = 0.6
        high_confidence = False
    elif green_score >= 10:
        label = 'green_bond'
        confidence = 0.6
        high_confidence = False
    else:
        label = 'obligasi_biasa'
        confidence = 0.8 if max(green_score, sustain_score, linked_score) < 5 else 0.5
        high_confidence = max(green_score, sustain_score, linked_score) < 5
    
    return {
        'label': label,
        'confidence': confidence,
        'high_confidence': high_confidence,
        'scores': {
            'green': green_score,
            'sustainability': sustain_score,
            'linked': linked_score
        }
    }

def determine_final_label(rule_result: dict, ml_result: dict, features: dict) -> dict:
    """Combine rule-based and ML results"""
    # Weight rule-based more if it has clear signals
    rule_weight = 0.6 if rule_result['scores']['green'] >= 5 or \
                         rule_result['scores']['sustainability'] >= 5 or \
                         rule_result['scores']['linked'] >= 5 else 0.4
    ml_weight = 1 - rule_weight
    
    # Calculate combined scores
    labels = ['green_bond', 'obligasi_biasa', 'sustainability_bond', 'sustainability_linked_bond']
    combined_scores = {}
    
    for label in labels:
        rule_score = 1.0 if rule_result['label'] == label else 0.0
        rule_score *= rule_result['confidence']
        
        ml_score = ml_result['probabilities'].get(label, 0.0)
        
        combined_scores[label] = rule_weight * rule_score + ml_weight * ml_score
    
    # Get best label
    best_label = max(combined_scores, key=combined_scores.get)
    best_score = combined_scores[best_label]
    
    return {
        'label': best_label,
        'confidence': best_score,
        'combined_scores': combined_scores
    }

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GREEN BOND CLASSIFIER - IMPROVED VERSION")
    print("=" * 60)
    
    # Load dataset
    dataset_path = os.path.join(BASE_DIR, 'ML_Dataset', 'labeled_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        exit(1)
    
    print(f"\n📂 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df)} samples")
    
    # Train model
    classifier = ImprovedGreenBondClassifier()
    results = classifier.train(df['text'].tolist(), df['label'].tolist())
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(BASE_DIR, 'ML_Models', f'improved_classifier_{timestamp}.joblib')
    classifier.save(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
