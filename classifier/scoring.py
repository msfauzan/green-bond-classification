"""
Scoring and ML classification logic for Green Bond Classification.
"""

import re
import os
import numpy as np
import joblib
from scipy.sparse import hstack

from classifier.keywords import GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS


def calc_score(text: str, keywords: list) -> tuple:
    """
    Calculate keyword score with proper word boundary matching.

    Uses regex word boundaries to avoid substring false positives.
    For multi-word keywords, looks for the exact phrase.
    """
    text_lower = text.lower()
    text_normalized = re.sub(r'\s+', ' ', text_lower)

    total = 0
    found = []

    for kw, weight in keywords:
        kw_lower = kw.lower()

        if ' ' in kw_lower:
            count = text_normalized.count(kw_lower)
        else:
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            matches = re.findall(pattern, text_normalized)
            count = len(matches)

        if count > 0:
            capped_count = min(count, 5)
            total += weight * capped_count
            found.append({
                'keyword': kw,
                'count': count,
                'weight': weight,
                'capped': count > 5
            })

    return total, found


def rule_based_classify(text: str) -> dict:
    """Rule-based classification with keyword scoring based on POJK 18/2023"""
    green_score, green_kw = calc_score(text, GREEN_KEYWORDS)
    sustain_score, sustain_kw = calc_score(text, SUSTAINABILITY_KEYWORDS)
    linked_score, linked_kw = calc_score(text, SUSTAINABILITY_LINKED_KEYWORDS)

    # Classification logic based on POJK 18/2023 Pasal 1:
    # Priority order (most specific -> least specific):
    #
    # 1. Sustainability-Linked Bond (Pasal 1 ayat 7):
    #    - Dikaitkan dengan pencapaian IKU dan TKK (priority 1)
    # 2. Sustainability Bond (Pasal 1 ayat 5):
    #    - Dana untuk KOMBINASI kegiatan lingkungan + sosial (priority 2)
    # 3. Green Bond (Pasal 1 ayat 3):
    #    - Dana KHUSUS untuk kegiatan lingkungan saja (priority 3)
    # 4. Obligasi Biasa: None of the above threshold met

    if linked_score >= 10:
        label = 'sustainability_linked_bond'
    elif sustain_score >= 10:
        label = 'sustainability_bond'
    elif green_score >= 10:
        label = 'green_bond'
    else:
        label = 'obligasi_biasa'

    max_score = max(green_score, sustain_score, linked_score)
    if label == 'obligasi_biasa':
        if max_score == 0:
            confidence = 0.90
        elif max_score < 5:
            confidence = 0.75
        else:
            confidence = 0.60
    else:
        if max_score >= 50:
            confidence = 0.95
        elif max_score >= 30:
            confidence = 0.85
        elif max_score >= 20:
            confidence = 0.75
        else:
            confidence = 0.70

    return {
        'label': label,
        'confidence': confidence,
        'scores': {
            'green': green_score,
            'sustainability': sustain_score,
            'linked': linked_score
        },
        'keywords': {
            'green': green_kw,
            'sustainability': sustain_kw,
            'linked': linked_kw
        }
    }


class ModelManager:
    """Manages ML model loading and prediction, replacing global state."""

    def __init__(self):
        self.model = None
        self.vectorizer = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def load(self, model_dir: str) -> bool:
        try:
            model_files = [f for f in os.listdir(model_dir)
                           if f.startswith('logistic_regression') and f.endswith('.joblib')]
            vec_files = [f for f in os.listdir(model_dir)
                         if f.startswith('vectorizer') and f.endswith('.joblib')]

            if model_files and vec_files:
                model_files.sort(reverse=True)
                vec_files.sort(reverse=True)
                self.model = joblib.load(os.path.join(model_dir, model_files[0]))
                self.vectorizer = joblib.load(os.path.join(model_dir, vec_files[0]))
                print(f"Loaded model: {model_files[0]}")
                return True
        except Exception as e:
            print(f"Could not load ML model: {e}")
        return False

    def predict(self, text: str) -> dict | None:
        """Run ML prediction. Returns None if model not loaded."""
        if not self.is_loaded:
            return None
        return ml_classify(text, self.model, self.vectorizer)


def ml_classify(text: str, model=None, vectorizer=None) -> dict | None:
    """ML-based classification using provided model and vectorizer."""
    if model is None or vectorizer is None:
        return None

    try:
        tfidf = vectorizer.transform([text])

        green_score, _ = calc_score(text, GREEN_KEYWORDS)
        sustain_score, _ = calc_score(text, SUSTAINABILITY_KEYWORDS)
        linked_score, _ = calc_score(text, SUSTAINABILITY_LINKED_KEYWORDS)

        word_count = len(text.split())

        extra_features = np.array([[green_score, sustain_score, linked_score, word_count, 0]])
        features = hstack([tfidf, extra_features])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        return {
            'label': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                label: float(prob)
                for label, prob in zip(model.classes_, probabilities)
            }
        }
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None
