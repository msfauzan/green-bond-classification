"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PSEUDO-LABELER - Generate labels using existing model              ║
║           Bank Indonesia - DSta-DSMF                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import PyPDF2
import pandas as pd
import joblib

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from classifier.scoring import calculate_scores


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return ' '.join(text_parts).strip()
    except Exception as e:
        print(f"❌ Error extracting {pdf_path}: {e}")
        return ""


def load_latest_model() -> tuple:
    """Load the latest trained model and vectorizer."""
    models_dir = BASE_DIR / "ML_Models"

    # Find latest model
    model_files = list(models_dir.glob("logistic_regression_*.joblib"))
    if not model_files:
        print("❌ No trained model found!")
        return None, None

    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

    # Find matching vectorizer
    timestamp = latest_model.stem.replace("logistic_regression_", "")
    vectorizer_file = models_dir / f"vectorizer_{timestamp}.joblib"

    if not vectorizer_file.exists():
        print(f"❌ Vectorizer not found: {vectorizer_file}")
        return None, None

    model = joblib.load(latest_model)
    vectorizer = joblib.load(vectorizer_file)

    print(f"✅ Loaded model: {latest_model.name}")
    print(f"   Classes: {list(model.classes_)}")
    return model, vectorizer


def rule_based_classify(text: str) -> Dict[str, Any]:
    """Classify using rule-based scoring."""
    scores = calculate_scores(text)

    # Determine label based on scores
    green = scores['green']
    sustain = scores['sustainability']
    linked = scores['linked']
    max_score = max(green, sustain, linked)

    # High confidence thresholds
    if linked >= 15 and linked >= sustain and linked >= green:
        label = "sustainability_linked_bond"
        confidence = min(0.95, 0.5 + linked / 50)
    elif sustain >= 15 and sustain >= green:
        label = "sustainability_bond"
        confidence = min(0.95, 0.5 + sustain / 50)
    elif green >= 15:
        label = "green_bond"
        confidence = min(0.95, 0.5 + green / 50)
    elif max_score >= 10:
        # Medium confidence - still make a prediction
        if linked > sustain and linked > green:
            label = "sustainability_linked_bond"
        elif sustain > green:
            label = "sustainability_bond"
        else:
            label = "green_bond"
        confidence = 0.5 + max_score / 100
    else:
        label = "obligasi_biasa"
        confidence = 0.8 if max_score < 5 else 0.6

    return {
        "label": label,
        "confidence": confidence,
        "scores": scores
    }


def ml_classify(text: str, model, vectorizer) -> Dict[str, Any]:
    """Classify using ML model."""
    if not text:
        return {"label": "unknown", "confidence": 0.0, "probabilities": {}}

    # Vectorize
    X = vectorizer.transform([text])

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    return {
        "label": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    }


def hybrid_classify(text: str, model, vectorizer) -> Dict[str, Any]:
    """Combine rule-based and ML predictions."""
    rule_result = rule_based_classify(text)
    ml_result = ml_classify(text, model, vectorizer) if model else None

    # High confidence rule-based wins
    if rule_result['confidence'] >= 0.9:
        return {
            "label": rule_result['label'],
            "confidence": rule_result['confidence'],
            "method": "rule-based",
            "scores": rule_result['scores'],
            "ml_probabilities": ml_result.get('probabilities') if ml_result else None
        }

    # ML wins if confident
    if ml_result and ml_result['confidence'] >= 0.7:
        return {
            "label": ml_result['label'],
            "confidence": ml_result['confidence'],
            "method": "machine-learning",
            "scores": rule_result['scores'],
            "ml_probabilities": ml_result['probabilities']
        }

    # Fallback to rule-based
    return {
        "label": rule_result['label'],
        "confidence": rule_result['confidence'],
        "method": "rule-based",
        "scores": rule_result['scores'],
        "ml_probabilities": ml_result.get('probabilities') if ml_result else None
    }


def process_pdfs(
    pdf_dir: str,
    model,
    vectorizer,
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process all PDFs in directory."""
    results = []
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))

    if max_files:
        pdf_files = pdf_files[:max_files]

    print(f"\n📁 Processing {len(pdf_files)} PDFs from {pdf_dir}...")

    for i, pdf_path in enumerate(pdf_files, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(pdf_files)}")

        # Extract text
        text = extract_text_from_pdf(str(pdf_path))

        if not text:
            print(f"   ⚠️ No text: {pdf_path.name}")
            continue

        # Classify
        result = hybrid_classify(text, model, vectorizer)

        results.append({
            "filepath": str(pdf_path),
            "filename": pdf_path.name,
            "label": result['label'],
            "confidence": round(result['confidence'], 4),
            "method": result['method'],
            "green_score": result['scores']['green'],
            "sustain_score": result['scores']['sustainability'],
            "linked_score": result['scores']['linked'],
            "text_length": len(text),
            "needs_review": result['confidence'] < 0.7
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels for PDFs')
    parser.add_argument(
        '--input', type=str, default='Prospektus_Downloaded',
        help='Input directory containing PDFs (default: Prospektus_Downloaded)'
    )
    parser.add_argument(
        '--output', type=str, default='ML_Dataset/pseudo_labeled.csv',
        help='Output CSV file (default: ML_Dataset/pseudo_labeled.csv)'
    )
    parser.add_argument(
        '--review-output', type=str, default='ML_Dataset/manual_review_priority.csv',
        help='Output for items needing review (default: ML_Dataset/manual_review_priority.csv)'
    )
    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Maximum number of files to process (default: all)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🏷️  PSEUDO-LABELER")
    print("=" * 60)

    # Load model
    model, vectorizer = load_latest_model()

    # Process PDFs
    results = process_pdfs(args.input, model, vectorizer, args.max_files)

    if not results:
        print("❌ No results!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save all results
    output_path = BASE_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} labels to: {output_path}")

    # Save items needing review
    review_df = df[df['needs_review'] == True]
    if len(review_df) > 0:
        review_path = BASE_DIR / args.review_output
        review_df.to_csv(review_path, index=False)
        print(f"⚠️  Saved {len(review_df)} items for review to: {review_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"\nTotal processed: {len(df)}")
    print(f"Needs review: {len(review_df)} ({len(review_df)/len(df)*100:.1f}%)")

    print("\n🏷️  Label distribution:")
    for label, count in df['label'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")

    print("\n📈 Confidence distribution:")
    high = len(df[df['confidence'] >= 0.8])
    med = len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.8)])
    low = len(df[df['confidence'] < 0.5])
    print(f"   High (>=0.8): {high} ({high/len(df)*100:.1f}%)")
    print(f"   Medium (0.5-0.8): {med} ({med/len(df)*100:.1f}%)")
    print(f"   Low (<0.5): {low} ({low/len(df)*100:.1f}%)")

    print("\n⚙️  Method distribution:")
    for method, count in df['method'].value_counts().items():
        print(f"   {method}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
