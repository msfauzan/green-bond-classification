"""
Final classification decision logic for Green Bond Classification.
"""


def determine_final_classification(rule_result: dict, ml_result: dict, method: str) -> tuple:
    """
    Shared logic to determine final classification from rule + ML results.
    Returns (final_label, final_confidence, final_method).
    """
    green_score = rule_result['scores']['green']
    sustain_score = rule_result['scores']['sustainability']
    linked_score = rule_result['scores']['linked']
    max_keyword_score = max(green_score, sustain_score, linked_score)

    if method == 'rule' or ml_result is None:
        return rule_result['label'], rule_result['confidence'], 'rule-based'
    elif method == 'ml':
        if ml_result['label'] != 'obligasi_biasa' and max_keyword_score < 10:
            return 'obligasi_biasa', 0.60, 'rule-based (ML override - no keyword evidence)'
        else:
            ml_label = ml_result['label']
            rule_label = rule_result['label']
            if ml_label == rule_label or rule_label == 'obligasi_biasa':
                return ml_result['label'], ml_result['confidence'], 'machine-learning'
            else:
                return rule_result['label'], rule_result['confidence'], 'rule-based (ML disagrees)'
    else:  # hybrid
        if ml_result['confidence'] >= 0.8:
            if ml_result['label'] != 'obligasi_biasa' and max_keyword_score < 10:
                return 'obligasi_biasa', 0.60, 'rule-based (ML override - no keyword evidence)'
            else:
                ml_label = ml_result['label']
                rule_label = rule_result['label']
                if ml_label == rule_label:
                    return ml_result['label'], ml_result['confidence'], 'machine-learning'
                else:
                    return rule_result['label'], rule_result['confidence'], 'rule-based (ML disagrees)'
        else:
            return rule_result['label'], rule_result['confidence'], 'rule-based'


def format_keyword_strings(rule_result: dict) -> tuple:
    """Format keyword results into display strings with proper capping."""
    def fmt_kw(kw_list):
        parts = []
        for k in kw_list:
            capped = min(k['count'], 5)
            cap_note = f" [capped from {k['count']}]" if k['count'] > 5 else ''
            parts.append(f"{k['keyword']} ({capped}x{k['weight']}={capped*k['weight']}{cap_note})")
        return '; '.join(parts)

    return (
        fmt_kw(rule_result['keywords']['green']),
        fmt_kw(rule_result['keywords']['sustainability']),
        fmt_kw(rule_result['keywords']['linked'])
    )


def get_confidence_level(confidence: float) -> str:
    """Determine confidence level string."""
    if confidence >= 0.8:
        return 'high'
    elif confidence >= 0.5:
        return 'medium'
    return 'low'
