"""Tests for classification decision logic."""
from classifier.classification import (
    determine_final_classification,
    format_keyword_strings,
    get_confidence_level,
)


def _make_rule_result(label='obligasi_biasa', confidence=0.9, green=0, sustain=0, linked=0):
    return {
        'label': label,
        'confidence': confidence,
        'scores': {'green': green, 'sustainability': sustain, 'linked': linked},
        'keywords': {'green': [], 'sustainability': [], 'linked': []}
    }


def _make_ml_result(label='obligasi_biasa', confidence=0.85):
    return {
        'label': label,
        'confidence': confidence,
        'probabilities': {label: confidence}
    }


class TestDetermineFinalClassification:
    def test_rule_method(self):
        rule = _make_rule_result('green_bond', 0.85, green=30)
        label, conf, method = determine_final_classification(rule, None, 'rule')
        assert label == 'green_bond'
        assert method == 'rule-based'

    def test_ml_none_fallback(self):
        rule = _make_rule_result('green_bond', 0.85, green=30)
        label, conf, method = determine_final_classification(rule, None, 'hybrid')
        assert label == 'green_bond'
        assert method == 'rule-based'

    def test_hybrid_high_confidence_ml_agrees(self):
        rule = _make_rule_result('green_bond', 0.85, green=30)
        ml = _make_ml_result('green_bond', 0.90)
        label, conf, method = determine_final_classification(rule, ml, 'hybrid')
        assert label == 'green_bond'
        assert 'machine-learning' in method

    def test_hybrid_high_confidence_ml_disagrees(self):
        rule = _make_rule_result('green_bond', 0.85, green=30)
        ml = _make_ml_result('sustainability_bond', 0.90)
        label, conf, method = determine_final_classification(rule, ml, 'hybrid')
        assert label == 'green_bond'
        assert 'rule-based' in method

    def test_hybrid_low_ml_confidence(self):
        rule = _make_rule_result('green_bond', 0.85, green=30)
        ml = _make_ml_result('sustainability_bond', 0.50)
        label, conf, method = determine_final_classification(rule, ml, 'hybrid')
        assert label == 'green_bond'
        assert method == 'rule-based'

    def test_ml_override_no_keyword_evidence(self):
        """ML should be overridden when predicting non-regular but no keywords found."""
        rule = _make_rule_result('obligasi_biasa', 0.90, green=0, sustain=0, linked=0)
        ml = _make_ml_result('green_bond', 0.90)
        label, conf, method = determine_final_classification(rule, ml, 'hybrid')
        assert label == 'obligasi_biasa'
        assert 'override' in method.lower()

    def test_ml_method_with_keyword_evidence(self):
        rule = _make_rule_result('green_bond', 0.85, green=30)
        ml = _make_ml_result('green_bond', 0.92)
        label, conf, method = determine_final_classification(rule, ml, 'ml')
        assert label == 'green_bond'
        assert method == 'machine-learning'


class TestFormatKeywordStrings:
    def test_empty_keywords(self):
        rule = _make_rule_result()
        green, sustain, linked = format_keyword_strings(rule)
        assert green == ''
        assert sustain == ''
        assert linked == ''

    def test_format_with_keywords(self):
        rule = _make_rule_result()
        rule['keywords']['green'] = [
            {'keyword': 'green bond', 'count': 3, 'weight': 12, 'capped': False}
        ]
        green, sustain, linked = format_keyword_strings(rule)
        assert 'green bond' in green
        assert '3x12=36' in green

    def test_capped_keyword_format(self):
        rule = _make_rule_result()
        rule['keywords']['green'] = [
            {'keyword': 'green bond', 'count': 10, 'weight': 12, 'capped': True}
        ]
        green, _, _ = format_keyword_strings(rule)
        assert 'capped from 10' in green
        assert '5x12=60' in green


class TestGetConfidenceLevel:
    def test_high(self):
        assert get_confidence_level(0.95) == 'high'
        assert get_confidence_level(0.80) == 'high'

    def test_medium(self):
        assert get_confidence_level(0.79) == 'medium'
        assert get_confidence_level(0.50) == 'medium'

    def test_low(self):
        assert get_confidence_level(0.49) == 'low'
        assert get_confidence_level(0.0) == 'low'
