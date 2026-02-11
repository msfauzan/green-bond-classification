"""Tests for scoring functions."""
from classifier.scoring import calc_score, rule_based_classify
from classifier.keywords import GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS


class TestCalcScore:
    def test_no_match(self):
        score, found = calc_score("random text with no keywords", GREEN_KEYWORDS)
        assert score == 0
        assert found == []

    def test_single_keyword_match(self):
        score, found = calc_score("this is a green bond document", GREEN_KEYWORDS)
        assert score > 0
        assert len(found) >= 1
        assert any(k['keyword'] == 'green bond' for k in found)

    def test_multiple_keyword_match(self):
        text = "green bond obligasi hijau green sukuk"
        score, found = calc_score(text, GREEN_KEYWORDS)
        assert score > 0
        assert len(found) >= 2

    def test_count_capped_at_5(self):
        text = " ".join(["green bond"] * 10)
        score, found = calc_score(text, GREEN_KEYWORDS)
        gb_entry = next(k for k in found if k['keyword'] == 'green bond')
        assert gb_entry['count'] == 10  # actual count
        assert gb_entry['capped'] is True
        # Score should be capped: 5 * weight, not 10 * weight
        expected_weight = next(w for kw, w in GREEN_KEYWORDS if kw == 'green bond')
        assert score == 5 * expected_weight

    def test_word_boundary_matching(self):
        """Single-word keywords should use word boundaries."""
        # 'kubl' should not match 'kubla' or 'tkubl'
        score_exact, found_exact = calc_score("kubl is important", GREEN_KEYWORDS)
        score_substring, found_sub = calc_score("kubla khan text", GREEN_KEYWORDS)

        kubl_exact = [k for k in found_exact if k['keyword'] == 'kubl']
        kubl_sub = [k for k in found_sub if k['keyword'] == 'kubl']

        assert len(kubl_exact) == 1
        assert len(kubl_sub) == 0

    def test_case_insensitive(self):
        text = "GREEN BOND FRAMEWORK"
        score, found = calc_score(text, GREEN_KEYWORDS)
        assert score > 0

    def test_multi_word_exact_phrase(self):
        """Multi-word keywords use exact phrase matching."""
        text = "efek bersifat utang berwawasan lingkungan"
        score, found = calc_score(text, GREEN_KEYWORDS)
        assert any(k['keyword'] == 'efek bersifat utang berwawasan lingkungan' for k in found)


class TestRuleBasedClassify:
    def test_green_bond_classification(self, green_bond_text):
        result = rule_based_classify(green_bond_text)
        assert result['label'] == 'green_bond'
        assert result['confidence'] > 0
        assert result['scores']['green'] >= 10

    def test_sustainability_bond_classification(self, sustainability_bond_text):
        result = rule_based_classify(sustainability_bond_text)
        assert result['label'] == 'sustainability_bond'
        assert result['scores']['sustainability'] >= 10

    def test_linked_bond_classification(self, sustainability_linked_text):
        result = rule_based_classify(sustainability_linked_text)
        assert result['label'] == 'sustainability_linked_bond'
        assert result['scores']['linked'] >= 10

    def test_regular_bond_classification(self, regular_bond_text):
        result = rule_based_classify(regular_bond_text)
        assert result['label'] == 'obligasi_biasa'

    def test_empty_text(self):
        result = rule_based_classify("")
        assert result['label'] == 'obligasi_biasa'
        assert result['confidence'] == 0.90  # high confidence for no keywords

    def test_result_structure(self, green_bond_text):
        result = rule_based_classify(green_bond_text)
        assert 'label' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert 'keywords' in result
        assert set(result['scores'].keys()) == {'green', 'sustainability', 'linked'}
        assert set(result['keywords'].keys()) == {'green', 'sustainability', 'linked'}

    def test_linked_priority_over_sustainability(self):
        """Linked bonds should be prioritized over sustainability bonds."""
        text = (
            "sustainability bond obligasi keberlanjutan "
            "ebus terkait keberlanjutan sustainability-linked bond "
            "iku keberlanjutan target kinerja keberlanjutan"
        )
        result = rule_based_classify(text)
        assert result['label'] == 'sustainability_linked_bond'

    def test_confidence_levels(self):
        """Higher scores should produce higher confidence."""
        weak = rule_based_classify("green bond " * 1)
        strong = rule_based_classify("green bond " * 5 + "obligasi hijau " * 5 + "kubl " * 5)
        assert strong['confidence'] >= weak['confidence']
