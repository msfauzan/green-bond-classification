"""Tests for keyword list integrity."""
from classifier.keywords import GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS


def test_green_keywords_not_empty():
    assert len(GREEN_KEYWORDS) > 0


def test_sustainability_keywords_not_empty():
    assert len(SUSTAINABILITY_KEYWORDS) > 0


def test_linked_keywords_not_empty():
    assert len(SUSTAINABILITY_LINKED_KEYWORDS) > 0


def test_keyword_format():
    """All keywords must be (str, int/float) tuples with positive weight."""
    for kw_list in [GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS]:
        for item in kw_list:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
            assert len(item) == 2, f"Expected 2 elements, got {len(item)}"
            keyword, weight = item
            assert isinstance(keyword, str), f"Keyword must be str: {keyword}"
            assert isinstance(weight, (int, float)), f"Weight must be numeric: {weight}"
            assert weight > 0, f"Weight must be positive: {weight}"


def test_keywords_are_lowercase():
    """Keywords should be lowercase for matching."""
    for kw_list in [GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS]:
        for keyword, _ in kw_list:
            assert keyword == keyword.lower(), f"Keyword not lowercase: '{keyword}'"


def test_no_duplicate_keywords_within_list():
    """No duplicate keywords within each list."""
    for name, kw_list in [
        ("GREEN", GREEN_KEYWORDS),
        ("SUSTAINABILITY", SUSTAINABILITY_KEYWORDS),
        ("LINKED", SUSTAINABILITY_LINKED_KEYWORDS),
    ]:
        keywords = [kw for kw, _ in kw_list]
        assert len(keywords) == len(set(keywords)), f"Duplicates in {name}_KEYWORDS"


def test_green_keywords_contain_key_terms():
    """Green keywords should contain essential identifiers."""
    kw_strings = [kw for kw, _ in GREEN_KEYWORDS]
    assert 'green bond' in kw_strings
    assert 'obligasi hijau' in kw_strings


def test_sustainability_keywords_contain_key_terms():
    kw_strings = [kw for kw, _ in SUSTAINABILITY_KEYWORDS]
    assert 'sustainability bond' in kw_strings
    assert 'obligasi keberlanjutan' in kw_strings


def test_linked_keywords_contain_key_terms():
    kw_strings = [kw for kw, _ in SUSTAINABILITY_LINKED_KEYWORDS]
    assert 'sustainability-linked bond' in kw_strings
    assert 'iku keberlanjutan' in kw_strings
