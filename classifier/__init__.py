"""
Shared Green Bond Classification Package
Bank Indonesia - DSta-DSMF

Single source of truth for keywords, scoring, and classification logic.
"""

from classifier.constants import LABELS, LABEL_DISPLAY, CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM
from classifier.keywords import GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS
from classifier.scoring import calc_score, rule_based_classify, ml_classify, ModelManager
from classifier.classification import (
    determine_final_classification,
    format_keyword_strings,
    get_confidence_level,
)
