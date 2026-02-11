"""
Constants for Green Bond Classification.
"""

LABELS = ['green_bond', 'obligasi_biasa', 'sustainability_bond', 'sustainability_linked_bond']

LABEL_DISPLAY = {
    'green_bond': {'name': 'Green Bond', 'emoji': '\u00f0\u009f\u008c\u00bf', 'color': '#22c55e'},
    'sustainability_bond': {'name': 'Sustainability Bond', 'emoji': '\u00e2\u0099\u00bb\u00ef\u00b8\u008f', 'color': '#3b82f6'},
    'sustainability_linked_bond': {'name': 'Sustainability-Linked Bond', 'emoji': '\u00f0\u009f\u0094\u0097', 'color': '#8b5cf6'},
    'obligasi_biasa': {'name': 'Obligasi Biasa', 'emoji': '\u00f0\u009f\u0093\u0084', 'color': '#6b7280'},
}

CONFIDENCE_THRESHOLD_HIGH = 0.80
CONFIDENCE_THRESHOLD_MEDIUM = 0.60
