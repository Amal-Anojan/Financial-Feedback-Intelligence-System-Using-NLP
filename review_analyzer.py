"""Shared input validation; inference is transformer-only."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MAX_REVIEW_LENGTH = 5000
TASKS = ("sentiment", "intent", "root_cause")


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_review(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a customer review before analyzing.")
    if len(text) > MAX_REVIEW_LENGTH:
        raise ValueError(f"Keep the review within {MAX_REVIEW_LENGTH:,} characters.")
    if not any(c.isalpha() for c in text):
        raise ValueError("Enter a review containing words, rather than only numbers or symbols.")
    return normalize_text(text)
