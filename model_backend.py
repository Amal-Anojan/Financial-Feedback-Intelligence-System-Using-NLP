"""Transformer-only model selection; missing exports never trigger a fallback."""
import os
from pathlib import Path
from review_analyzer import ROOT

def load_configured_analyzer():
    path = Path(os.environ.get("FINSENTIMENT_MODEL_CONFIG", str(ROOT / "transformer_models.json")))
    if not path.is_file():
        raise FileNotFoundError("Transformer setup is incomplete. Train/export all four tasks, then copy transformer_models.example.json to transformer_models.json and set the export paths. See README.md.")
    from transformer_analyzer import TransformerReviewAnalyzer
    return TransformerReviewAnalyzer(path)
