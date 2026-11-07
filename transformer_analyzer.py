"""Inference adapter for validated exports from notebook 10. Imports torch lazily."""
import hashlib
import json
import math
from pathlib import Path

from review_analyzer import ROOT, validate_review
from security_policy import apply_security_policy, POLICY_VERSION

TASKS = ("sentiment", "intent", "root_cause", "risk")
BAND_MIDPOINTS = json.loads((ROOT / "taxonomy.json").read_text(encoding="utf-8"))["risk_score_method"]["representatives"]


def read_exports(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("risk_score_method") != "expected_band_midpoint":
        raise ValueError("Set risk_score_method to expected_band_midpoint to explicitly accept the disclosed band conversion.")
    exports = {}
    for task in TASKS:
        folder = (config_path.parent / config["exports"][task]).resolve()
        meta = json.loads((folder / "deployment.json").read_text(encoding="utf-8"))
        model_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        names = meta.get("label_names", [])
        if meta.get("schema_version") != 2 or meta.get("task") != task or len(names) < 2 or len(set(names)) != len(names):
            raise ValueError(f"{task}: invalid deployment manifest")
        if meta.get("text_preprocessing") != "as_is" or type(meta.get("max_length")) is not int or meta["max_length"] <= 0:
            raise ValueError(f"{task}: invalid preprocessing settings")
        if meta.get("taxonomy_sha256") != hashlib.sha256((ROOT / "taxonomy.json").read_bytes()).hexdigest():
            raise ValueError(f"{task}: export taxonomy differs from current policy; audit and retrain/reversion explicitly")
        if not (folder / "evaluation.json").is_file():
            raise ValueError(f"{task}: evaluation.json missing; evaluate before deployment")
        mapping = model_config.get("id2label", {})
        if mapping != {str(i): name for i, name in enumerate(names)}:
            raise ValueError(f"{task}: model label mapping differs from export")
        if task == "risk" and not set(names).issubset(BAND_MIDPOINTS):
            raise ValueError("Unknown risk bands")
        # Require safe tensor exports, then verify the exact bytes once at startup.
        hashes = meta.get("weight_sha256", {})
        actual = {p.name for p in folder.glob("*.safetensors")}
        if not actual or actual != set(hashes):
            raise ValueError(f"{task}: missing/extra weights or non-safetensors export")
        for name, expected in hashes.items():
            digest = hashlib.sha256()
            with (folder / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"{task}: weight checksum mismatch")
        exports[task] = (folder, meta)
    return config, exports


class TransformerReviewAnalyzer:
    def __init__(self, config_path):
        config, exports = read_exports(config_path)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = config.get("device", "cpu")
        self.tasks = {}
        for task, (folder, meta) in exports.items():
            tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(folder), local_files_only=True, use_safetensors=True)
            model.to(self.device).eval()
            self.tasks[task] = (tokenizer, model, meta)
        fingerprint = hashlib.sha256(json.dumps({k:v[1] for k,v in exports.items()}, sort_keys=True).encode()).hexdigest()[:16]
        self.model_version = "transformer-" + fingerprint + "+" + POLICY_VERSION

    def predict(self, text):
        validate_review(text)  # Validation only; do NOT apply the baseline's lowercasing.
        result = {"review": text.strip(), "model": "ModernBERT transformer exports", "warnings": []}
        for task, (tokenizer, model, meta) in self.tasks.items():
            inputs = tokenizer(text, truncation=True, max_length=meta["max_length"], return_tensors="pt")
            inputs = {key: value.to(self.device) for key,value in inputs.items()}
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().tolist()
            names = meta["label_names"]
            if len(probabilities) != len(names) or any(not math.isfinite(p) for p in probabilities):
                raise RuntimeError(f"{task}: model returned invalid probabilities")
            best = max(range(len(names)), key=lambda i: probabilities[i])
            if task == "risk":
                result["risk_band"] = {"label": names[best], "confidence": probabilities[best]}
                result["risk_score"] = round(sum(p * BAND_MIDPOINTS[name] for p,name in zip(probabilities,names)), 1)
                result["risk_score_method"] = "expected_band_midpoint"
                result["warnings"].append("Risk score is a probability-weighted band midpoint, not a trained numerical regression prediction.")
            else:
                result[task] = {"label": names[best], "confidence": probabilities[best],
                    "alternatives": [{"label": names[i], "confidence": probabilities[i]} for i in
                        sorted(range(len(names)), key=lambda i: probabilities[i], reverse=True)[:3]]}
                if probabilities[best] < .5:
                    result["warnings"].append(f"Low confidence for {task}; human review required.")
        return apply_security_policy(result)
