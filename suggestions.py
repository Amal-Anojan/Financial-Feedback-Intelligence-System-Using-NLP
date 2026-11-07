"""Existing taxonomy workflow suggestions, not a trained model or automatic actions."""
import hashlib
import json

from review_analyzer import ROOT

_raw = (ROOT / "taxonomy.json").read_bytes()
_taxonomy = json.loads(_raw)
RULES = {"risk_actions": _taxonomy["risk_bands"], "intent_owners": {name: info["action_owner"] for name, info in _taxonomy["intents"].items()}}
RULE_VERSION = "taxonomy-" + hashlib.sha256(_raw).hexdigest()[:12]


def suggest_action(result):
    if result.get("security_policy"):
        return {"suggested_action": "Reported security incident: urgent human verification by the fraud/security team. "
                "Investigate access and transaction records; use the company's incident-response procedure. "
                "Do not treat the review as proof of fraud or automatically close the account.",
                "action_owner": "Fraud Operations + Security + Customer Support",
                "suggestion_version": result["security_policy"]["version"]}
    # The existing taxonomy defines integer bands; round only for this lookup.
    score = min(10, max(0, int(result["risk_score"] + 0.5)))
    action = next(band["action"] for band in RULES["risk_actions"] if band["min"] <= score <= band["max"])
    owner = RULES["intent_owners"].get(result["intent"]["label"], "Customer Support")
    uncertain = any(result[task]["confidence"] is None or result[task]["confidence"] < 0.5 for task in ("sentiment", "intent", "root_cause"))
    prefix = "Verify low-confidence predictions first. " if uncertain else "Confirm the review details. "
    return {"suggested_action": prefix + "Suggested next step: " + action + ".",
            "action_owner": owner, "suggestion_version": RULE_VERSION}
