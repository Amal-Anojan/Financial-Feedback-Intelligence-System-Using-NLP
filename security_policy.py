"""Conservative English incident triage, separate from learned predictions.

Rules flag reported incidents for people to investigate; they do not prove fraud.
They intentionally avoid generic words such as 'account', 'password' or 'money'.
"""
import copy
import hashlib
import re
import json
from pathlib import Path

_taxonomy_bytes = (Path(__file__).parent / "taxonomy.json").read_bytes()
TRIAGE = json.loads(_taxonomy_bytes)["security_triage"]
POLICY_VERSION = "security-" + hashlib.sha256(Path(__file__).read_bytes() + _taxonomy_bytes).hexdigest()[:12]


def security_signal(text):
    text = str(text).casefold().replace("’", "'")
    signals = []
    # Evaluate clauses separately so a negated first issue cannot hide another incident.
    for clause in re.split(r"[.!?;\n]|\bbut\b|\bhowever\b", text):
        hypothetical = re.search(r"\b(if|what if|could|might|would|how (?:do|can|to)|prevent|worried|afraid)\b", clause)
        negated = re.search(r"\b(?:not|never|hasn't|haven't|isn't|wasn't|weren't|no)\b.{0,28}\b(?:leaked|stolen|hacked|compromised|unauthori[sz]ed)\b", clause)
        if hypothetical or negated:
            continue
        money = re.search(r"\b(?:money|funds|payments?|transactions?|transfers?|withdrawals?|charges?|debits?)\b", clause)
        unauthorized = re.search(r"\bunauthori[sz]ed\b|without (?:my )?(?:permission|consent|authori[sz]ation|approval|knowledge|concern)\b|\b(?:didn't|did not|never) (?:make|authori[sz]e|approve)\b", clause)
        theft = re.search(r"\b(?:someone|somebody|hacker|fraudster)\b.{0,45}\b(?:stole|taking|taken|took|withdrew|withdrawing|getting|transferred)\b.{0,30}\b(?:money|funds)\b", clause)
        if money and (unauthorized or theft):
            signals.append({"rule": "reported_unauthorized_transaction", "risk_floor": TRIAGE["unauthorized_transaction_risk_floor"],
                    "intent": "security_concern", "root_cause": "suspected_unauthorized_transaction"})
        credential = re.search(r"\b(?:password|passcode|pin|credentials|account|otp)\b.{0,35}\b(?:leaked|stolen|compromised|hacked|exposed)\b|\b(?:someone|somebody)\b.{0,25}\b(?:accessed|hacked)\b.{0,25}\b(?:my )?account\b|\b(?:stole|leaked|exposed)\b.{0,20}\b(?:my )?(?:password|credentials|pin)\b", clause)
        if credential:
            signals.append({"rule": "reported_credential_or_account_compromise", "risk_floor": TRIAGE["credential_risk_floor"],
                    "intent": "security_concern", "root_cause": "suspected_account_compromise"})
    return max(signals, key=lambda value: value["risk_floor"]) if signals else None


def apply_security_policy(result):
    result = copy.deepcopy(result)
    signal = security_signal(result["review"])
    if signal:
        result["model_outputs"] = {key: copy.deepcopy(result[key]) for key in
                                   ("risk_score", "intent", "sentiment", "root_cause")}
        result["risk_score"] = max(result["risk_score"], signal["risk_floor"])
        for task in ("intent", "root_cause"):
            result[task] = {"label": signal[task], "confidence": None,
                            "source": "security_policy", "alternatives": result[task]["alternatives"]}
        result["security_policy"] = {**signal, "version": POLICY_VERSION}
        result["warnings"].append("Security triage rule applied: reported incident needs urgent human verification. "
                                  "Risk, intent and suspected cause include a rule decision; original model predictions are retained.")
    return result
