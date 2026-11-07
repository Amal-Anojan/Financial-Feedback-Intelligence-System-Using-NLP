import ast
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_quality import clean_task, split_task
from security_policy import security_signal, apply_security_policy
from transformer_analyzer import read_exports


def test_rare_classes_duplicates_conflicts_and_compact_split():
    frame = pd.DataFrame({"text": [f"review {label} {i}" for label in ["a","b"] for i in range(12)],
                          "label": [label for label in ["a","b"] for i in range(12)]})
    extras = pd.DataFrame({"text": ["singleton", " REVIEW A 0 ", "conflict", "CONFLICT", None],
                           "label": ["rare", "a", "a", "b", "a"]})
    clean, excluded = clean_task(pd.concat([frame,extras],ignore_index=True),"label")
    assert set(excluded.exclusion_reason) == {"rare_class","duplicate_text","conflicting_labels","invalid_text_or_label"}
    assert len(clean)==24
    parts = split_task(clean,"label")
    assert all(set(part.label)=={"a","b"} for part in parts.values())
    assert sum(map(len,parts.values()))==24
    assert not set(parts["train"].text)&set(parts["test"].text)
    assert parts["test"].equals(split_task(clean,"label")["test"])


def test_all_rare_fails_with_actionable_message():
    with pytest.raises(ValueError,match="collect more data"):
        clean_task(pd.DataFrame({"text":["a review","b review"],"label":["a","b"]}),"label")


@pytest.mark.parametrize("review,floor", [
    ("my account password leaked",9),
    ("someone getting my money from my account without my concern",10),
    ("Someone withdrew money from my account without my permission",10),
    ("My password leaked. Someone took money from my account",10),
    ("My password was not leaked but someone took my money without consent",10),
    ("I didn't make this card payment",10),
])
def test_incident_triage(review,floor):
    assert security_signal(review)["risk_floor"]==floor


@pytest.mark.parametrize("review", [
    "How can I prevent my password being leaked?", "My password was not leaked",
    "There were no unauthorized transactions", "I forgot my password",
    "I want to close my account", "The transfer I made failed", "What if my account is hacked?",
    "Developer options are blocked", "I transferred my money with permission",
])
def test_no_incident_from_keywords_alone(review):
    assert security_signal(review) is None


def test_policy_retains_model_values_without_invented_confidence():
    original={"review":"my account password leaked","risk_score":5.5,"warnings":[],
              **{task:{"label":"old", "confidence":.9,"alternatives":[]} for task in ["intent","sentiment","root_cause"]}}
    result=apply_security_policy(original)
    assert original["risk_score"]==5.5
    assert result["model_outputs"]["risk_score"]==5.5
    assert result["risk_score"]==9
    assert result["intent"]["confidence"] is None


def test_adapter_rejects_implicit_score_conversion(tmp_path):
    path=tmp_path/"config.json";path.write_text('{}')
    with pytest.raises(ValueError,match="explicitly accept"):
        read_exports(path)


def test_adapter_checks_labels_and_weight_integrity(tmp_path):
    import hashlib
    exports={}
    for task in ("sentiment","intent","root_cause","risk"):
        folder=tmp_path/task;folder.mkdir();exports[task]=task
        names=["LOW","HIGH"] if task=="risk" else ["a","b"]
        (folder/"model.safetensors").write_bytes(b"test fixture, never loaded as a model")
        meta={"schema_version":2,"task":task,"label_names":names,"text_preprocessing":"as_is","max_length":128,
              "taxonomy_sha256":hashlib.sha256((Path(__file__).resolve().parents[1]/"taxonomy.json").read_bytes()).hexdigest(),
              "weight_sha256":{"model.safetensors":hashlib.sha256((folder/"model.safetensors").read_bytes()).hexdigest()}}
        (folder/"deployment.json").write_text(json.dumps(meta))
        (folder/"evaluation.json").write_text(json.dumps({"accuracy":0.5,"macro_f1":0.5,"test_rows":10}))
        (folder/"config.json").write_text(json.dumps({"id2label":{str(i):name for i,name in enumerate(names)}}))
    config=tmp_path/"selection.json"
    config.write_text(json.dumps({"exports":exports,"risk_score_method":"expected_band_midpoint"}))
    assert len(read_exports(config)[1])==4
    (tmp_path/"intent"/"model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError,match="checksum mismatch"):
        read_exports(config)


def test_labeling_normalization_uses_local_policy_and_rejects_missing_fields():
    import typing
    import re
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/"4_Taxonomy_and_Labeling.ipynb").read_text(encoding="utf-8"))
    namespace={"Path":Path,"json":json,"re":re,"pd":pd,"os":__import__("os"),
               "Dict":typing.Dict,"List":typing.List,"Optional":typing.Optional,"Any":typing.Any,
               "API_KEYS_PATH":None}
    for index in [5,6,8]:
        exec(''.join(notebook["cells"][index]["source"]),namespace)
    normalize=namespace["normalize_result"]
    item={"sentiment":"negative","intent":"security_concern","root_cause":"suspected_unauthorized_transaction", "risk_score":1,"confidence":.8}
    assert normalize(item)["risk_score"]==10
    with pytest.raises(ValueError,match="Incomplete"):
        normalize({"intent":"security_concern"})
    with pytest.raises(ValueError,match="Invalid"):
        normalize(dict(item,confidence=float('nan')))


def test_all_notebook_code_parses():
    for path in Path(__file__).resolve().parents[1].glob("*.ipynb"):
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"]=="code": ast.parse(''.join(cell["source"]))


def test_policy_evidence_is_committed_with_analysis(tmp_path):
    from sqlalchemy import select
    from review_store import ReviewStore, sqlite_engine, analysis_evidence
    store=ReviewStore(sqlite_engine(tmp_path/"audit.db"));store.initialize()
    from review_store import reviews
    from datetime import datetime
    with store.engine.begin() as connection:
        key=connection.execute(reviews.insert().values(source_key="x",source="test",review_text="test",raw_row_json="{}",
            received_at=datetime.now(),imported_at=datetime.now())).inserted_primary_key[0]
    token=store.claim(key)
    assert not store.finish(key,"wrong-token",{"status":"completed","evidence":{"rule":"wrong"}})
    assert store.finish(key,token,{"status":"completed","evidence":{"model_score":5.5,"rule":"test"}})
    with store.engine.connect() as connection:
        details=connection.execute(select(analysis_evidence.c.details_json)).scalar_one()
    assert json.loads(details)=={"model_score":5.5,"rule":"test"}
    store.engine.dispose()
