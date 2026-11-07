from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process_reviews import analyze_pending
from review_store import ReviewStore, analyses, database_retry, reviews, sqlite_engine, utcnow
from suggestions import suggest_action


class Analyzer:
    model_version = "test-v1"

    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if not text.strip():
            raise ValueError("Empty review")
        if text == "broken":
            raise RuntimeError("secret detail that should not be saved")
        return {"risk_score": 8.2, "intent": {"label": "payment_failure", "confidence": 0.8},
                "sentiment": {"label": "negative", "confidence": 0.9},
                "root_cause": {"label": "api_failure", "confidence": 0.7}}


@pytest.fixture
def store(tmp_path):
    result = ReviewStore(sqlite_engine(tmp_path / "reviews.db"))
    result.initialize()
    yield result
    result.engine.dispose()


def import_texts(store, texts, source="bank"):
    return store.import_rows([{"id": str(i), "review": text, "extra": "original"}
                              for i, text in enumerate(texts)], source, "review", "id")


def test_first_all_then_only_new_and_queryable_results(store):
    import_texts(store, ["good", "bad"])
    analyzer = Analyzer()
    assert analyze_pending(store, analyzer)["completed"] == 2
    assert analyze_pending(store, analyzer)["completed"] == 0
    assert len(analyzer.calls) == 2
    result = import_texts(store, ["good", "bad", "new"])
    assert result["imported"] == 1 and result["duplicates_skipped"] == 2
    assert analyze_pending(store, analyzer)["completed"] == 1
    rows, count, totals, _ = store.dashboard()
    assert count == 3 and totals == {"completed": 3}
    for row in rows:
        assert row["model_version"] == "test-v1"
        assert row["processing_ms"] >= 0 and row["processed_at"]
        assert row["intent_confidence"] == 0.8
        assert row["suggested_action"] and row["suggestion_version"]


def test_original_content_and_all_columns_preserved(store):
    import_texts(store, ["  Keep my text!  "])
    changed = import_texts(store, ["changed text"])
    assert changed["changed_rows_preserved"] == 1
    analyze_pending(store, Analyzer())
    with store.engine.connect() as conn:
        row = conn.execute(select(reviews)).mappings().one()
    assert row["review_text"] == "  Keep my text!  "
    assert '"extra": "original"' in row["raw_row_json"]


def test_same_text_distinct_ids_and_sources_remain_distinct(store):
    import_texts(store, ["same", "same"])
    import_texts(store, ["same"], source="another bank")
    assert store.dashboard()[1] == 3


def test_no_id_import_is_idempotent(store):
    rows = [{"text": "hello"}]
    assert store.import_rows(rows, "source", "text")["imported"] == 1
    assert store.import_rows(rows, "source", "text")["duplicates_skipped"] == 1


def test_invalid_and_failure_do_not_stop_other_reviews(store):
    import_texts(store, ["", "broken", "valid"])
    analyzer = Analyzer()
    summary = analyze_pending(store, analyzer)
    assert summary == {"completed": 1, "invalid": 1, "failed": 1, "skipped": 0}
    assert analyze_pending(store, analyzer)["failed"] == 0
    assert analyze_pending(store, analyzer, retry_failed=True)["failed"] == 1
    rows = store.dashboard()[0]
    failed = next(row for row in rows if row["status"] == "failed")
    assert failed["attempts"] == 2
    assert "secret detail" not in failed["error"]


def test_restart_recovers_expired_work_without_reprocessing_completed(store):
    import_texts(store, ["finished", "interrupted"])
    token = store.claim(1)
    store.finish(1, token, {"status": "completed", "risk_score": 3})
    assert store.claim(2)
    restarted = ReviewStore(store.engine)
    assert not restarted.candidates(0, 2)
    with store.engine.begin() as conn:
        conn.execute(analyses.update().where(analyses.c.review_id == 2).values(lease_until=utcnow() - timedelta(seconds=1)))
    analyzer = Analyzer()
    assert analyze_pending(restarted, analyzer)["completed"] == 1
    assert analyzer.calls == ["interrupted"]


def test_duplicate_claim_and_old_claim_cannot_overwrite_new_result(store):
    import_texts(store, ["text"])
    old = store.claim(1)
    assert store.claim(1) is None
    with store.engine.begin() as conn:
        conn.execute(analyses.update().values(lease_until=utcnow() - timedelta(seconds=1)))
    new = store.claim(1)
    assert new and new != old
    assert not store.finish(1, old, {"status": "failed"})
    assert store.finish(1, new, {"status": "completed"})


def test_new_arrivals_during_run_wait_for_next_click(store):
    import_texts(store, ["first"])
    def arrival(counts):
        import_texts(store, ["first", "next"])
    assert analyze_pending(store, Analyzer(), progress=arrival)["completed"] == 1
    assert analyze_pending(store, Analyzer())["completed"] == 1


def test_risk_filters_boundaries_status_intent_sentiment_and_date(store):
    import_texts(store, ["a", "b", "c", "d", "pending"])
    for review_id, score in enumerate([3.9, 4, 7, 7.1], 1):
        token = store.claim(review_id)
        store.finish(review_id, token, {"status": "completed", "risk_score": score,
                                      "sentiment": "negative", "intent": "complaint"})
    assert store.dashboard({"risk": "Above 7"})[1] == 1
    assert store.dashboard({"risk": "4–7"})[1] == 2
    assert store.dashboard({"risk": "Below 4"})[1] == 1
    assert store.dashboard({"status": "pending"})[1] == 1
    assert store.dashboard({"sentiment": "negative", "intent": "complaint"})[1] == 4
    assert store.dashboard({"start": utcnow() + timedelta(days=1)})[1] == 0


def test_iso_date_import_and_invalid_import_atomicity(store):
    store.import_rows([{"id": "1", "text": "hello", "date": "2026-01-01T05:30:00+05:30"}], "bank", "text", "id", "date")
    assert store.dashboard()[0][0]["received_at"] == datetime(2026, 1, 1)
    with pytest.raises(ValueError):
        store.import_rows([{"id": "2", "text": "hello"}, {"id": "", "text": "oops"}], "bank", "text", "id")
    assert store.dashboard()[1] == 1


def test_database_retries_are_bounded_and_do_not_retry_auth_errors():
    calls = []
    def temporary():
        calls.append(1)
        if len(calls) < 3:
            raise OperationalError("query", {}, Exception(2006, "server gone"))
        return "ok"
    assert database_retry(temporary, sleep=lambda _: None) == "ok"
    assert len(calls) == 3
    calls.clear()
    def outage():
        calls.append(1)
        raise OperationalError("query", {}, Exception(2003, "unreachable"))
    with pytest.raises(OperationalError):
        database_retry(outage, sleep=lambda _: None)
    assert len(calls) == 3
    calls.clear()
    def bad_password():
        calls.append(1)
        raise OperationalError("query", {}, Exception(1045, "denied"))
    with pytest.raises(OperationalError):
        database_retry(bad_password, sleep=lambda _: None)
    assert len(calls) == 1


def test_write_outage_leaves_recoverable_claim(store, monkeypatch):
    import_texts(store, ["hello"])
    original = store.finish
    def disconnected(*args):
        raise OperationalError("update", {}, Exception(2003, "unreachable"))
    monkeypatch.setattr(store, "finish", disconnected)
    with pytest.raises(OperationalError):
        analyze_pending(store, Analyzer())
    assert store.dashboard()[2] == {"processing": 1}
    monkeypatch.setattr(store, "finish", original)
    with store.engine.begin() as conn:
        conn.execute(analyses.update().values(lease_until=utcnow() - timedelta(seconds=1)))
    assert analyze_pending(store, Analyzer())["completed"] == 1


def test_temporary_predict_failure_retries(store, monkeypatch):
    import_texts(store, ["hello"])
    analyzer = Analyzer()
    predict = analyzer.predict
    attempts = []
    def temporary(text):
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError()
        return predict(text)
    monkeypatch.setattr(analyzer, "predict", temporary)
    monkeypatch.setattr("process_reviews.time.sleep", lambda _: None)
    assert analyze_pending(store, analyzer)["completed"] == 1
    assert len(attempts) == 3


def test_low_confidence_suggestions_require_review():
    result = Analyzer().predict("hello")
    result["intent"]["confidence"] = 0.2
    suggestion = suggest_action(result)
    assert suggestion["suggested_action"].startswith("Verify low-confidence")
    assert suggestion["action_owner"] == "Engineering + Finance + Payments Team"


def test_sqlite_persistence_across_connections(tmp_path):
    path = tmp_path / "restart.db"
    first = ReviewStore(sqlite_engine(path))
    first.initialize()
    import_texts(first, ["hello"])
    analyze_pending(first, Analyzer())
    first.engine.dispose()
    second = ReviewStore(sqlite_engine(path))
    assert second.dashboard()[2] == {"completed": 1}
    assert analyze_pending(second, Analyzer())["completed"] == 0
    second.engine.dispose()
