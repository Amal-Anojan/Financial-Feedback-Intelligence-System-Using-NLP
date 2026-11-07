"""One-shot processing: python process_reviews.py [--retry-failed]. Never polls."""
import argparse
import time

from model_backend import load_configured_analyzer
from review_store import ReviewStore, configured_engine
from suggestions import suggest_action


def analyze_pending(store, analyzer, retry_failed=False, progress=None):
    maximum = store.max_review_id()
    after = 0
    summary = {"completed": 0, "failed": 0, "invalid": 0, "skipped": 0}
    while rows := store.candidates(after, maximum, retry_failed):
        for row in rows:
            after = row["id"]
            token = store.claim(after, retry_failed)
            if not token:
                summary["skipped"] += 1
                continue
            start = time.perf_counter()
            values = {"model_version": analyzer.model_version}
            try:
                for attempt in range(3):
                    try:
                        result = analyzer.predict(row["review_text"])
                        break
                    except (TimeoutError, ConnectionError):
                        if attempt == 2:
                            raise
                        time.sleep(0.25 * (2 ** attempt))
                values.update(status="completed", risk_score=result["risk_score"])
                for task in ("intent", "sentiment", "root_cause"):
                    values[task] = result[task]["label"]
                    values[f"{task}_confidence"] = result[task]["confidence"]
                values.update(suggest_action(result))
                values["evidence"] = {key: result[key] for key in
                    ("model", "model_outputs", "security_policy", "risk_band", "risk_score_method", "warnings") if key in result}
                values["error"] = None
            except ValueError as exc:
                values.update(status="invalid", error=str(exc)[:1000])
            except Exception as exc:
                # Do not persist raw exception text: it can contain credentials or review data.
                values.update(status="failed", error=f"{type(exc).__name__}: prediction failed; retry this review after checking the model.")
            values["processing_ms"] = round((time.perf_counter() - start) * 1000, 2)
            if store.finish(after, token, values):
                summary[values["status"]] += 1
            else:
                summary["skipped"] += 1
            if progress:
                progress(summary.copy())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    try:
        store = ReviewStore(configured_engine())
        store.initialize()
        print(analyze_pending(store, load_configured_analyzer(), args.retry_failed))
    except Exception as exc:
        raise SystemExit(f"Processing stopped ({type(exc).__name__}). Check database settings and availability. Saved results are retained; retry after recovery. Unfinished claims become eligible after 10 minutes.") from None
