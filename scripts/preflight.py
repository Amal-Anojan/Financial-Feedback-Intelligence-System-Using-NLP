"""Read-only readiness check. Does not connect to a database or run inference."""
import importlib.util
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_analyzer import ROOT
from transformer_analyzer import read_exports

def main():
    missing = [name for name in ("torch", "transformers", "streamlit", "sqlalchemy", "pymysql") if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install the runtime requirements and the appropriate PyTorch CPU/CUDA build.")
    config = Path(os.environ.get("FINSENTIMENT_MODEL_CONFIG", str(ROOT / "transformer_models.json")))
    if not config.is_file():
        print("Missing transformer selection. Train/export/evaluate all tasks, then configure transformer_models.json.")
        return 1
    try:
        _, exports = read_exports(config)
    except (OSError, ValueError, KeyError) as exc:
        print("Model readiness failed:", str(exc))
        return 1
    print("Verified export manifests, evaluation presence, class mappings and weight hashes for:", ", ".join(exports))
    print("Database mode:", "configured in private database.toml" if (ROOT / "database.toml").exists() else "SQLite local default")
    print("Actual inference and database connectivity still require smoke tests.")
    return int(bool(missing))

if __name__ == "__main__":
    raise SystemExit(main())
