from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_backend import load_configured_analyzer
from review_analyzer import validate_review

@pytest.mark.parametrize("value",[None,"", "   ","123", "!", "a"*5001])
def test_invalid_input(value):
    with pytest.raises(ValueError): validate_review(value)

def test_no_baseline_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv("FINSENTIMENT_MODEL_CONFIG",str(tmp_path/"missing.json"))
    with pytest.raises(FileNotFoundError,match="Transformer setup"):
        load_configured_analyzer()

def test_ui_explains_missing_models(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    import dashboard
    from review_store import ReviewStore,sqlite_engine
    store=ReviewStore(sqlite_engine(tmp_path/"test.db"));store.initialize()
    monkeypatch.setattr(dashboard,"load_store",lambda:store)
    monkeypatch.setenv("FINSENTIMENT_MODEL_CONFIG",str(tmp_path/"missing.json"))
    st.cache_resource.clear()
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30).run()
    app.radio[0].set_value("Manual review test").run()
    app.text_area[0].input("My account password leaked")
    app.button[0].click().run()
    assert not app.exception
    assert "Transformer setup is incomplete" in app.error[0].value
    assert not app.metric
    st.cache_resource.clear();store.engine.dispose()
