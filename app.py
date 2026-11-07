"""Run with: python -m streamlit run app.py"""
import json
import logging

import pandas as pd
import streamlit as st

from review_analyzer import MAX_REVIEW_LENGTH, validate_review
from model_backend import load_configured_analyzer

st.set_page_config(page_title="FinSentiment | Review intelligence", page_icon="💬", layout="wide")


@st.cache_resource
def load_analyzer():
    return load_configured_analyzer()


page = st.sidebar.radio("View", ["Review dashboard", "Manual review test"])
if page == "Review dashboard":
    from dashboard import show_dashboard
    show_dashboard(load_analyzer)
    st.stop()

st.caption("FINSENTIMENT / CUSTOMER REVIEW INTELLIGENCE")
st.title("Turn customer feedback into insight")
st.write("Enter a financial app review to estimate its risk score, intent, sentiment, and likely root cause.")
with st.sidebar:
    st.header("About this analyzer")
    st.write("Local review models with explicit security-incident triage. The result identifies the active model.")
    st.caption("Manual tests are not saved. CSV reviews and dashboard results are stored in your configured database.")
    st.info("Risk is a 0–10 triage score, not a probability. It comes from transformer band conversion, with disclosed floors for explicit security incidents. Human verification is required.")
    st.caption("English reviews work best. Predictions need human review, especially for rare labels or mixed issues.")

with st.form("review_form"):
    review = st.text_area("Customer review", height=170, max_chars=MAX_REVIEW_LENGTH,
                          placeholder="Example: My transfer failed but the money was deducted, and support has not replied.")
    submitted = st.form_submit_button("Analyze review", type="primary")

if submitted:
    st.session_state.pop("analysis", None)
    try:
        validate_review(review)
        with st.spinner("Analyzing your review…"):
            st.session_state.analysis = load_analyzer().predict(review)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Review analysis failed")
        st.error("The models could not be loaded or run. Check the selected backend, export paths, packages and server log. Follow the transformer export instructions in README.md.")

if "analysis" in st.session_state:
    result = st.session_state.analysis
    st.subheader("Review analysis")
    st.caption("Results apply to the last submitted review:")
    st.write(result["review"])
    st.caption("Active model: " + result["model"])
    columns = st.columns(4)
    columns[0].metric("Estimated risk score", f"{result['risk_score']:.1f} / 10")
    for column, task, title in zip(columns[1:], ("intent", "sentiment", "root_cause"), ("Intent", "Sentiment", "Likely root cause")):
        column.metric(title, result[task]["label"].replace("_", " ").capitalize())
        confidence = result[task]["confidence"]
        column.caption("Security triage rule; no model confidence" if confidence is None else f"Model confidence: {confidence:.0%}")
    st.progress(result["risk_score"] / 10, text="Estimated review severity")
    for warning in result["warnings"]:
        st.warning(warning)
    if "model_outputs" in result:
        with st.expander("Original model prediction before security triage"):
            st.json(result["model_outputs"])
    with st.expander("Alternative labels and confidence"):
        st.caption("Confidence values are uncalibrated model probabilities; they do not guarantee correctness.")
        rows = [{"Output": task.replace("_", " "), "Label": item["label"], "Confidence": f"{item['confidence']:.1%}"}
                for task in ("intent", "sentiment", "root_cause") for item in result[task]["alternatives"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Download result as JSON", json.dumps(result, indent=2), "review_analysis.json", "application/json")

with st.expander("Model quality and evaluation"):
    st.write("All four predictions use separately fine-tuned ModernBERT models. See evaluation.json in each selected export. No trained exports are bundled until training and evaluation are complete.")
    st.caption("Confidence is uncalibrated. Risk is a band-derived triage score; any security policy adjustment is shown separately. Test metrics against weak labels do not establish real-world fraud detection accuracy.")
