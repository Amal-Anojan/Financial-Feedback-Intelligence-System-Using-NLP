"""On-demand CSV import, analysis and stored-result browsing."""
import csv
import io
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from process_reviews import analyze_pending
from review_store import ReviewStore, configured_engine, database_error_message


@st.cache_resource
def load_store():
    store = ReviewStore(configured_engine())
    store.initialize()
    return store


def show_dashboard(load_analyzer):
    st.title("Customer review dashboard")
    st.write("Import your CSV, then analyze when you need to. Results stay in the database for future queries.")
    try:
        store = load_store()
        mode = "MySQL" if store.engine.dialect.name == "mysql" else "local SQLite"
        st.caption(f"Connected to {mode}. No polling or background worker. Dates are shown in UTC.")
        if mode != "MySQL":
            st.info("Local mode works without a server. To connect MySQL, follow MYSQL_SETUP.md and restart the app.")
        import_and_analyze = False
        with st.expander("Import CSV reviews", expanded=True):
            st.caption("All CSV columns are preserved. Use the same source name and stable review ID for later imports. Existing reviews are never overwritten.")
            uploaded = st.file_uploader("Reviews CSV (UTF-8)", type="csv")
            if uploaded:
                try:
                    reader = csv.DictReader(io.StringIO(uploaded.getvalue().decode("utf-8-sig")))
                    columns = reader.fieldnames or []
                    if not columns or len(columns) != len(set(columns)):
                        raise ValueError("The CSV needs a header with unique column names.")
                    rows = list(reader)
                    if any(None in row for row in rows):
                        raise ValueError("Some rows have more fields than the CSV header.")
                    st.info(f"CSV ready: {len(rows):,} rows. Choose the columns below, then click Import and analyze reviews to save and process them.")
                    default_text = "review_text" if "review_text" in columns else "text" if "text" in columns else columns[0]
                    with st.form("import_csv"):
                        source = st.text_input("Source name", value=uploaded.name.rsplit(".", 1)[0])
                        text_col = st.selectbox("Review text column", columns, index=columns.index(default_text))
                        id_col = st.selectbox("Stable review ID column", ["(None)"] + columns,
                                              index=columns.index("source_row_id") + 1 if "source_row_id" in columns else 0)
                        date_col = st.selectbox("Review date column (optional, ISO format)", ["(Use import time)"] + columns)
                        st.caption("Without an ID, identical rows under the same source are treated as duplicates. For a combined CSV, use IDs unique across all its apps, or import each app separately.")
                        import_and_analyze_clicked = st.form_submit_button("Import and analyze reviews", type="primary")
                        import_clicked = st.form_submit_button("Import reviews only")
                    if import_clicked or import_and_analyze_clicked:
                        with st.spinner(f"Importing {len(rows):,} rows…"):
                            result = store.import_rows(rows, source, text_col,
                                None if id_col == "(None)" else id_col,
                                None if date_col == "(Use import time)" else date_col)
                        st.success(f"Imported {result['imported']:,}; skipped {result['duplicates_skipped']:,} existing rows.")
                        import_and_analyze = import_and_analyze_clicked
                        if result["changed_rows_preserved"]:
                            st.warning(f"{result['changed_rows_preserved']} reused IDs had different content. Stored originals were preserved. Check your ID/source mapping.")
                except (ValueError, UnicodeError, csv.Error) as exc:
                    st.error(str(exc))

        _, _, current_totals, _ = store.dashboard(limit=0)
        empty_database = not sum(current_totals.values())
        if empty_database:
            st.info("The database has no reviews yet. Upload a CSV above and click Import and analyze reviews. Selecting a file alone does not save it.")
        left, middle, right = st.columns(3)
        analyze = left.button("Analyze new reviews", type="primary", disabled=empty_database)
        retry = middle.button("Retry failed reviews", disabled=empty_database)
        right.button("Refresh dashboard")
        st.caption("The first run processes all imported reviews. Later runs skip completed and invalid reviews. Retry failed also processes pending reviews. Interrupted work can resume after a 10-minute safety lease.")
        if (analyze or retry or import_and_analyze) and not empty_database:
            with st.spinner("Analyzing reviews; each result is saved as it completes…"):
                analyzer = load_analyzer()
                progress = st.empty()
                result = analyze_pending(store, analyzer, retry_failed=retry,
                    progress=lambda counts: progress.caption(" · ".join(f"{k}: {v}" for k, v in counts.items())))
            if sum(result.values()):
                st.success("Run finished: " + ", ".join(f"{key} {value:,}" for key, value in result.items()))
            else:
                st.info("No eligible reviews to analyze. Completed and invalid reviews are skipped; use Retry failed reviews for failures. Interrupted processing becomes eligible after its 10-minute lease expires.")

        _, _, totals, options = store.dashboard(limit=0)
        for col, label in zip(st.columns(5), ("pending", "processing", "completed", "failed", "invalid")):
            col.metric(label.capitalize(), totals.get(label, 0))
        st.subheader("Stored reviews and analysis")
        cols = st.columns(4)
        risk = cols[0].selectbox("Risk category", ["All", "Above 7", "4–7", "Below 4"])
        sentiment = cols[1].selectbox("Sentiment", ["All"] + options["sentiment"])
        intent = cols[2].selectbox("Intent", ["All"] + options["intent"])
        status = cols[3].selectbox("Processing status", ["All", "pending", "processing", "completed", "failed", "invalid"])
        st.caption("Above 7 = highest attention; 4–7 = important; below 4 = lower priority. These dashboard categories do not change the project's action rules.")
        filters = {key: value for key, value in dict(risk=risk, sentiment=sentiment, intent=intent, status=status).items() if value != "All"}
        if st.checkbox("Filter by review date"):
            dates = st.columns(2)
            start = dates[0].date_input("From date")
            end = dates[1].date_input("To date")
            if end < start:
                st.warning("The end date must be on or after the start date.")
                return
            filters.update(start=datetime.combine(start, datetime.min.time()),
                           end=datetime.combine(end + timedelta(days=1), datetime.min.time()))
        rows, count, _, _ = store.dashboard(filters)
        st.caption(f"Showing {len(rows):,} of {count:,} matching reviews (latest 1,000). All results remain queryable in the database.")
        if rows:
            frame = pd.DataFrame([dict(row) for row in rows])
            st.dataframe(frame, hide_index=True, use_container_width=True)
            # Prevent spreadsheet applications treating review text as formulas.
            export = frame.map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
            st.download_button("Download displayed rows", export.to_csv(index=False), "review_analysis.csv", "text/csv")
        else:
            st.info("No matching reviews. Import a CSV or change the filters.")
        st.caption("Suggested actions require human confirmation; no action is executed. Explicit security incidents can receive a rule-based risk floor and labels. Blank confidence means a rule decision, not model certainty. Original model outputs and policy evidence are retained in review_analysis_evidence. Transformer risk scores are derived from bands, not regression confidence.")
    except Exception as exc:
        st.error(database_error_message(exc))
        st.caption("Setup guide: MYSQL_SETUP.md. Passwords and connection strings are deliberately not displayed.")
