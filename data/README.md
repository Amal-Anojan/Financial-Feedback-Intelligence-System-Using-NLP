# Data prerequisites

Raw company reviews, labeler API results, encoded data and tokenized Arrow files are not committed by default. Obtain data under its source terms; no source-data redistribution permission is implied by this repository.

- Banking77: run 1_download.py. Source metadata must provide label names; code refuses to guess integer labels.
- Financial PhraseBank: obtain all-data.csv separately; notebook 3 reads Latin-1, headerless sentiment/text columns. Review the dataset license and attribution.
- App reviews: notebook 2 lists configured app IDs. It creates per-app *_raw_reviews.csv. Required review fields include review_text, source_app, star_rating, recorded_at; platform IDs are retained when available.
- Notebook 3 explicitly loads the configured app exports. If an app has no data, edit the source list/concatenation intentionally rather than silently claiming it was covered.
- Notebook 4 creates labeled_reviews_v3.csv with taxonomy version/hash. Manually audit before notebook 5. Do not substitute old labeled_reviews.csv and assume rules were updated retroactively.
- Notebook 5 writes task CSVs and dataset_manifest.json. Notebook 8 validates those hashes, records exclusions, and writes schema-2 manifests and saved tokenizers. Exclusions are retained as audit artifacts.

The raw data is immutable in the deployment import path. Source_row_id is a labeling-session ID; use a stable original ID or source_review_key for repeated SQL imports. Do not silently change ID strategies for an existing source.
