# Evidence, decisions and limits

Reviewed 8 November 2026 against primary documentation. These references support engineering methods, not a guarantee that the project's labels or scores are correct.

| Decision | Evidence | Project interpretation / limit |
|---|---|---|
| Four transformer classifiers with explicit class maps | [Hugging Face sequence classification](https://huggingface.co/docs/transformers/tasks/sequence_classification), [ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert) | Preserve num_labels/id2label/label2id from the exact retained training data. A mismatch is a data/model compatibility failure, not a cosmetic config change. |
| Separate train/validation/test; do not fit on test | [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [AWS split guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-operations-planning/splits-leakage.html) | Group normalized identical text before splitting. Use validation for model selection. A time/app held-out human-labeled set is still required for deployment claims. |
| Avoid augmentation/resampling before splits | [imbalanced-learn pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) | Do not duplicate rare examples to create apparent validation/test support. No guarantee of accuracy follows from the chosen minimum of 10. |
| Smaller microbatches, accumulation, memory controls | [Trainer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer) | Start at batch 8, accumulate 4; early stop on validation. This is an adjustable configuration, not a benchmarked optimum. |
| Prioritize reported security incidents by impact and context | [NIST SP 800-61r3](https://csrc.nist.gov/pubs/sp/800/61/r3/final) | Suspected account compromise/unauthorized funds warrant urgent human review. NIST does NOT prescribe this project's numerical 9/10 scores. |
| Store large trained artifacts outside regular Git objects | [GitHub large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) | GitHub blocks regular Git files above 100 MiB. Existing transformers are about 598 MB each; use LFS or a model artifact release, not plain git add. |

## Are the rules and risk numbers correct?

Not established. They are a versioned, provisional internal prioritization policy. Correctness requires independent human annotation and measured false positives/negatives, plus company approval of priorities. No internet source can validate a label or exact score for every review in this dataset.

One source of truth is taxonomy.json. Sentiment is predicted separately and no longer adds an automatic +1 to risk: negative tone alone is not evidence of greater loss or compromise. Non-security combinations are capped at 8; reported credential compromise has a policy floor of 9, and reported unauthorized transactions 10. Those thresholds can be revised after validation, but changing taxonomy changes its fingerprint and requires an explicit data/model compatibility review.

Combination consistency: a suspected security-incident cause requires security_concern intent; invalid combinations from the external labeler are rejected. Incident reports take priority over simultaneous closure/refund requests. Mentioning account/password is insufficient. Negated and hypothetical reports do not establish compromise. A blocked login, developer-option restriction or KYC request does not prove fraud or a legal violation. Unknown causes may remain other rather than inventing a technical diagnosis. Historical Banking77 mappings remain weak labels and require manual auditing.

The risk transformer learns five bands from these weak policy labels. It is NOT independent proof that the policy is correct. Its displayed score is the expected value of configured band representatives. This ordinal-to-number conversion should not be reported as a trained regression model, probability of fraud, probability of loss, or regulatory score.

Rules run after inference and preserve raw predictions. Their selected labels have null model confidence. They are limited English patterns and may miss complex negation, historical/resolved incidents, mixed phrasing and Sinhala/Tamil transliteration. Never describe the regex rules as general fraud detection. Measure both the raw model and final policy-adjusted output on the same independent challenge set.

## Minimum evaluation before publishing model weights

- Independently review labels, including security and non-security controls; record the audit sample and disagreements.
- Report per-class precision/recall/F1, macro F1, confusion matrices and class support; do not hide excluded labels.
- Evaluate risk-band confusion and under-prioritization, not only accuracy. Evaluate displayed-score behavior separately.
- Use validation for calibration/threshold choices, then a frozen human-labeled test set. Confidence is not calibrated merely because softmax returns a number.
- Check same-user/templates, near-duplicate and app/time leakage; exact-text dedup alone is insufficient.
- Banking77's original train/test sets are pooled into this custom research dataset. Results are NOT official Banking77 benchmark scores.
- Do not infer multilingual performance from an English-character scraping filter.
- Verify weight/tokenizer/config hashes, local inference, MySQL persistence and restart behavior with the final artifacts.
