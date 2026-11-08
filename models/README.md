# Transformer artifacts

No deployable trained weights are included yet. Local historical checkpoint inspection confirmed classifier.weight [26,768] and classifier.bias [26] for root cause, while the available encoder has 25 labels. Current retained-class counts can change after exclusions. Recovering label order cannot be done from these tensor shapes; fresh training with the manifest is the supported correction.

Run notebook 8 after audited data preparation. Keep each complete timestamped export: safetensors weights/shards, config.json, tokenizer files, deployment.json and evaluation.json. Do not publish optimizer.pt, scheduler.pt, rng_state.pth, raw tokenized datasets or generic unlabeled historical checkpoints as ready-to-use models.

Before publishing weights: check label provenance, licenses/data permissions, evaluation and a real inference smoke test. Use Git LFS for *.safetensors (configured in .gitattributes) or a versioned model release. Example after validation:

```powershell
git lfs install
git add -f models/transformer_exports/sentiment/ACTUAL_TIMESTAMP
# Repeat only for the three other verified complete exports.
git lfs ls-files
```

Do not use a broad git add -f models: that would include old invalid checkpoints. LFS has account-specific quota/billing considerations. This source release must not be described as containing four newly trained models.
