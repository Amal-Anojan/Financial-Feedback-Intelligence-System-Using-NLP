# Keep, archive and publish

This release replaces the current repository tree with transformer-only source and documentation through a normal commit. Git history remains; deleting a file from the current tree does not purge its history. Original local notebooks, reviews, checkpoints and the previous website model are preserved outside the isolated publication checkout.

Keep: all numbered research stages included here, taxonomy, training_quality, runtime Python modules, tests, requirements, documentation, and each selected COMPLETE verified transformer export. Keep database.toml/API keys private. Keep original data, labels, exclusion reports, source/split manifests and exact environment snapshots in an appropriate private research archive.

Archive: old *_best/checkpoint folders with unverifiable mappings, old tokenized datasets and older evaluations. They are not deployment inputs. Delete only disposable caches after verifying their paths. Do not delete the actual MySQL database, runtime SQLite results, raw reviews or the only copy of an experiment.

Excluded from this transformer-only publication: train_web_models.py, web_baseline.joblib, classical inference code, the classical-baseline notebook, baseline evaluation reports, old broad handbooks describing baseline behavior, credentials, actual review data, cached/generated artifacts and incomplete transformer checkpoints.

Model readiness remains blocked on relabeling/auditing and training in a suitable environment. Fixing the training workflow does not magically repair existing weights. Once all four exports pass review, publish those artifacts with LFS or a versioned model release and update the model-selection example to actual release paths.
