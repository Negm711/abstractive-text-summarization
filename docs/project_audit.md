# Repository cleanup

The local application, saved login and VS Code launch settings are preserved. Unused model experiments, their compatibility modules and the old GPU smoke-check code are removed from the working tree after backup. Conversion and data preparation implementations live in `scripts/`, with small root entry points.

Training figures use a named, adapter-verified run under `reports/training/qwen3_curated_v2/`. CI compiles current source and scripts and runs dependency-free local-service tests. It does not load the model or certify factual quality.

The uploaded inventory contains 95 files and no exact duplicate groups. An empty `__init__.py` is a valid package marker. One-off update scripts are retired to the local backup area only when their content matches a known delivered version. Modified or unknown scripts are left alone.

The updater checks original file hashes before changing source and backs up overwritten files. Models, datasets, credentials and caches are untouched. Article-bearing historical reports remain on disk but are excluded from public Git content. Full dataset history is not discarded to make the repository appear cleaner.

Remaining release checks: frozen final evaluation, clean-checkout setup, aggregate metric publication and upstream licensing review.
