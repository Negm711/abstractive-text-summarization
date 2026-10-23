# Verification

The local application has been run on the user workstation with the trained adapter, including authenticated sharing. The Toyota demonstration returned the previously recorded 27-word summary. This is a successful smoke test, not an overall quality score.

For the architecture update, the dependency-free local-service tests and Python compilation are rerun. Training plots are regenerated from the supplied run log and checked against the best-checkpoint metadata. Inference service, prompt policy, UI, runtime, saved-login code and VS Code launch settings are byte-for-byte unchanged by this update.

The update installer checks source hashes before modifying files, saves replaced files, and runs the local-service tests. Known one-off installers are archived only if their hash matches a delivered version. Rerunning the updater is idempotent; rollback preserves subsequent user edits rather than overwriting them.

CUDA execution cannot be rerun in this review environment. After applying the update, run `python run_local_summary.py --check`, then start the app with the existing F5 configuration. Clean-checkout installation and final test evaluation remain release tasks.
