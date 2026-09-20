# casmi-compute
Generic cheminformatics / mass-spectrometry compute workflows (GitHub Actions matrix jobs, `workflow_dispatch` only).

Rules (binding):
1. Code only: no data, no credentials, no competition-derived files are ever committed. Secrets reach jobs only via `${{ secrets.* }}` → env; never echoed. Logs print counts and timings only (no structures, spectra or record ids).
2. Public-data jobs (metric keys of public structure databases etc.): inputs from public sources or private Kaggle datasets, outputs to private Kaggle datasets; no workflow artifacts unless encrypted.
3. Jobs on non-public inputs: inputs pulled from private Kaggle datasets inside the job over HTTPS, outputs uploaded to private Kaggle datasets; any artifact/cache encrypted (`openssl enc -aes-256-cbc -pbkdf2 -pass env:CASMI_ENC_KEY`).
4. Every job: `timeout-minutes`, idempotent chunk, matrix `max-parallel` ≤ 20, `fail-fast: false`; no PR-triggered workflows; PRs are not accepted.

Workflows: `tiera-keys` (jobs/tiera_keys/worker.py — RDKit metric keys for chunks of a pre-staged structure pool).
