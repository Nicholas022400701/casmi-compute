# casmi-compute
Generic cheminformatics / mass-spectrometry compute workflows (GitHub Actions matrix jobs, `workflow_dispatch` only).

Rules (binding):
1. Code only: no data, no credentials, no competition-derived files are ever committed. Secrets reach jobs only via `${{ secrets.* }}` → env; never echoed. Logs print counts and timings only (no structures, spectra or record ids).
2. Public-data jobs (metric keys of public structure databases etc.): inputs from public sources or private Kaggle datasets, outputs to private Kaggle datasets; no workflow artifacts unless encrypted.
3. Jobs on non-public inputs: inputs pulled from private Kaggle datasets inside the job over HTTPS, outputs uploaded to private Kaggle datasets; any artifact/cache encrypted (`openssl enc -aes-256-cbc -pbkdf2 -pass env:CASMI_ENC_KEY`).
4. Every job: `timeout-minutes`, idempotent chunk, matrix `max-parallel` ≤ 20, `fail-fast: false`; no PR-triggered workflows; PRs are not accepted.

Workflows: `tiera-keys` (jobs/tiera_keys/worker.py — RDKit metric keys for chunks of a pre-staged structure pool).

`run-script` (jobs/run_script/*.sh — sharded replay of a bundle script, encrypted shard artifacts, ONE collect job publishing to a private Kaggle dataset). **`outDs` is a bare slug** (`casmi-m3b-gha-<suffix>`, never `nicholasooo/<slug>`): the collect job normalises a leading `nicholasooo/` and refuses anything else, asserts the published dataset equals the requested one and fails loudly instead of exiting 0 unverified (fix of 22:4x UTC 2026-09-27). Record of the two mis-slugged landings before the fix: dispatches that passed the full ref produced id `nicholasooo/nicholasooo/<slug>`, which Kaggle resolved to the dataset **`nicholasooo/nicholasooo`** (title `nicholasooo/casmi-m3b-gha-prp1`): v1 = run 36326537525 (CELL=c81+c82, dev1gS, 15:22 UTC 09-27, intended `casmi-m3b-gha-prp1`), v2 = run 36338007146 (CELL=c81+c83, dev1gS F1 probe pair, 18:29 UTC 09-27, intended `casmi-m3b-gha-pf1`); the dataset stays as is (no rename), the run id is in every shard's `_gha_sNN.txt`.
