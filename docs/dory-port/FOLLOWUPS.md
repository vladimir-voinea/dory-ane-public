# Follow-ups for the final code-review pass (Lead's running list)

1. **`anemll/eval/kld.py` CLI ergonomics**: `--min-top1` takes a FRACTION (0.93) but everyone types a percent (`99` → "top-1 99.682% < 9900.000%"). Accept both forms (values > 1 are percentages) or fail with a message that says which unit is expected. Same question for `--max-kld`. Owner: DANE-2 code, fix in the review pass.
2. **Branch hygiene**: DANE-3 merged `ticket/DANE-2` directly into `ticket/DANE-3` before DANE-2 reached the integration branch (commit 5fc3d7b). Harmless (DANE-2 is now merged as 63375a5) but the protocol is: merge `integration/dory-ane`, never another ticket branch. Check the final history for other cross-ticket merges.
3. **Stray `uv.lock`** keeps appearing untracked in every worktree (created by an agent's uv run). Confirm none of them is committed, and add `uv.lock` to .gitignore if the project does not want it.
4. **`tests/fixtures/dory_eval` provenance**: Gutenberg #1342, public domain, 2x4096 contexts — check PROVENANCE.json is complete enough for a stranger to rebuild the fixture.
5. **Numerical claims in docs**: several KLD figures in `docs/dory-eval.md` and the tickets were produced on different texts at different times. At the end, make sure every number quoted in the repo's docs comes from the shared `refs/` yardstick, or is labelled with the text it used.

6. **`.gitignore` `uv.lock`**: the repo's pre-merge hook runs `uv run lefthook`, which writes a `uv.lock` into whatever worktree merges. Never tracked, but it dirties pre-flight. Add it.
7. **Decode reference sidecars lack the `-fa on` pin** (exact ones have it). Re-scoring reproduces the numbers, but a `--force` rebuild would re-seed the floor values. Pin it in the builder + docs.
8. **`python -m anemll.eval.kld` prints a RuntimeWarning** from an eager import in `anemll/eval/__init__.py`; make the import lazy so later tickets' logs are clean.
9. **Safetensors "Hub checksum" is unproven**: HF exposes no sha256 for that repo's non-LFS files; the merger verified size only. Say so in the docs instead of claiming checksum verification.
