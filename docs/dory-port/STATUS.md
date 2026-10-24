# STATE — Dory-on-ANE implementation (updated 2026-10-24 03:22)

## Branch state (all local, nothing pushed)
- `integration/dory-ane` @ 6d73b07: base f4ad26d (0.3.5_beta) + DANE-2 (63375a5) + DANE-2 follow-on (6d73b07) + DANE-3 (b0fe754).
- `ticket/DANE-4` @ dd86da8: ANE-safe FP16 module + caches + 34 tests + gate tooling (NOT yet merged; waiting on its 2-loop gate numbers).
- `ticket/DANE-5` @ 4c2ff11: converter (per-function packages), chunk wrappers, chat runtime, meta schema, placement gate, docs, 63 tests (NOT yet merged).
- `ticket/DANE-2` @ 4b73a2d (merged), `ticket/DANE-3` @ 6122b2b (merged).

## References (the one yardstick) — complete
`~/dev/dory-ane-artifacts/refs/` — 10 files, 10.73 GB:
exact loop{1,2,4} (per_loop, ONE ubatch per context), decode loop{1,2,4} batch{1,64} (last_loop), q8_0 exact loop2.
Measured last_loop floors (decode vs same-loop exact, 4094 rows):
loops 1: 1.6e-05 / 99.66 % (1 token), 2.5e-05 / 99.63 % (64)
loops 2: 0.006975 / 96.12 %, 0.002857 / 97.56 %
loops 4: 0.003490 / 97.31 %, 0.002581 / 97.56 %

## Ticket status
| ticket | code | gates |
|---|---|---|
| DANE-2 | MERGED | met: 38 tests, Q8_0 proof 0.000423/98.46 %, floors measured |
| DANE-3 | MERGED | met: loops 1/2/4 KLD -5e-5 (noise floor), top-1 99.66/99.81/99.68 %, greedy 32/32 |
| DANE-4 | committed dd86da8 | folding 3.7e-6 PASS, FP16 preflight PASS, cache-reset PASS, FP16 smoke PASS; prefill2 gate in flight (~3072/8192 tokens done), decode2/decode1/decode4/prefill1/prefill4 not run |
| DANE-5 | committed 4c2ff11 | dry run + plan/metadata PASS; embeddings converted (2 x 640 MiB); input/recurrent/output/lm_head NOT yet converted; placement, KLD, chat, tok/s, compile-cache NOT run |
| DANE-6 | not started | blocked on DANE-5 |
| DANE-7 | not started | blocked on DANE-5 |
| DANE-8 | not started | blocked on DANE-6/7 |

## Environment facts that matter
- **The machine was shared with a concurrent, unrelated build session.** Load oscillated
  between 9 and 1016 and swap reached 41/42 GB. That is why several long conversions were
  restarted rather than finished, and why the ANE gates were run only in exclusive windows.
  It was *not* the cause of the -14 loading failures: those turned out to be (a) a sandboxed
  shell unable to write the Core ML cache and (b) a per-state size ceiling. See
  findings/ane-state-ceiling.md.
- coremltools 8.3 on this OS build: `save_multifunction` yields packages that either fail to load (no Manifest.json) or silently keep only the default function; hence one package per function (DANE-5 pivot, committed).
- Sub-session file policy is workspace-write on `/Users/vladimir/dev/dory-ane`: teammates cannot write worktrees/artifacts, so the Lead applies their patches and commits (`dane5-out/`, `.dane2-followon/`).
- Disk: ~25 GB free. DANE-3's superseded references were deleted (freed 8 GB); its kept candidates are in `artifacts/DANE-3/kept-candidates/`.

## Next actions, in order
1. DANE-4: finish prefill2, report; then hold for decode2 (needs ~20 min of quiet box).
2. DANE-5: convert input/recurrent/output/lm_head, audit; then exclusive window for placement -> KLD -> chat -> tok/s -> compile cache.
3. Merge DANE-4 and DANE-5 into integration.
4. Start DANE-6 and DANE-7 on DANE-5's outputs; each in its own worktree.
5. DANE-8 (sweep + report), then `code-review` on the integration branch and one fixer pass.
6. Close tickets in the tracker (Done), comment the numbers the tickets ask for (DANE-5 tok/s, DANE-6 memory, DANE-8 table), clean up worktrees.


## Update 2026-10-24 05:45 (quiet window opened)
- **DANE-4 bug found and fixed (commit 907cc46)**: full-attention caches were written at the sliding-ring offset instead of the absolute position; invisible below 1024 tokens, catastrophic above. After the fix the decode-mode gates pass by ~500x: prefill2 (full 4094-row set) mean KLD −0.00002 / top-1 99.76 % and 99.56 %; decode2 (256-row window 2048..2303) mean KLD −0.00002 / top-1 99.61 %. Threshold: 0.01 / 98 %.
- **All DANE-4 numbers are MPS** ("ANE-safe FP16 form, evaluated on MPS (Metal), lm_head_parts=16 — not ANE, not CPU"). ANE placement, ANE tok/s and 24K recall are still unproven.
- **DANE-5 conversion progress**: embeddings (2x640 MiB) and lm_head (2x242 MiB) done and audited; input converting; recurrent and output queued; then compile + meta + artifacts. ETA ~07:00-07:30. Recalibrated sizes: coremltools palettizes over groups of 16, so LUT6 = 0.759 B/param; ten packages total ~4.7 GiB (not 8.6 GB).
- Disk 45 GB free; load oscillates 6..200 as the other session comes and goes.
- The audit's hardcoded size floors were a bug (8-bit assumption); they are now calibrated predictions ±15 %, and the same estimator feeds the disk-floor guard.


## Update 2026-10-24 07:10 — all code merged, gates pending
`integration/dory-ane` @ 718dc6f contains ALL seven tickets' code:
- DANE-2 merge 63375a5 + follow-on 6d73b07 (references, scorer, eval set)
- DANE-3 merge b0fe754 (PyTorch FP32 model, tests, greedy fixture)
- DANE-4 merge 85fdece (ANE-safe FP16 module, caches, 40 tests, gate scripts)
- DANE-5 merge a6b5514 (converter, runtime, meta, placement gate, 63 tests) + 8775d39 (package audit)
- DANE-6 merge 7f484cf (24K caches, tile fallback, prefill progress, recall harness)
- DANE-7 merge 718dc6f (OpenAI server, 80 tests; 55 pass on the merged tree)
Merges were done by the Lead because teammate sessions cannot write the worktrees; conflicts resolved in favour of DANE-4's module (DANE-5's branch carried a stale copy of it).

### What is measured vs not
- DANE-3 FP32 model: at the reference noise floor (KLD -5e-5, top-1 99.66/99.81/99.68 at loops 1/2/4), greedy 32/32.
- DANE-4 ANE-safe FP16: all seven gates pass on MPS (KLD -2e-5..-6e-5, top-1 98.44-99.76 %), FP16 preflight PASS, CPU-vs-MPS agreement 2.8e-4.
- DANE-5: packages rebuilt at ctx 4096 / ub 64 (chain job, ~3 h) after DANE-4's cache fix invalidated the 1024 build; placement, KLD, chat, tok/s, compile cache NOT yet run.
- DANE-6: 24K placement probe running; full 24K conversion, recall, tok/s NOT yet run.
- DANE-7: real-model run + Studio NOT yet run (blocked on packages); preflight tooling ready in dane7-out.
- DANE-8: not started.

### Rules still in force
- ANE measurement only in a quiet window (stateful Core ML loads fail with -14 under load; verified healthy at load 5).
- Conversion is opportunistic (load < 400, disk floor 12 GB, one component at a time).
- Teammate sessions are workspace-write on `/Users/vladimir/dev/dory-ane`: they hand over patches, the Lead applies/commits/merges.


## Update 2026-10-24 17:10 — conversion done, ANE gates blocked by host
- **integration/dory-ane @ ad05978**: all eight tickets' code merged, including the full code-review fix pass (position off-by-one in chat+recall, gates that could fail, matched GPU/ANE workloads, tiled 24K converter, ring-write guard, RoPE rows beside the package, real assertions, audit that fails on an unreadable package).
- **Ten packages converted and audited** at ctx 4096 / ub 64, 6.4 GiB, each within 1.6 % of the calibrated size prediction, meta.yaml valid with provenance (four blob shas + toolchain) and Dory's own chat template.
- **BLOCKER (host)**: this Mac cannot build a Core ML execution plan for a stateful mlprogram. 50/50 fresh-process attempts returned -14; a stateless control built a plan but reported every op as 'other' (macOS 27.2 + coremltools 8.3 resolves no device assignment); predict() fails for every package except dory_embeddings_infer, which executed twice. MLModel() constructs lazily, so "it loads" is not evidence of executability.
- **Consequence**: placement, Core ML KLD, chat, decode tok/s and compile-cache timings are UNMEASURED. Numerical correctness stands on DANE-4's MPS gates (7 gates, worst -6e-5 KLD / 98.44 % top-1).
- **Single remaining action**: a reboot. The stateful path flapped all night on an unchanged script (pass 02:10/02:44/15:42; fail 02:37/15:44/15:55/16:23/16:45-17:03); swap >90 % on two days of uptime.
- Resume after reboot: DANE-5's watch (/tmp/dory5-artifacts/logs-ane/recovery.log) will report ok=4/5, then KLD -> tok/s -> chat; then DANE-6 (probe alone, 24K conversion, recall, KLD>16K); then DANE-7 real-model run; then DANE-8 sweep.
- Tracker: DANE-2/3/4 moved to Done; DANE-5/6/7/8 In Progress; DANE-1 closes last.


## Update 2026-10-25 00:55 — packages rebuilt, execution narrowed to one defect
- **All ten packages converted and audited** at ctx 4096 / ub 64 into `/Users/vladimir/dev/dory-ane/dane5-artifacts/4096ctx` (workspace, survives reboots): embeddings 641+641, input 538+538, output 538+538, recurrent 1331+1331, lm_head 243+243 MiB. meta.yaml + conversion_report.json generated from the merged converter.
- **The root cause of the "-14 all night" was found twice over**: (1) a sandboxed shell cannot write ~/Library/Caches/com.apple.CoreML, and Core ML needs that to load a model — the denial surfaces as -14; (2) a stateful model must be given `state = m.make_state()` and `predict(..., state=state)`, otherwise Core ML reports the cache as an unmet input.
- **What still fails, precisely**: with both of those fixed, `lm_head` (243 MiB, no attention) loads, makes state and predicts (5.18 s load, 0.018 s predict, 8 logit heads); every **attention-bearing** chunk (input/output/embeddings prefill, 537-1331 MiB) fails plan building with -14 at load. coremltools 9.0 fails the same way. So the outstanding defect is in the attention/state graph, not in the artifacts and not in the harness.
- Next diagnostic: bisect a minimal attention chunk (one block, tiny context, with and without the mask input, with and without the KV states) to find the op or signature that the plan builder rejects.


## Update 2026-10-25 01:40 — THE ANE CEILING IS PER STATE (24K is reachable)
- All ten packages converted and audited (ctx 4096, ub 64) in `dane5-artifacts/4096ctx` with meta.yaml + conversion_report.json.
- **ANE plan-build bisect (all in `dane5-artifacts/variants/`)**: states, dynamic state reads, dynamic gated cache writes (slice_update + write_state), a 5376-wide softmax, 36 declared states — ALL PASS on CPU_AND_NE. No semantic op is unsupported.
- **The failure is state RESIDENCY, and it is per single state, not per program**:
  - 1 state x 18 MiB -> passes (3/3)
  - 1 state x 36 MiB -> fails (3/3)
  - 2 states x 18 MiB (36 total) -> passes
  - 4 states x 9 MiB (36 total) -> passes
  - 18 states x 1 MiB -> passes
  - 144 MiB as a graph INPUT/OUTPUT (no state) -> passes
- **So 24,576 context is reachable on the ANE**: one layer's K+V at 24K is 96 MiB (over the cap) but one head's K alone is 12 MiB (under it). Design: declare caches per head — 9 full layers x 4 heads x {K,V} = 72 states of 12 MiB, sliding split the same way; total 999 MiB of state, all of it legal. Alternative: host offload of a tensor (works at 144 MiB, costs ~288 MiB of wire per 4K token, ~2 GiB per 24K token).
- Next: pin the exact single-state ceiling (12/16/24/32 MiB probes), then rework the cache declaration from aggregated (18,4,ctx,256) to per-head through converter, runtime and meta.
