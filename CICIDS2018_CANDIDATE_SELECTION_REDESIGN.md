# CICIDS2018 OWL: analyst candidate-selection redesign (v3/v4)

Follow-up to `CICIDS2018_CDIST_VS_CLUSTERING_ABLATION.md`, which concluded with
clustering chosen as the winning member-inference mechanism. This log covers
the next phase: reducing analyst-labeling effort and targeting *which*
candidates get sent to the analyst, on top of `cicids2018_spider_owl_2.py`.

## Design (v3 -- `cicids2018_spider_owl_3.py`)

Two changes, both scoped to the attack class only (benign self-labeling was
already ~99.9% accurate and was never the problem):

1. **CReST-style class-adaptive purity threshold** (Wei et al., CVPR 2021).
   The shared `mode_value` purity threshold was starving attack self-labeling
   under severe class imbalance. `class_adaptive_mode_value()` (in
   `active_candidate_selection.py`) relaxes the threshold for the attack
   class in proportion to how underrepresented it is in memory, floored at
   60 so it's never meaninglessly permissive.
2. **OOD/diversity-ranked analyst selection**, replacing
   `np.random.choice` for the attack class's analyst-labeling budget:
   novelty score (distance to nearest memory cluster centroid / that
   cluster's radius -- a compressed kNN-OOD, Sun et al. ICML 2022) +
   ensemble disagreement (BALD, Houlsby et al. 2011, contributes nothing
   when `mlps=1`, which is what these runs use) + BADGE-style (Ash et al.,
   ICLR 2020) diversification via k-means++ over a score-ranked shortlist.
   All three implemented and self-test-verified in
   `active_candidate_selection.py` before wiring in.

The analyst *budget size* itself is unchanged (`labels_ratio`-derived, same
formula as before) -- only *which* candidates fill it changed.

## v3 results (seed=1, both ratios) vs. the v2 clustering baseline

| Metric | v2@0.4 | v3@0.4 | v2@0.1 | v3@0.1 |
|---|---|---|---|---|
| Seen AUT | 0.8587 | 0.7232 | 0.7039 | 0.8673 |
| Unseen AUT | 0.4396 | 0.4368 | 0.2384 | 0.6864 |
| All-tasks AUT | 0.5799 | 0.5169 | 0.4201 | 0.6949 |
| Attack self-label accuracy | 7.72% | 0.23% | 1.46% | 36.59% |

**ratio=0.1: a dramatic, consistent win** across every metric. **ratio=0.4:
a real regression** on seen AUT, all AUT, and attack accuracy -- not just
noise, a genuine harmful side effect that needed root-causing before this
design could be trusted.

## RCA: why ratio=0.4 regressed

Traced directly from the logs, not inferred:

- At ratio=0.4, task 6 alone had 78,687 attack candidates accepted through
  the relaxed threshold at only **0.108% actual accuracy** (85 correct).
  Every one of those ~78,600 wrongly-labeled samples was written straight
  into the persistent replay buffer (`labeled_X`/`labeled_y_classname`
  from `owl_data_labeling_strategy` feed directly into
  `memory_update_equal_allocation2` -- confirmed by tracing `train()`).
- Because member-inference's trust signal (cluster purity) is computed
  *from memory's own labels*, this corrupted purity for every subsequent
  task: task 7's `[CLUSTER-MI]` mean purity had fallen to 0.782 (vs. a
  healthy ~0.90+ before), and benign agreements -- previously ~99.9% every
  task, every prior run -- collapsed to exactly 0/1,687,222. Not because
  benign candidates vanished (1.68M candidates were present), but because
  no cluster anywhere cleared the (unchanged) benign threshold once the
  buffer was this contaminated.
- At ratio=0.1, the SAME relaxed threshold value (~0.80-0.83, nearly
  identical to ratio=0.4's) produced far smaller absolute accept volumes
  per task, so this compounding-corruption cascade never got triggered --
  explaining why the identical design change had opposite outcomes at the
  two ratios: the threshold *value* wasn't the differentiator, absolute
  *volume* written into memory was.

## Fix (v4 -- `cicids2018_spider_owl_4.py`)

Decoupled *training signal* from *memory eligibility*:
- The CReST-relaxed threshold's accepted attack self-labels are still used
  for the current task's training data (`labeled_X`/`labeled_y`,
  unchanged) -- keeps the demonstrated accuracy benefit.
- A **parallel check against the original, unrelaxed threshold**
  (`memory_safe_mask`, computed via a second cheap `cluster_based_
  member_inference` call reusing the same `mem_km`/`mem_purity`/
  `mem_radius`) gates what's actually allowed into the persistent buffer.
  Benign self-labels (unchanged threshold) and all analyst-labeled samples
  (human-verified) are always memory-safe; only the *relaxed-threshold-only*
  attack self-labels are excluded from memory while still being used for
  this task's training.
- `train()`'s buffer-memory-update step (`temp_x,temp_y,temp_yname`) now
  reads from `labeled_X[memory_safe_mask]` etc. instead of the full
  relaxed-inclusive set.

Also carried into v4 (built earlier, same session): **BBSE** (Black Box
Shift Estimation, Lipton et al., ICML 2018) replacing the pipeline's frozen
`avg_CI` extrapolation (seen-tasks' average class ratio, carried forward
unchanged into every unseen task) with a per-unseen-task, ground-truth-free
class-prior estimate -- uses the model's own confusion matrix (from the
last seen task, legitimate ground-truth use) plus its predicted-label
distribution on the new unlabeled task to solve for the true class prior.
Validated on synthetic label-shift data: 0.006 error vs. the frozen
average's 0.399 error under a simulated 30%->70% shift. Seen tasks keep
using `avg_CI` unchanged -- that was never the broken part.

Both fixes verified via syntax check + full module-level smoke test before
launch (matching this session's standing verify-before-trust practice).

## Status as of this write-up

v4 launched (seed=1, ratio=0.1 on GPU 0, ratio=0.4 on GPU 1) -- results not
yet in. Once complete, the key question is whether v4 recovers ratio=0.4 to
at least v2-baseline levels while preserving ratio=0.1's large gains.
