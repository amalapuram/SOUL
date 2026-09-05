# CICIDS2018 OWL: cdist vs. clustering member-inference ablation

Log of the investigation comparing the original exhaustive-cdist member-inference
mechanism against the faiss-clustering redesign in `cicids2018_spider_owl_2.py`,
across a 2-ratio x 3-seed sweep. Written up for the repo so this travels with the
code rather than living only in chat history.

## Background

`owl_data_labeling_strategy()` decides, for each self-labeling candidate, whether
to trust the model's prediction based on how well it agrees with the labels
already known in the replay buffer (`memory_X`/`memory_y`). Two mechanisms were
compared for that trust decision:

- **cdist (original)**: exhaustive cosine distance from every candidate to every
  point currently in memory (`scipy.spatial.distance.cdist`), with a per-task
  adaptive distance threshold and majority-vote label.
- **clustering (redesign)**: cluster memory into up to 25 groups (faiss k-means,
  in the model's embedding space), assign each candidate to its nearest cluster,
  and accept that cluster's majority label only if the cluster clears a purity
  threshold AND the candidate is within `radius_margin x cluster_radius`.

## Bugs found and fixed along the way

1. **`utils/metadata.py` typo**: `task2_list` had `'70,'` (trailing comma) instead
   of `'70'`, breaking string-based benign-class matching for CICIDS2018 day 7.
   Caused 100% of task 7's self-labeled samples to be mislabeled. Fixed;
   resolved task 7's self-label accuracy from ~0% to ~99.99%.
2. **Softmax-scale bug**: raw model logits were used where softmax probabilities
   were expected in threshold calibration / Formula-1 checkpoint selection.
   Fixed across all 4 `_spider_owl_neurips2024_2.py` / `_spider_owl_2.py` files.
3. **Unclipped main-task gradient** could explode independently of the
   anchor-loss gradient; fixed with an independent `clip_grad_norm_`.
4. **Threshold shrinkage**: fixed `0.6 * n_pos/(n_pos+K)` cap replaced with pure
   `n_pos/(n_pos+K)` -- the fixed ceiling was needlessly damping well-sampled
   tasks' thresholds after the softmax fix made `raw_thresh` trustworthy again.
5. **cdist ablation's initial CPU implementation was computationally
   infeasible**: a single member-inference call was `cdist(1.84M candidates,
   13,334 memory points)` -- a ~197GB (float64) distance matrix. Rewritten as
   GPU-batched, fully vectorized (`torch.sort`/`cumsum`/masked-sum`, no
   per-candidate Python loop) in `cicids2018_spider_owl_2_oldmembercompare.py`.
   Verified bit-for-bit equivalent to the original CPU logic (threshold counts,
   selection labels, support counts) on synthetic class-conditional data before
   trusting it on the real sweep. Speedup: ~103-164x per member-inference call,
   erasing what had been a ~46-minute single call.
6. **Orchestration bug (not a modeling bug)**: the shell dispatcher used to run
   the 12-job seed x ratio sweep unattended incremented its queue index inside
   a `$(...)` command substitution, which runs in a subshell -- the index never
   persisted, so it silently relaunched the same job (clustering seed=3,
   ratio=0.4) four times over ~4 hours before being caught. Fixed by moving
   queue-position tracking to a plain state file read/written directly in the
   main loop (no subshell in the path), and verified the fix's queue-advance
   logic in isolation before trusting it unattended again.

## Sweep design

2 label ratios (0.4, 0.1) x 3 seeds (1, 2, 3) x 2 mechanisms (cdist, clustering)
= 12 runs, `training_cutoff=5` (tasks 0-4 supervised warm-up, tasks 5-9
self-labeled/OWL). ratio=0.1 was chosen as the second point specifically because
it's closer to OWL's actual motivating regime (scarce analyst labels) than a
ratio near 0.4 or 0.6 would be.

## Results (mean +/- SD across 3 seeds; see caveats below)

| Metric | cdist@0.4 | cluster@0.4 | cdist@0.1 | cluster@0.1 |
|---|---|---|---|---|
| Seen attack PR-AUC-AUT | 0.7772 +/- 0.1028 | 0.7556 +/- 0.1182 | 0.6656 +/- 0.0935 | 0.6856 +/- 0.0762 |
| Unseen attack PR-AUC-AUT | 0.6467 +/- 0.0534 | 0.6179 +/- 0.1855 | 0.2391 +/- 0.2213 | 0.4435 +/- 0.1783 |
| All-tasks attack PR-AUC-AUT | 0.6572 +/- 0.0735 | 0.6306 +/- 0.0466 | 0.4208 +/- 0.1714 | 0.5210 +/- 0.1027 |
| Self-labeled benign (median, n=3) | ~3.9M (skewed, see below) | ~5.1M (skewed) | ~1.4M | ~1.7M |
| Self-labeled attack (median, n=3) | 5,216 | **285** (one outlier seed at 38,740) | 2,350 | 0 |
| Analyst-labeled attack | 191,239 +/- 9,310 | 169,086 +/- 29,532 | 129,853 +/- 2,628 | 128,555 +/- 1,214 |
| Benign self-label accuracy | 99.93% +/- 0.06 | 99.85% +/- 0.11 | 99.69% +/- 0.17 | 99.80% +/- 0.13 |
| Attack self-label accuracy | 0.13% +/- 0.18 (n=2/3 valid) | 15.12% +/- 10.47 (n=2/3 valid) | **0.00% exactly, all 3 seeds** | 1.46% (n=1/3 valid) |
| Wall-clock (seed=1) | 7,871s | 7,790s | ~7,700s both | ~7,700s both |

**Caveat on mean+/-SD for count metrics**: self-labeled-attack counts are
heavily right-skewed with small n (e.g. clustering@0.4's three seeds were
[285, 38740, 0]) -- mean+/-SD is a misleading summary there (SD exceeds the
mean, which is nonsensical for a non-negative count); median or raw per-seed
values are more honest for those specific rows.

## Analysis

- **Speed is no longer a differentiator.** The original motivation for the
  clustering redesign (a ~103-193x member-inference speedup) is moot once cdist
  was GPU-vectorized -- both mechanisms now finish full 10-task runs within ~1%
  of each other.
- **AUT accuracy is not statistically separable at n=3, at either ratio.** Every
  gap is comparable to or smaller than its SD. At ratio=0.4, point estimates
  nominally favor cdist; at ratio=0.1, they favor clustering by a wider margin.
  Neither is proven at this sample size.
- **The cleanest signal: attack self-label accuracy.** cdist produced 0.00%
  attack accuracy (or zero attempts) in every one of its 6 runs across both
  ratios -- a fully consistent negative result. Clustering is inconsistent
  (sometimes zero attempts, sometimes double-digit accuracy: 7.72%, 22.53%,
  1.46%) but has never been observed to do *worse* than cdist when it does
  attempt attack self-labeling.
- **Analyst-attack-labeling burden** is directionally lower for clustering
  across every seed observed, though the margin varies a lot seed to seed.
- **ratio=0.1 matters more than ratio=0.4 for OWL's actual premise** (scarce
  labels): clustering's largest edge shows up exactly there.
- One clustering@0.4 run (seed=2) self-labeled 38,740 attack samples --
  an order of magnitude above every other run -- worth a dedicated look, not
  just averaging away, since it could reveal a real reproducible mechanism
  (a genuinely well-formed attack cluster) rather than noise.

## Status as of this write-up

All 12 sweep runs complete. **No final mechanism verdict has been declared or
acted upon** -- per explicit instruction, this is being held for manual review
before any implementation proceeds.

### Planned next step (pending review, not yet started)

If clustering is confirmed as the better mechanism, or even independently of
that verdict: replace the current `np.random.choice`-based analyst-selection
step (identical, untargeted, in both mechanisms) with a SOTA-grounded
candidate-selection redesign, scaffolded (and self-test-verified) in
`active_candidate_selection.py`:
- CReST-style (Wei et al., CVPR 2021) class-adaptive acceptance threshold to
  reduce total analyst-labeling volume, specifically fixing the attack-class
  starvation caused by one shared global confidence threshold.
- kNN-OOD (Sun et al., ICML 2022) or cluster-centroid-distance novelty scoring
  for whichever mechanism wins, to rank the analyst's queue by how genuinely
  out-of-distribution a candidate is, rather than random sampling.
- BALD (Houlsby et al., 2011) ensemble-disagreement scoring, reusing the
  existing multi-learner (`mlps>1`) setup.
- BADGE (Ash et al., ICLR 2020) diverse-batch selection so the analyst's queue
  isn't a redundant cluster of near-identical novel points.

If cdist is confirmed as the better mechanism instead: fall back to a hybrid
IVF-style design (cluster as a cheap coarse pre-filter, exact cosine cdist only
within the assigned cluster's members) to recover cdist's exact-neighbor
accuracy without the O(candidates x full memory) cost.
