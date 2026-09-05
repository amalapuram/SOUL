"""
active_candidate_selection.py

Mechanism-agnostic scaffolding for the analyst-labeling candidate-selection
redesign (replaces owl_data_labeling_strategy's current
`np.random.choice(remaining_X_indices, ...)` analyst-selection step, which is
identical -- and identically un-targeted -- in both cicids2018_spider_owl_2.py
and cicids2018_spider_owl_2_oldmembercompare.py).

Built BEFORE the cdist-vs-clustering seed x ratio sweep concludes, so it
deliberately does not import or modify either experiment file -- it is not
wired into any running pipeline yet. Once the sweep names a winner, wiring in
is: (1) call the matching compute_novelty_score_* function from inside
owl_data_labeling_strategy's analyst-selection block, (2) call
ensemble_disagreement_score using the existing `models` list (student_model1/
student_model2/student_supervised, mlps>1), (3) call build_analyst_queue to
get the final index set, (4) replace the current np.random.choice calls with
those indices. Everything below already works standalone -- verified with a
synthetic correctness check at the bottom of this file (run directly:
`python active_candidate_selection.py`).

SOTA grounding (see conversation record for full rationale):
  - class_adaptive_mode_value: CReST (Wei et al., CVPR 2021) -- class-
    rebalanced pseudo-label acceptance, fixes minority/attack-class
    starvation under one shared global confidence threshold.
  - compute_novelty_score_cdist: Sun et al.'s kNN-OOD (ICML 2022) in its
    exact original form (distance to k nearest individual neighbors, no
    compression).
  - compute_novelty_score_clustering: the same idea adapted to compressed
    memory (distance to nearest cluster centroid / that cluster's radius) --
    used only if clustering wins, since cdist's per-point exact form is
    strictly more faithful to the source paper when available.
  - ensemble_disagreement_score: BALD (Houlsby et al., 2011) mutual-
    information-based disagreement, applied to the multi-learner setup this
    codebase already has (mlps>1) with no new model infrastructure.
  - diverse_topk_selection / build_analyst_queue: BADGE (Ash et al., ICLR
    2020) -- score-ranked shortlist, then k-means++ seeding for a diverse
    (not redundant) final batch.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. CReST-style class-adaptive acceptance threshold (reduces TOTAL analyst
#    labeling volume by letting genuinely scarce classes clear the gate at a
#    lower confidence bar than the majority class, instead of one shared
#    global mode_value for both).
# ---------------------------------------------------------------------------
def class_adaptive_mode_value(base_mode_value, class_pos_count, class_total_count,
                               min_mode_value=60.0, relax_power=0.25):
    """
    `base_mode_value` is the existing single global threshold (the CLI
    --mode_val value, 0-100 scale). Classes scarce in memory (small
    class_pos_count relative to class_total_count) get a LOWER required-
    majority-percentage, so genuine minority-class (attack) neighborhoods
    aren't rejected just for being smaller/noisier than the majority
    (benign) class's neighborhoods -- same fix CReST applies to pseudo-label
    acceptance under class imbalance, adapted here to the purity/majority-
    vote gate already used by both member-inference mechanisms.

    relax_power controls how aggressively the threshold relaxes as a class
    gets scarcer (frequency_ratio -> 0): 0.25 (default) gives a gentle
    relaxation for moderate imbalance, steep only once a class is severely
    underrepresented, matching CReST's own reported preference for a
    sub-linear (not linear) relaxation curve. Never drops below
    min_mode_value -- a majority vote weaker than that isn't a meaningful
    "confident neighborhood" under any class balance.
    """
    frequency_ratio = class_pos_count / max(class_total_count, 1)
    frequency_ratio = min(max(frequency_ratio, 0.0), 1.0)
    relaxed = base_mode_value * (frequency_ratio ** relax_power)
    return max(relaxed, min_mode_value)


# ---------------------------------------------------------------------------
# 2a. [CLUSTERING-WINS BRANCH] Novelty score via nearest-cluster-centroid
#     distance, normalized by that cluster's own radius.
# ---------------------------------------------------------------------------
def compute_novelty_score_clustering(candidates, model, embed_fn, km, radius):
    """
    Distance to nearest cluster centroid / that cluster's own radius.
    >1 means "farther than this cluster's typical member" -- a direct
    novelty signal. Deliberately NOT gated by purity here (unlike the
    accept/reject decision) -- we want this score high specifically for
    genuinely novel candidates regardless of whether their nearest cluster
    happens to be pure or not; purity answers "can we trust a label", this
    answers "is this worth a human's attention".

    `embed_fn` is cicids2018_spider_owl_2.py's own _embed_in_batches, passed
    in rather than imported so this module stays standalone/decoupled from
    the live experiment file until wiring time. `km`/`radius` are
    cluster_memory_for_member_inference's existing outputs -- no new
    clustering pass needed, this is pure reuse.
    """
    cand_emb = embed_fn(candidates, model)
    cand_dist, assigned_cluster = km.index.search(cand_emb, 1)
    assigned_cluster = assigned_cluster.ravel()
    cand_dist = cand_dist.ravel()
    cluster_radius_per_sample = radius[assigned_cluster]
    return cand_dist / np.clip(cluster_radius_per_sample, 1e-6, None)


# ---------------------------------------------------------------------------
# 2b. [CDIST-WINS BRANCH] True kNN-OOD score -- exact per-point k nearest
#     neighbor distance, no cluster-centroid compression. Reuses the same
#     GPU-batched cosine-distance pattern already validated in
#     cicids2018_spider_owl_2_oldmembercompare.py's
#     _gpu_cosine_member_inference_select.
# ---------------------------------------------------------------------------
def compute_novelty_score_cdist(candidates, memory_X_t, device, k=10, batch_size=20000):
    """
    Mean cosine distance to the k nearest individual memory points, for each
    candidate. This is Sun et al.'s kNN-OOD in its exact original form (no
    compression) -- available specifically because cdist keeps every memory
    point individually resolved, unlike clustering's compressed centroids.
    Higher score = more novel/out-of-distribution relative to memory as it
    currently stands.

    `memory_X_t` must already be L2-normalized (same convention as
    _gpu_cosine_member_inference_select / _gpu_cosine_threshold_calibration
    in the ablation file) so that `1 - cand_n @ memory_X_t.T` is cosine
    distance.
    """
    n = candidates.shape[0]
    scores = np.empty(n, dtype=np.float32)
    k_eff = min(k, memory_X_t.shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cand_t = torch.as_tensor(candidates[start:end], dtype=torch.float32, device=device)
        cand_n = torch.nn.functional.normalize(cand_t, p=2, dim=1, eps=1e-12)
        dist = 1.0 - cand_n @ memory_X_t.T
        kth_smallest, _ = torch.topk(dist, k=k_eff, dim=1, largest=False)
        scores[start:end] = kth_smallest.mean(dim=1).detach().cpu().numpy()
        del cand_t, cand_n, dist, kth_smallest
    return scores


# ---------------------------------------------------------------------------
# 3. BALD-style ensemble disagreement -- reuses the existing multi-learner
#    setup (mlps>1), no new model infrastructure.
# ---------------------------------------------------------------------------
def ensemble_disagreement_score(models, X, device, batch_size=5000):
    """
    Predictive-entropy-minus-expected-entropy (mutual information) across
    the model ensemble -- high value means the models disagree with each
    other more than any single model's own uncertainty would suggest, the
    classic BALD signal for "this is worth a human's judgment". Falls back
    to an all-zero score (no preference from this signal) when only one
    model is available (mlps=1), since disagreement is undefined with a
    single learner -- callers should rely on the novelty score alone then.
    """
    if len(models) < 2:
        return np.zeros(X.shape[0], dtype=np.float32)
    n = X.shape[0]
    scores = np.empty(n, dtype=np.float32)
    eps = 1e-12
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
        probs = []
        with torch.no_grad():
            for m in models:
                probs.append(torch.softmax(m(batch), dim=1))
        probs = torch.stack(probs, dim=0)              # (n_models, b, n_classes)
        mean_probs = probs.mean(dim=0)                  # (b, n_classes)
        predictive_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)
        expected_entropy = -(probs * torch.log(probs + eps)).sum(dim=2).mean(dim=0)
        bald = predictive_entropy - expected_entropy
        scores[start:end] = bald.detach().cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# 4. BADGE-style diverse batch selection.
# ---------------------------------------------------------------------------
def diverse_topk_selection(candidate_X, scores, k, shortlist_multiplier=5, random_state=None):
    """
    Rather than taking the raw top-k by score (which can cluster together --
    many near-duplicate "novel" points wasting the analyst's budget on
    redundant examples), take a generous shortlist by score
    (shortlist_multiplier x k), then use k-means++ seeding over that
    shortlist's raw features to pick k spread-out representatives -- BADGE's
    seeding step. Returns integer indices into `candidate_X` (positions
    relative to whatever candidate slice the caller passed in).
    """
    n = candidate_X.shape[0]
    if k <= 0:
        return np.array([], dtype=np.int64)
    if k >= n:
        return np.arange(n)
    shortlist_size = min(n, k * shortlist_multiplier)
    shortlist_idx = np.argsort(-scores)[:shortlist_size]
    if shortlist_size <= k:
        return shortlist_idx
    shortlist_X = candidate_X[shortlist_idx]
    from sklearn.cluster import kmeans_plusplus
    _, seed_indices = kmeans_plusplus(shortlist_X.astype(np.float64), n_clusters=k,
                                       random_state=random_state)
    return shortlist_idx[seed_indices]


# ---------------------------------------------------------------------------
# 5. Top-level orchestration: combine novelty + disagreement, then diversify.
# ---------------------------------------------------------------------------
def build_analyst_queue(candidate_X, novelty_scores, disagreement_scores, budget,
                         novelty_weight=0.7, disagreement_weight=0.3, random_state=None):
    """
    Combines novelty + disagreement into one acquisition score (min-max
    normalized first so their arbitrary original scales don't dominate one
    another), then applies BADGE-style diverse selection to fill the
    analyst's budget. Weights default to favoring novelty (0.7) since it's
    the better-validated signal for this domain (attack traffic is
    disproportionately OOD relative to memory); disagreement is a secondary
    signal, most useful when mlps>1 gives it something real to say -- when
    it's all-zero (mlps=1), the combination degrades gracefully to pure
    novelty-ranked selection.
    """
    def _norm(s):
        s = np.asarray(s, dtype=np.float64)
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    combined = novelty_weight * _norm(novelty_scores) + disagreement_weight * _norm(disagreement_scores)
    return diverse_topk_selection(candidate_X, combined, budget, random_state=random_state)


# ---------------------------------------------------------------------------
# Self-test: synthetic correctness check, run directly with
# `python active_candidate_selection.py`. Exercises every function above
# except compute_novelty_score_clustering (needs a live faiss km object from
# the actual experiment file -- checked at wiring time instead, against real
# data, the same way the GPU cdist rewrite was verified earlier this session).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=== class_adaptive_mode_value ===")
    # balanced class: threshold should stay at base
    t_balanced = class_adaptive_mode_value(98.0, class_pos_count=5000, class_total_count=5000)
    # severely scarce class: threshold should relax toward the floor
    t_scarce = class_adaptive_mode_value(98.0, class_pos_count=15, class_total_count=13334)
    print(f"balanced (5000/5000): {t_balanced:.2f} (expect ~98.0)")
    print(f"scarce (15/13334):    {t_scarce:.2f} (expect meaningfully below 98, >= 60 floor)")
    assert abs(t_balanced - 98.0) < 0.5
    assert 60.0 <= t_scarce < 90.0

    print("\n=== compute_novelty_score_cdist ===")
    n_mem, d = 2000, 20
    memory_X = np.random.randn(n_mem, d).astype(np.float32)
    memory_X_t = torch.nn.functional.normalize(
        torch.as_tensor(memory_X, dtype=torch.float32, device=device), p=2, dim=1, eps=1e-12)
    # one candidate identical to a memory point (should score ~0 novelty),
    # one candidate far away in a novel direction (should score high novelty)
    near_cand = memory_X[0:1] + np.random.randn(1, d).astype(np.float32) * 0.01
    far_cand = (np.random.randn(1, d) * 5 + 50).astype(np.float32)
    candidates = np.vstack([near_cand, far_cand])
    scores = compute_novelty_score_cdist(candidates, memory_X_t, device, k=10)
    print(f"near-duplicate candidate score: {scores[0]:.4f}")
    print(f"far/novel candidate score:      {scores[1]:.4f}")
    assert scores[1] > scores[0], "novel candidate should score higher than near-duplicate"

    print("\n=== ensemble_disagreement_score ===")
    class TinyNet(torch.nn.Module):
        def __init__(self, class_bias):
            # class_bias is a length-2 vector -- softmax is shift-invariant
            # to a UNIFORM bias across classes, so disagreement has to come
            # from a differential (per-class) bias, not a scalar one.
            super().__init__()
            self.lin = torch.nn.Linear(d, 2)
            with torch.no_grad():
                self.lin.weight.zero_()  # isolate the effect to bias alone
                self.lin.bias.copy_(torch.tensor(class_bias, dtype=torch.float32))
        def forward(self, x):
            return self.lin(x)
    base = TinyNet([0.0, 0.0]).to(device)
    models_agree = [base, base]  # literally the same model twice -> zero disagreement
    models_disagree = [TinyNet([5.0, -5.0]).to(device), TinyNet([-5.0, 5.0]).to(device)]
    X_test = np.random.randn(50, d).astype(np.float32)
    s_single = ensemble_disagreement_score([TinyNet([0.0, 0.0]).to(device)], X_test, device)
    s_agree = ensemble_disagreement_score(models_agree, X_test, device)
    s_disagree = ensemble_disagreement_score(models_disagree, X_test, device)
    print(f"single-model score (expect all zero): max={s_single.max():.6f}")
    print(f"low-disagreement ensemble mean score:  {s_agree.mean():.6f}")
    print(f"high-disagreement ensemble mean score: {s_disagree.mean():.6f}")
    assert s_single.max() == 0.0
    assert s_disagree.mean() > s_agree.mean()

    print("\n=== diverse_topk_selection / build_analyst_queue ===")
    n_cand, k = 200, 10
    cand_X = np.random.randn(n_cand, d).astype(np.float32)
    novelty = np.random.rand(n_cand)
    disagreement = np.random.rand(n_cand)
    idx = build_analyst_queue(cand_X, novelty, disagreement, budget=k, random_state=0)
    print(f"requested budget={k}, got {len(idx)} unique indices: {len(set(idx.tolist())) == len(idx)}")
    assert len(idx) == k
    assert len(set(idx.tolist())) == k

    print("\nALL CHECKS PASSED")
