# Phase 3r — multiclass LightGBM on bucketed longest-daytime-dry-run

**Status:** plan (drafted 2026-05-14). Awaiting Harry's sign-off before any code lands.

## Hypothesis

3b is three independent binary LightGBM heads — one per window N ∈ {3, 4, 6}. They don't share representation and don't enforce cross-window monotonicity (`P(longest ≥ 3) ≥ P(longest ≥ 4) ≥ P(longest ≥ 6)`); 3g/3j/3n exist partly to engineer that constraint in via MC.

3r reframes the problem at the response variable directly: train **one multiclass LightGBM per (station, lead)** predicting which bucket the day's `longest_dry_run_hours` falls into. P(window ≥ N) is then derived as `sum of buckets ≥ N`. Monotonicity is free; one model replaces three; bake-off question is whether bucketing's information loss costs more than the structural wins.

Untouched corner of the dry-window bake-off — to date everything has been per-hour-chain (RNN/iid-MC/copula-MC/regime-MC/Markov/joint-loss-MLP). 3r is the first model to target the response variable directly.

## Design

### Label

`longest_dry_run_hours` = longest contiguous run of dry hours (≥ 0.1 mm/h threshold matches 3a/3b) inside the daytime window 09:00–18:00 Europe/London, computed per target_date from EA hydrology truth. Integer in [0, 9].

Reuse `DryWindowLabelBuilder` — it already computes this internally; expose the raw longest-run via an internal helper rather than building a parallel labeller.

### Bucket grid (bake-off across these)

| Variant | Buckets | Rationale |
|---------|---------|-----------|
| 3r-4a   | `[0-2], [3], [4-5], [6+]` | 4 classes; one-bucket-per-window-question + a "definitely short" tail |
| 3r-4b   | `[0-2], [3-4), [4-6), [6+]` | 4 classes; matches the exact 3b windows {3, 4, 6} by sum-of-tail |
| 3r-5    | `[0-2], 3, 4, 5, [6+]` | 5 classes; finer at the decision points |
| 3r-6    | `[0-1], 2, 3, 4, 5, [6+]` | 6 classes; finer still |

3r-4b is the natural baseline — sum-of-tail derivation gives `P(≥3) = b1+b2+b3`, `P(≥4) = b2+b3`, `P(≥6) = b3` exactly. Others test whether finer granularity buys anything.

### Model

- `LightGbmMulticlassTrainer` from Microsoft.ML (softmax objective under the hood; ML.NET wraps that as raw class-score per class).
- Same 53 features as 3b lean — keep architecture comparison apples-to-apples. Rich-feature variant (3d-shape's 60-feat) is a separate axis to sweep later.
- Per (station, lead) — one model. Lead set `{24, 48, 72}` matching 3b/3g/3j/3n.
- 9 cells total per bucket-grid variant (3 stations × 3 leads).

### Calibration

LightGBM multiclass raw probabilities are typically under-confident; ML.NET's wrapper exposes them in the `Score` column. Two options:

1. **Per-cumulative-head PAV** (recommended starting point) — derive `P(longest ≥ N)` for N ∈ {3, 4, 6} from the raw PMF, then PAV-calibrate each cumulative head on validation. Same calibration toolchain as 3b. Bonus: this is what we actually score against in Brier, so the calibration target = the scoring target.
2. **Dirichlet calibration** on the PMF itself. More principled, more bespoke. Defer to a 3r v2 if v1 wins the bake-off.

### Architecture sketch

```
WeatherBlend/Train/DryWindow/
├── DryWindow3rPredictor.cs        # train + predict + bucket schema + cumulative derivation
├── DryWindowTrainer.cs             # add TrainVectorMulticlass(...) overload
DryWindowTrainCommand.cs            # add "multiclass" feature-set + RunPhase3rAsync
DryWindowPredictCommand.cs          # dispatch metadata.Phase == "3r" to RunPhase3rAsync
config/phases.yaml                  # register 3r as challenger
Site/DryWindowPhase.cs              # add Phase3r display record
```

Bucket grid is a hyperparameter persisted in `training_metadata.Hyperparameters.bucket_edges` (array of int edges), so each saved bundle is self-describing.

## Bake-off setup

1. Reuse `scripts/DryWindowStartHour/dry_window_4way_bakeoff.py` — extend to discover 3r bundles + a `--bucket-grid` axis, emit Brier-per-window for each bucket variant alongside 3b/3g/3j/3n.
2. Same train/val/test split as the 2026-05-13 15-way bake-off: replay parquet from 3a champion `v2026-04-28_232709`.
3. Score:
   - Aggregate Brier across (station, window, lead) cells
   - Per-window Brier (3h / 4h / 6h separately — historically the per-window split has decided winners)
   - Cross-window monotonicity sanity (% of predictions where `P(≥3) ≥ P(≥4) ≥ P(≥6)`). 3r-4b should be 100% by construction; others should be ≥ 99% with PAV applied independently per head (a small fraction may flip near boundaries — log and inspect).
4. Pick the winning bucket variant; if any 3r variant beats 3g aggregate AND wins ≥ 1 window decisively, promote as challenger.

## Risks

- **Bucketing throws away information.** Going from continuous (well, integer 0–9) to ~4 buckets is a real coarsening. Sparse tail buckets ("all daytime dry" at lead 72h might be < 5% of train days) calibrate poorly. Mitigation: bucket grid sweep + PAV on cumulative outputs.
- **Multiclass treats ordinal as nominal.** Softmax doesn't know bucket 5 is "closer to" 4 than to 0. Manifests as: predictions can be spiky / non-unimodal across buckets. If the bake-off reveals this, the v2 move is ordinal regression (cumulative-link), which Microsoft.ML doesn't expose — would need to drop to raw LightGBM via Python (like 3j/3n) or pivot the architecture.
- **Class imbalance.** Top bucket is rare. LightGBM multiclass options include `UnbalancedSets`-style class weighting; need to confirm what the ML.NET wrapper exposes. If unbalanced bites hard, try class-weighted loss or undersample dominant buckets in train.
- **Single model = single point of failure across windows.** If 3r-4b's training fails or its bundle corrupts, all three windows drop simultaneously. 3b/3g/3j/3n are per-window so a single-cell failure only takes that cell out. Operationally fine — retrain-blenders' RetrainGuard catches this — but worth knowing.

## Acceptance criteria

3r ships as a challenger if BOTH:
- Aggregate Brier ≤ 3g's by at least 1% (so it's not a wash)
- Per-window: wins or ties at least 1 of {3h, 4h, 6h} decisively (≥ 2% improvement) without losing any window by > 5%

Otherwise: write up as negative result alongside 3o/3p/3q in `scripts/DryWindowStartHour/`, strip the code, save to memory.

## Effort estimate

- 3r core code (trainer + predictor + train/predict dispatch + manifest registration): ~1 day
- Bucket-grid bake-off (4 variants × 9 cells × train + score): ~half a day of compute on this box, mostly LightGBM training (fast) + replay scoring
- Bake-off harness extension to discover 3r and slice by bucket-grid: ~2 hours
- Site display metadata (DryWindowPhase, PhaseDescription, color): trivial
- Retrain wiring (retrain-blenders.yml + RetrainGuard tolerances): ~half a day

Total: ~2–3 days of focused work.

## Open decisions for Harry

1. **Bucket grid bake-off vs. single bucket choice?** Recommend bake-off all four — cheap, and the answer to "which bucketing is best" is itself useful.
2. **Calibration: per-cumulative-head PAV in v1, or skip calibration and report raw Brier?** Recommend PAV — same toolchain as 3b, fair comparison.
3. **Lean-feature set (53) or rich (60, with the 7 within-day shape features)?** Recommend lean for v1 to isolate the architecture change; rich is a v2 sweep.
4. **Lead 120h not in 3b's scope** — keep 3r at {24, 48, 72} for direct comparison, or extend? Recommend match-3b.
5. **If it wins, does it replace 3b or just sit alongside?** Recommend alongside — same convention as 3g/3j/3n. Per-window champion routing if 3r dominates a specific window.
