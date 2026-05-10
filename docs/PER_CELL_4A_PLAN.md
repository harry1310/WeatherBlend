# Per-cell 4a deployment plan (SHIPPED 2026-05-11)

> **Status update 2026-05-11:** shipped. User decision late in the same
> session as the plan was drafted: "make the per cell 4a the one and
> only 4a and then kick off a retrain on it" — so per-cell REPLACED
> lead-pooled 4a outright rather than coexisting as a separate
> `4a_percell` challenger as originally planned. Phase tag stays
> plain `4a`. The plan text below is preserved as historical context;
> the bits that diverged from execution are flagged inline.

## Why

Aligned-row bake-off on 2026-05-10 showed deployed (lead-pooled) 4a is
~tied with or slightly behind per-cell 3a aggregated, while a per-cell
BART (the research architecture that produced the original -5.7% memory)
beats 3a by -4.0% aggregate. The architectural switch from
lead-as-feature pooling to per-(station, lead) BART specialisation is
the load-bearing change.

Two corroborating findings from the same session:

- **3a + lead-pooled 4a linear pool**: -2.7% Brier vs 4a alone. Real
  blend gain from architectural diversity (LightGBM vs BART).
- **lead-pooled 4a + per-cell BART linear pool**: 0.0% Brier vs
  per-cell alone. **w\* aggregate = 0.09**, exactly zero on 5 of 9
  cells. Same model class + same hyperparameters → no diversity to
  exploit, blending is decoration. Lead-pooled flavour is strictly
  redundant once per-cell exists.

Conclusion: ship per-cell 4a as a challenger, drop the lead-pooled
flavour entirely once per-cell is verified live.

## Scope decisions (locked in by user 2026-05-10)

- **Role: challenger, NOT champion.** 3a stays Current at all leads.
  Per-cell 4a renders as an alternative card on the Models page +
  appears in the predictions tree under its own `model_version`.
- **All 5 leads** (24/48/72/96/120). The 9-cell research only covered
  24/48/72; we extend the per-cell training loop to 96/120 inline.
  Per-lead Brier already rises with lead in the lead-pooled metadata,
  so specialisation may help less at long leads — but as a challenger
  the downside is minimal.
- **No Bovey override.** Per-cell BART loses to 3a on Bovey by
  0.8-4.6%, but since per-cell 4a is a challenger (not promoted),
  3a stays the Current pick for Bovey automatically. Revisit only if
  we ever consider promoting per-cell 4a to champion.
- **Defer to after 5a training stability is resolved.** The 5a CI
  parallelism issue (~4h wall on what should be a ~1h job) needs a
  fix first.

## Implementation outline

### 1. `train_4a.py` — refactor to per-cell loop (~3-4h)

Today:
```python
for station_input in stations:
    result = train_one_station(station_friendly)   # 1 BART, all leads pooled
    write_bundle(bundle_dir, station_slug, ...)
```

New:
```python
for station_input in stations:
    for lead in LEADS:                              # NEW: nested per-lead
        result = train_one_cell(station_friendly, lead)   # 1 BART, single lead
        write_lead_bundle(bundle_dir, station_slug, lead, ...)
```

- `train_one_cell` mirrors `train_one_station` but builds features for
  ONE lead via `build_features_via_duckdb(station, lead)` (already
  exists, used by `run_phase6_bart_9cell.py`).
- Bundle layout: flat files `state_lead_24h.rds`, `state_lead_48h.rds`,
  ... inside one version dir per station. Mirrors 3a's `lead_NNh.zip`
  pattern. Single `arrays_lead_NNh.npz` per lead. Shared
  `preprocess.json` if all leads use the same feature list (they do —
  22-feat 3a base + 3 synoptic + no `lead` feature anymore since it's
  fixed per cell).
- `test_predictions.parquet`: keep one consolidated parquet per station
  (already per-cell-rowed by the slice 1 commit). Now naturally
  covers all leads since each is fit separately.
- RetrainGuard: stays per-station. Aggregate row counts across leads;
  the guard's bands still fire on per-feature NaN% / label-rate.

### 2. `predict_4a.py` — refactor to per-cell load + dispatch (~2-3h)

Today: load one state.rds per station, warm-scaffold, predict on rows
for all leads with `lead` as a feature column.

New: load N state.rds per station (one per lead). For each output row,
dispatch to the matching lead's state. Same warm-scaffold pattern per
state, just N of them per station.

- Cache loaded states per (station, lead) for the duration of one
  predict invocation — avoid re-loading per row.
- Shared preprocess.json applies the same scaler to all leads.
- Predictions tree partition: stays
  `data/predictions/precipitation/{station}/model_version=v..._phase4a_percell/`
  (rename suffix from `_phase4a` to `_phase4a_percell` so deployed
  lead-pooled 4a and new per-cell 4a coexist in the predictions tree
  during transition; site / verify filter on suffix).

### 3. Bundle layout (~30 min)

Flat-file convention:

```
data/models/precipitation/{station}/v..._phase4a_percell/
  state_lead_24h.rds
  state_lead_48h.rds
  state_lead_72h.rds
  state_lead_96h.rds
  state_lead_120h.rds
  arrays_lead_24h.npz
  arrays_lead_48h.npz
  ... (one per lead)
  preprocess.json                    (shared: 22+3 feats, scaler params)
  training_metadata.json             (PerLead populated for 5 leads)
  training_summary.json              (slice-1 sidecar, aggregated across leads)
  test_predictions.parquet           (all 5 leads, ~15k rows)
  feature_schema.json                (per-lead spec, Models card)
```

Phase tag: `4a_percell` so the suffix in the version-dir name is
unambiguous from deployed lead-pooled `4a`.

### 4. `phases.yaml` — add `4a_percell` as a second precipitation challenger (~5 min)

```yaml
precipitation:
  phases:
    - id: "3a"        # champion
      role: champion
      impl: dotnet
    - id: "3c"        # challenger
      role: challenger
      impl: dotnet
    - id: "3d"        # challenger
      role: challenger
      impl: dotnet
    - id: "4a"        # existing lead-pooled (deprecated soon)
      role: challenger
      impl: python
    - id: "4a_percell"  # NEW per-cell challenger
      role: challenger
      impl: python
    - id: "5a"        # confidence overlay
      role: confidence
      impl: python
```

Once per-cell is verified live, the deprecated lead-pooled `4a` entry
can come out in a follow-up. Until then both coexist — gives us a
direct on-site comparison via the Models cards.

### 5. Smoke test (~1h)

- Train one station per-cell end-to-end via local `python train_4a.py
  --stations ea_bellever_dartmoor --percell` (new flag).
- Inspect bundle dir contents match the layout above.
- Run linear_pool_3a_4apc.py against the new bundle's
  test_predictions.parquet — should reproduce ~-4% vs 3a aggregate
  (with all 5 leads now, vs 9-cell research scope of 3).
- Confirm predict_4a.py loads the new layout + emits predictions
  matching the test parquet within rpy2 nondeterminism.

### 6. Sunday auto-retrain integration

No workflow file changes needed — `retrain-python.yml` calls
`python -u scripts/train_4a.py` with no flags. To opt into per-cell,
either:

(a) Make `--percell` the default in `train_4a.py` after the
    transition (clean cut).
(b) Add a `train_4a_per_cell` boolean workflow_dispatch input that
    threads `--percell` to the script.

(a) is the recommended approach once we're confident — Sunday cron
just retrains the right thing without per-run config.

### 7. Site rendering

ActivePhasePolicy already iterates 4a as a precipitation challenger
phase. The `4a_percell` entry will get a Models card automatically
once phases.yaml is updated + the predictions tree has at least one
prediction. No SitePages.cs changes needed.

Verify pipeline reads `training_metadata.Phase` and filters via
`IsActivePhase` — also automatic.

### 8. Documentation + memory (~30 min)

- Update `CLAUDE.md` Auto-retrain section: mention per-cell 4a as the
  precipitation challenger architecture.
- Memory entry: `project_phase4a_percell_shipped_<date>.md` once
  deployed, with bake-off numbers.
- Update `project_phase4a_shipped.md`'s memory to note the lead-pooled
  version was superseded by per-cell on whatever date.

## Total effort

**~6-7h** of focused work for the deployment + smoke test. Sunday
auto-retrain catches the change automatically on the next noon UTC
tick.

## Pre-deployment checklist

- [ ] 5a CI training stability resolved (parallelism issue investigated
      and fixed — not blocking per-cell 4a but no point shipping new
      things while the auto-retrain Sunday cron is unstable).
- [ ] One final aligned-row bake-off after the most recent Sunday
      retrain, to confirm the -4.0% gap hasn't shifted (sanity check).

## What this DOESN'T touch

- Champion-by-lead pinning (per-cell 4a is a challenger, not a
  champion at any lead).
- Bovey-specific overrides (skipped per user instruction; only
  matters if per-cell 4a were promoted).
- The `ChampionByLead` infrastructure (used by 2d temp + 3d precip,
  irrelevant for a challenger).
- 5a's lead-as-feature posterior (per-cell would destroy the
  hierarchical pooling story; explicitly OUT of scope per earlier
  discussion).
- Stacking / non-linear blends (theory work confirmed they can't
  manufacture diversity; not on the roadmap).
