using Microsoft.Extensions.Logging;
using WeatherBlend.Train;

namespace WeatherBlend.Train.Common;

/// <summary>
/// Pre-train sanity gate (Phase 1b of AUTO_RETRAIN_PLAN.md). Compares the
/// current run's <see cref="TrainingSummary"/> against the previous run's
/// and reports any tolerance-band breaches. Trainer is expected to abort
/// the bundle write on a fail — the new model never lands on R2 if
/// upstream data has shifted outside the bands.
///
/// First-ever-train path (no previous summary on disk) MUST skip the
/// check and just write its own summary as the new baseline; the guard
/// has no opinion when there's nothing to compare against. Caller
/// signals this by passing <c>previous = null</c> — <see cref="Check"/>
/// returns a passing result with an explanatory note.
///
/// Tolerance bands per the plan:
/// - <see cref="GuardTolerances.RowsDeltaPct"/> = 0.30
/// - <see cref="GuardTolerances.NanPctAbsolute"/> = 0.20
/// - <see cref="GuardTolerances.LabelRateDeltaAbsolute"/> = 0.10
/// - <see cref="GuardTolerances.FeaturesEffectiveDelta"/> = 0
/// </summary>
public static class RetrainGuard
{
    /// <summary>
    /// Default tolerance bands per AUTO_RETRAIN_PLAN.md "locked-in
    /// decisions". Per-(target, phase) overrides come later via
    /// data/models/retrain_tolerances.json once we have evidence the
    /// defaults are too tight or too loose for any specific cell.
    /// </summary>
    public static readonly GuardTolerances Defaults = new(
        RowsDeltaPct: 0.30,
        NanPctAbsolute: 0.20,
        LabelRateDeltaAbsolute: 0.10,
        FeaturesEffectiveDelta: 0);

    /// <summary>
    /// Check the current summary against the previous one with the given
    /// tolerances. Returns a <see cref="GuardResult"/> with Passed=true
    /// and an empty breach list when all checks pass.
    /// </summary>
    /// <param name="current">Summary being written by the in-progress fit.</param>
    /// <param name="previous">Summary from the most-recent prior fit. Pass null
    /// for the first-ever train of a phase — guard returns Passed=true with
    /// a "no baseline" note.</param>
    /// <param name="tolerances">Tolerance bands. Pass <see cref="Defaults"/>
    /// unless per-cell overrides apply.</param>
    public static GuardResult Check(
        TrainingSummary current,
        TrainingSummary? previous,
        GuardTolerances tolerances)
    {
        if (previous is null)
        {
            return new GuardResult(
                Passed: true,
                Breaches: Array.Empty<GuardBreach>(),
                Note: "First-ever train (no previous summary on disk) — guard skipped, current summary becomes the baseline.");
        }

        var breaches = new List<GuardBreach>();

        // ---- LocationName: must not have changed between runs ----------
        // The catastrophic failure mode the Phase A safety work guards
        // against: a trainer invocation accidentally writing a Membury
        // bundle into Bonehill's manifest slot (or vice versa) would
        // promote it as the new champion and silently start serving
        // wrong-location predictions until verify flags drift days later.
        // Both empty → legacy bundle pair, no info to compare; skip.
        // Only one side present → backfill landing or first pinned write
        // after the legacy era; pass with note.
        // Both present and different → hard breach.
        if (!string.IsNullOrEmpty(current.LocationName)
            && !string.IsNullOrEmpty(previous.LocationName)
            && !string.Equals(current.LocationName, previous.LocationName, StringComparison.OrdinalIgnoreCase))
        {
            breaches.Add(new GuardBreach(
                Field: "locationName",
                Previous: previous.LocationName,
                Current: current.LocationName,
                Threshold: "must not change",
                Reason: $"LocationName changed from '{previous.LocationName}' to '{current.LocationName}' for the same composite — refusing to overwrite. Either you've accidentally pointed this trainer at the wrong location's NWP, or the manifest entry is in the wrong location's slot. Investigate before forcing through."));
        }

        // ---- Row count: relative ±RowsDeltaPct -------------------------
        // Empty previous slices (0 rows) skip the check — would divide by
        // zero and never converge to a useful tolerance anyway.
        CheckRelative("rowsTrain", previous.RowsTrain, current.RowsTrain,
            tolerances.RowsDeltaPct, breaches);
        CheckRelative("rowsVal", previous.RowsVal, current.RowsVal,
            tolerances.RowsDeltaPct, breaches);
        CheckRelative("rowsTest", previous.RowsTest, current.RowsTest,
            tolerances.RowsDeltaPct, breaches);

        // ---- Features-effective: absolute delta (0 = any change abort)
        var featDelta = current.FeaturesEffective - previous.FeaturesEffective;
        if (Math.Abs(featDelta) > tolerances.FeaturesEffectiveDelta)
        {
            breaches.Add(new GuardBreach(
                Field: "featuresEffective",
                Previous: previous.FeaturesEffective,
                Current: current.FeaturesEffective,
                Threshold: tolerances.FeaturesEffectiveDelta,
                Reason: $"feature count changed by {featDelta:+0;-0;0} (tolerance ±{tolerances.FeaturesEffectiveDelta}) — a column appearing or disappearing always signals an upstream schema shift; abort and inspect"));
        }

        // ---- Per-feature NaN%: absolute delta in NaN fraction ---------
        // Shared keys only — features that exist in only one summary are
        // already caught by the FeaturesEffective check above. Avoids
        // double-flagging.
        foreach (var key in current.PerFeature.Keys.Intersect(previous.PerFeature.Keys))
        {
            var prevNan = previous.PerFeature[key].NanPct;
            var currNan = current.PerFeature[key].NanPct;
            var delta = Math.Abs(currNan - prevNan);
            if (delta > tolerances.NanPctAbsolute)
            {
                breaches.Add(new GuardBreach(
                    Field: $"perFeature.{key}.nanPct",
                    Previous: prevNan,
                    Current: currNan,
                    Threshold: tolerances.NanPctAbsolute,
                    Reason: $"NaN fraction changed by {delta:0.00} (tolerance {tolerances.NanPctAbsolute:0.00}) — feature column reliability shifted, often signals an upstream model dropping out or a feature builder change"));
            }
        }

        // ---- Per-station label rate: absolute delta -------------------
        foreach (var key in current.LabelRates.Keys.Intersect(previous.LabelRates.Keys))
        {
            var prevRate = previous.LabelRates[key];
            var currRate = current.LabelRates[key];
            var delta = Math.Abs(currRate - prevRate);
            if (delta > tolerances.LabelRateDeltaAbsolute)
            {
                breaches.Add(new GuardBreach(
                    Field: $"labelRates.{key}",
                    Previous: prevRate,
                    Current: currRate,
                    Threshold: tolerances.LabelRateDeltaAbsolute,
                    Reason: $"label rate changed by {delta:0.00} (tolerance {tolerances.LabelRateDeltaAbsolute:0.00}) — base rate shift this large at a single retrain interval is unusual; suspect truth-source change or upstream gauge issue"));
            }
        }

        return new GuardResult(
            Passed: breaches.Count == 0,
            Breaches: breaches,
            Note: breaches.Count == 0
                ? "All bands within tolerance."
                : $"{breaches.Count} breach(es); abort the bundle write and inspect.");
    }

    private static void CheckRelative(
        string field, int previous, int current, double pctTolerance,
        List<GuardBreach> breaches)
    {
        if (previous <= 0) return;  // skip undefined relative on zero baseline
        var delta = Math.Abs(current - previous) / (double)previous;
        if (delta > pctTolerance)
        {
            breaches.Add(new GuardBreach(
                Field: field,
                Previous: previous,
                Current: current,
                Threshold: pctTolerance,
                Reason: $"row count changed by {delta:P1} (tolerance {pctTolerance:P1}) — large drift this size at one retrain interval signals an upstream collector / refresh issue"));
        }
    }

    /// <summary>
    /// Build a TrainingSummary for the in-progress run, find the previous
    /// summary in the same (composite, phase) lineage, run the guard, and
    /// on pass write the new summary alongside training_metadata.json.
    ///
    /// Returns the <see cref="GuardResult"/> so the caller can decide
    /// whether to abort (typically `if (!result.Passed) return 4;` BEFORE
    /// the Promote* call — that keeps the orphan version dir on disk but
    /// out of the manifest, so predict + verify never see it).
    ///
    /// Note: per-lead model zips are typically already written by the
    /// time this fires (they happen inside the trainer's lead loop). The
    /// guard's "abort the bundle write" guarantee is enforced via the
    /// promotion skip — without a manifest update, the orphan dir is
    /// invisible to the rest of the pipeline. A future refinement could
    /// stage to a tmp dir and rename on guard-pass, but that's a bigger
    /// trainer-flow change.
    ///
    /// <paramref name="trainFeatures"/> may be empty (e.g. a degraded run
    /// that produced no usable rows); the helper returns a "no baseline"
    /// passing result without writing a summary, mirroring the existing
    /// <see cref="TrainingSummaryBuilder.BuildAndSave"/> degraded-mode
    /// behaviour.
    /// </summary>
    public static GuardResult BuildCheckAndSave(
        ILogger log,
        string versionDir,
        string composite, string phase, string version,
        DateTime computedAtUtc,
        int rowsTrain, int rowsVal, int rowsTest,
        IReadOnlyList<float[]>? trainFeatures,
        IReadOnlyList<string> featureNames,
        Dictionary<string, double>? labelRates = null,
        GuardTolerances? tolerances = null,
        // Configured location whose NWP fed the training data — pinned at
        // train time so predict can refuse to score the bundle against any
        // other location's NWP (Phase A multi-location safety, 2026-05-12).
        // Optional argument for the duration of the trainer-side rollout;
        // call sites that don't pass it write a legacy summary the guard
        // treats as "location unknown" (passes with a note). Once every
        // trainer threads it through, the schema gates it as required.
        string? locationName = null)
    {
        // Degraded path: empty features → can't compute stats, can't
        // guard. Don't write a summary; the next run will see "no
        // previous" and pass with the no-baseline note. Same posture as
        // BuildAndSave's pre-existing empty-features short-circuit.
        if (trainFeatures is null || trainFeatures.Count == 0
            || featureNames.Count == 0)
        {
            log.LogWarning(
                "Retrain guard skipped for {Composite}/{Phase} {Version} — buffered train slice was empty (no summary written, no guard check).",
                composite, phase, version);
            return new GuardResult(
                Passed: true,
                Breaches: Array.Empty<GuardBreach>(),
                Note: "Empty train slice — guard skipped, no summary written.");
        }

        var summary = TrainingSummaryBuilder.Build(
            composite, phase, version, computedAtUtc,
            rowsTrain, rowsVal, rowsTest,
            trainFeatures, featureNames, labelRates);
        summary.LocationName = locationName;

        var parentDir = Path.GetDirectoryName(versionDir)
            ?? throw new InvalidOperationException(
                $"versionDir '{versionDir}' has no parent — can't locate previous summary.");

        var previous = TryLoadPreviousSummary(parentDir, phase, excludeVersion: Path.GetFileName(versionDir)!);
        var result = Check(summary, previous, tolerances ?? Defaults);

        if (!result.Passed)
        {
            log.LogError(
                "Retrain guard FAIL for {Composite}/{Phase} {Version}: {Note}",
                composite, phase, version, result.Note);
            foreach (var breach in result.Breaches)
            {
                log.LogError(
                    "  {Field}: previous={Previous}, current={Current}, threshold={Threshold} — {Reason}",
                    breach.Field, breach.Previous, breach.Current, breach.Threshold, breach.Reason);
            }
            // Don't write the summary on fail — leaving the previous
            // training_summary.json in place means the next attempt's
            // guard sees the same baseline rather than chasing a
            // partial/corrupted summary from this run.
            return result;
        }

        ModelArtifact.SaveTrainingSummary(versionDir, summary);
        log.LogInformation(
            "Retrain guard PASS for {Composite}/{Phase} {Version} — train={RowsTrain} val={RowsVal} test={RowsTest} features={Features}{PrevNote}",
            composite, phase, version,
            summary.RowsTrain, summary.RowsVal, summary.RowsTest, summary.FeaturesEffective,
            previous is null ? " (no previous baseline; this becomes the new baseline)" : "");
        return result;
    }

    /// <summary>
    /// Resolve the most recent training_summary.json on disk for a
    /// (target, [station,] phase) lineage, EXCLUDING the in-progress
    /// version dir the trainer is currently writing into. Returns null
    /// when no predecessor exists (first-ever train).
    ///
    /// Lookup is by directory naming convention: under
    /// <paramref name="parentDir"/> (e.g. data/models/temperature/ or
    /// data/models/precipitation/{station}/), version dirs are sorted
    /// lex-descending (timestamp prefix is fixed-width so this is the
    /// chronological order), filtered to those whose
    /// <c>training_metadata.json</c> reports the matching phase, and
    /// the first one that has a <c>training_summary.json</c> on disk
    /// wins. Skips <paramref name="excludeVersion"/> so the trainer can
    /// pass its own newly-created versionDir name.
    /// </summary>
    public static TrainingSummary? TryLoadPreviousSummary(
        string parentDir, string phase, string excludeVersion)
    {
        if (!Directory.Exists(parentDir)) return null;
        var candidates = Directory.GetDirectories(parentDir)
            .Select(Path.GetFileName)
            .Where(n => !string.IsNullOrEmpty(n) && n != excludeVersion)
            .OrderByDescending(n => n, StringComparer.Ordinal)
            .ToList();

        foreach (var versionName in candidates)
        {
            var versionDir = Path.Combine(parentDir, versionName!);
            // Phase-match: the metadata file is the source of truth (the
            // dir-name suffix is convention but not contract). Skip dirs
            // missing metadata — they're partial writes / aborted runs.
            try
            {
                var metaPath = Path.Combine(versionDir, ModelArtifact.TrainingMetadataFileName);
                if (!File.Exists(metaPath)) continue;
                var meta = ModelArtifact.LoadTrainingMetadata(versionDir);
                if (!string.Equals(meta.Phase, phase, StringComparison.Ordinal)) continue;
            }
            catch
            {
                // Treat unreadable metadata as "skip this dir". A guard
                // miss on a stale predecessor is better than crashing
                // mid-retrain on a corrupted JSON.
                continue;
            }
            var summary = ModelArtifact.TryLoadTrainingSummary(versionDir);
            if (summary is not null) return summary;
            // Phase matches but no summary — predates Phase 1a. Keep
            // walking back; if no older version has one, return null
            // (first-ever-with-summary semantics, the guard passes with
            // a "no baseline" note).
        }
        return null;
    }
}

/// <summary>
/// Tolerance band thresholds for <see cref="RetrainGuard.Check"/>. Each
/// field is the maximum allowed delta before the guard fires.
/// </summary>
public sealed record GuardTolerances(
    /// <summary>Maximum relative row-count change (e.g. 0.30 = ±30%).</summary>
    double RowsDeltaPct,
    /// <summary>Maximum absolute change in NaN fraction per feature (0..1).</summary>
    double NanPctAbsolute,
    /// <summary>Maximum absolute change in label rate per station (0..1).</summary>
    double LabelRateDeltaAbsolute,
    /// <summary>Maximum absolute change in featuresEffective. 0 = any
    /// change aborts; bumping above 0 effectively disables the check.</summary>
    int FeaturesEffectiveDelta);

/// <summary>Outcome of a guard check. Caller aborts the bundle write
/// when <see cref="Passed"/> is false.</summary>
public sealed record GuardResult(
    bool Passed,
    IReadOnlyList<GuardBreach> Breaches,
    string Note);

/// <summary>One field's tolerance breach for structured logging.</summary>
public sealed record GuardBreach(
    string Field,
    object Previous,
    object Current,
    object Threshold,
    string Reason);
