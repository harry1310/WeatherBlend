using System.Text.Json;

namespace WeatherBlend.Train;

/// <summary>
/// LEAD_POLICY.json — the per-lead-band model policy for the precip phases
/// (3c / 3o), fitted quarterly by <c>precip-fit-lead-policy</c> and consumed
/// by <see cref="Commands.PrecipPredictCommand"/> (docs/PRECIP_LEAD_POLICY_PLAN.md).
///
/// Lives at <c>data/models/precipitation/LEAD_POLICY.json</c>, next to the
/// precipitation MANIFEST. The file records DEVIATIONS from the production
/// bucket policy only: an absent band (or absent file, or unparsable file)
/// means "use the bucket model" — so the default/empty policy is exactly
/// today's behaviour and predict can never break on a policy problem.
///
/// Bands are 6-hourly over actual lead τ (never per-hour — live input is
/// cycle-selected at ~6h NWP cadence, so per-hour policy is false precision).
/// Entries are singles or EQUAL-WEIGHT pairs only: fitted blend weights
/// overfit at this data scale (wind opt-weight lost to 50/50; LGB-meta
/// overfit) — the producer never fits weights, and this artifact cannot
/// express them.
/// </summary>
public sealed class PrecipLeadPolicy
{
    public const string FileName = "LEAD_POLICY.json";

    public DateTime FittedAtUtc { get; set; }
    public string Location { get; set; } = "";

    /// <summary>Live-OOS scoring window the policy was fitted on.</summary>
    public DateTime WindowStartUtc { get; set; }
    public DateTime WindowEndUtc { get; set; }

    /// <summary>Study bundles trained on ValidTime ≤ this (walk-forward —
    /// keeps the scoring window true OOS).</summary>
    public DateTime StudyCutoffUtc { get; set; }

    /// <summary>SELECT &lt; split ≤ SCORE date split: choices are made on the
    /// SELECT slice and graded on the held-out SCORE slice, so margin gates
    /// never read a candidate's own selection data.</summary>
    public DateTime SelectScoreSplitUtc { get; set; }

    public ThresholdsBlock Thresholds { get; set; } = new();

    /// <summary>phase id ("3c"/"3o") → band deviations, ordered by LeadLo.
    /// Bands not listed use the production bucket model.</summary>
    public Dictionary<string, List<BandEntry>> Phases { get; set; } = new();

    public sealed class ThresholdsBlock
    {
        /// <summary>Deviate from the bucket model only when the SCORE-slice
        /// Brier gain is at least this (% of baseline).</summary>
        public double MarginPct { get; set; } = 0.75;

        /// <summary>An incumbent band entry only flips when the challenger
        /// beats it on SCORE by at least this (% of incumbent).</summary>
        public double HysteresisPct { get; set; } = 0.5;

        /// <summary>Length of the held-out SCORE window (settled truth).</summary>
        public int HoldoutDays { get; set; } = 21;
    }

    public sealed class BandEntry
    {
        /// <summary>Actual-lead band [LeadLo, LeadHi) in hours.</summary>
        public int LeadLo { get; set; }
        public int LeadHi { get; set; }

        /// <summary>"single" | "blend" (blend = equal-weight pair, always).</summary>
        public string Kind { get; set; } = "single";

        /// <summary>Nominal training lead(s) of the bundle model(s) to run —
        /// one entry for single, two for blend. These key into the bundle's
        /// per-lead model files (lead_24h.zip etc.), NOT into manifest versions.</summary>
        public List<int> Leads { get; set; } = new();

        /// <summary>Provenance: SCORE-slice Brier of the production bucket
        /// model / this entry, and the relative gain, at fit time.</summary>
        public double BaselineBrier { get; set; }
        public double PolicyBrier { get; set; }
        public double DeltaPct { get; set; }
        public int ScoreN { get; set; }
    }

    public static string PathFor(string modelsRoot)
        => Path.Combine(modelsRoot, "precipitation", FileName);

    /// <summary>Load, or null when the file is absent or unreadable — the
    /// caller treats null as "production bucket policy" (predict must never
    /// break on a policy problem; the producer treats null as "no incumbent").</summary>
    public static PrecipLeadPolicy? TryLoad(string modelsRoot)
    {
        var path = PathFor(modelsRoot);
        try
        {
            if (!File.Exists(path)) return null;
            return JsonSerializer.Deserialize<PrecipLeadPolicy>(File.ReadAllText(path));
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Atomic write (temp + rename) so a crash mid-write can't leave
    /// a half policy for predict to trip on.</summary>
    public void Save(string modelsRoot)
    {
        var path = PathFor(modelsRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>The deviation entry covering actual lead τ for this phase, or
    /// null when the band has no deviation (→ use the bucket model).</summary>
    public BandEntry? Lookup(string phase, double tauHours)
    {
        if (!Phases.TryGetValue(phase, out var bands)) return null;
        foreach (var b in bands)
            if (tauHours >= b.LeadLo && tauHours < b.LeadHi)
                return b;
        return null;
    }

    /// <summary>Production bucket model for actual lead τ — the fallback
    /// policy everywhere, and the baseline the producer gates against.
    /// Mirrors the predict-time lead buckets {24,48,72,96,120} (τ &lt; 48 →
    /// m24 … τ ≥ 120 → m120; τ below 24, e.g. the lead-12 targets, also
    /// lands on m24 — the short-lead cell of the plan).</summary>
    public static int BucketModelFor(double tauHours)
        => tauHours >= 120 ? 120 : tauHours >= 96 ? 96 : tauHours >= 72 ? 72 : tauHours >= 48 ? 48 : 24;
}
