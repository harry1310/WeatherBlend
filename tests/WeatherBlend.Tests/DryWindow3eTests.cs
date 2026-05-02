using FluentAssertions;
using WeatherBlend.Models;
using WeatherBlend.Site;
using WeatherBlend.Train.DryWindow;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Pins the load-bearing pieces of Phase 3e (B2 conditional decomposition
/// for the 3h + 4h dry-window cascade). The full pipeline (DuckDB queries,
/// LightGBM training, parquet writes) is integration-tested through the
/// CLI; these unit tests cover the arithmetic + projection helpers that
/// would silently break the cascade if regressed.
/// </summary>
public class DryWindow3eTests
{
    // ---------- Cascade arithmetic ----------

    [Fact]
    public void MultiplyForCascade_returns_product_of_factors()
    {
        var pBase   = new[] { 0.8, 0.5, 0.2, 0.0, 1.0 };
        var pExtend = new[] { 0.6, 0.5, 0.9, 0.7, 0.5 };
        var product = DryWindow3eCascadeArtefact.MultiplyForCascade(pBase, pExtend);

        product.Should().HaveCount(5);
        product[0].Should().BeApproximately(0.48, 1e-9);   // 0.8 × 0.6
        product[1].Should().BeApproximately(0.25, 1e-9);   // 0.5 × 0.5
        product[2].Should().BeApproximately(0.18, 1e-9);   // 0.2 × 0.9
        product[3].Should().Be(0.0);                       // 0.0 × anything = 0
        product[4].Should().BeApproximately(0.50, 1e-9);   // 1.0 × 0.5
    }

    [Fact]
    public void MultiplyForCascade_enforces_monotonicity_P4h_le_P3h()
    {
        // The whole point of the cascade: P(4h) ≤ P(3h) is structural,
        // not an empirical hope. For any base × extend pair the product
        // must not exceed pBase.
        var rng = new Random(42);
        var pBase   = Enumerable.Range(0, 200).Select(_ => rng.NextDouble()).ToArray();
        var pExtend = Enumerable.Range(0, 200).Select(_ => rng.NextDouble()).ToArray();

        var product = DryWindow3eCascadeArtefact.MultiplyForCascade(pBase, pExtend);

        for (int i = 0; i < pBase.Length; i++)
            product[i].Should().BeLessThanOrEqualTo(pBase[i] + 1e-12,
                $"row {i}: product {product[i]:F4} must not exceed base {pBase[i]:F4}");
    }

    [Fact]
    public void MultiplyForCascade_clamps_out_of_range_inputs_to_zero_one()
    {
        // Defensive: if either factor came back out of [0,1] from a poorly
        // calibrated upstream, clamp before multiplying. Avoids a negative
        // product or one > 1 surfacing on the home card.
        var product = DryWindow3eCascadeArtefact.MultiplyForCascade(
            pBase:   new[] { -0.1, 1.2,  0.5 },
            pExtend: new[] {  0.5, 0.5, -0.5 });

        product[0].Should().Be(0.0);    // clamp(-0.1) × 0.5 = 0
        product[1].Should().Be(0.5);    // clamp(1.2) × 0.5 = 1 × 0.5
        product[2].Should().Be(0.0);    // 0.5 × clamp(-0.5) = 0.5 × 0
    }

    [Fact]
    public void MultiplyForCascade_throws_on_mismatched_lengths()
    {
        var act = () => DryWindow3eCascadeArtefact.MultiplyForCascade(
            pBase: new[] { 0.5, 0.5 }, pExtend: new[] { 0.5 });
        act.Should().Throw<ArgumentException>()
            .WithMessage("*pBase*pExtend*same length*");
    }

    [Fact]
    public void MultiplyForCascade_returns_empty_for_empty_inputs()
    {
        var product = DryWindow3eCascadeArtefact.MultiplyForCascade(
            pBase: Array.Empty<double>(), pExtend: Array.Empty<double>());
        product.Should().BeEmpty();
    }

    // ---------- Multi-label row projection ----------

    [Fact]
    public void ToCommonRow_preserves_features_and_uses_chosen_label()
    {
        var multi = new DryWindow3eMultiLabelRow
        {
            TargetDateUtc = new DateTime(2026, 5, 4, 0, 0, 0, DateTimeKind.Utc),
            Features      = new[] { 1.0f, 2.0f, 3.0f },
            Label3h       = true,
            Label4h       = false,
            PrecipMmDay   = 0.5f,
        };

        var row3h = DryWindow3eFeatureBuilder.ToCommonRow(multi, multi.Label3h, outputWindowHours: 3);
        row3h.TargetDateUtc.Should().Be(multi.TargetDateUtc);
        row3h.Features.Should().BeSameAs(multi.Features);    // shared reference, no copy
        row3h.Label.Should().BeTrue();
        row3h.WindowHours.Should().Be(3);
        row3h.PrecipMmDay.Should().Be(0.5f);

        var row4h = DryWindow3eFeatureBuilder.ToCommonRow(multi, multi.Label4h, outputWindowHours: 4);
        row4h.Label.Should().BeFalse();
        row4h.WindowHours.Should().Be(4);
    }

    [Fact]
    public void Phase3e_constants_match_the_strings_disk_artefacts_persist()
    {
        // Pin the exact string values — these end up in
        // training_metadata.Phase, manifest version-dir suffixes, and the
        // ActivePhasePolicy allowlist. Renaming silently breaks the
        // predict path (the metadata.Phase != Phase3e check there branches
        // to the wrong code).
        DryWindow3eFeatureBuilder.Phase3e.Should().Be("3e");
        DryWindow3eCascadeArtefact.VersionSuffix.Should().Be("phase3e");
        DryWindow3eCascadeArtefact.ExtendModelFileName(24).Should().Be("lead_24h_extend.zip");
        DryWindow3eCascadeArtefact.ExtendModelFileName(72).Should().Be("lead_72h_extend.zip");
        DryWindow3eFeatureBuilder.OutputWindows.Should().Equal(new[] { 3, 4 });
    }

    // ---------- Active phase policy + display registration ----------

    [Fact]
    public void ActivePhasePolicy_dry_window_includes_3e_after_3b()
    {
        // Order matters: champion-first ordering drives the per-card sort
        // on the Models page. 3b stays the champion; 3e is appended as a
        // challenger so a glance at the page reads "lean champion → cascade
        // challenger".
        var ordered = ActivePhasePolicy.ByTarget["dry_window"];
        ordered.Should().Equal(new[] { "3b", "3e" });
    }

    [Theory]
    [InlineData("3b",       true)]
    [InlineData("3e",       true)]
    [InlineData("3d-shape", false)]    // dropped 2026-04-29
    [InlineData("3a",       false)]    // wrong target
    [InlineData("",         false)]
    public void ActivePhasePolicy_IsActive_dry_window_recognises_3e(string phase, bool expected)
    {
        ActivePhasePolicy.IsActive("dry_window", phase).Should().Be(expected);
    }

    [Fact]
    public void DryWindowPhases_All_now_includes_3e()
    {
        // DryWindowPhases.All is derived from ActivePhasePolicy via
        // _byKey lookup — adding "3e" to the policy + a Phase3e display
        // record together is what makes it render. Pin both ends.
        DryWindowPhases.All.Select(p => p.Key).Should().Equal(new[] { "3b", "3e" });
        DryWindowPhases.Phase3e.Key.Should().Be("3e");
        DryWindowPhases.Phase3e.ShortTitle.Should().Contain("3e");
    }

    [Fact]
    public void DryWindowPhases_Bucket_resolves_3e_version_to_Phase3e_record()
    {
        // The renderer's per-phase loop looks each version up in
        // PhaseByVersion → DryWindowPhases.Bucket. A version trained as 3e
        // (training_metadata.Phase = "3e") must bucket into Phase3e so its
        // rows surface under the right card.
        var phaseByVersion = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["v2026-05-03_010101_phase3e"] = "3e",
            ["v2026-04-23_101107"]         = "3b",
        };
        DryWindowPhases.Bucket(phaseByVersion, "v2026-05-03_010101_phase3e")
            .Should().BeSameAs(DryWindowPhases.Phase3e);
        DryWindowPhases.Bucket(phaseByVersion, "v2026-04-23_101107")
            .Should().BeSameAs(DryWindowPhases.Phase3b);
        DryWindowPhases.Bucket(phaseByVersion, "v_unknown").Should().BeNull();
    }
}
