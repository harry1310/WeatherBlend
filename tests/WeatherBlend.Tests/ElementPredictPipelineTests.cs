using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using FluentAssertions;
using WeatherBlend.Train.Element.Wind;
using Xunit;

namespace WeatherBlend.Tests;

/// <summary>
/// Predict-pipeline output-row composition invariants.
///
/// These tests exist because of a real audit-display bug found 2026-04-26:
/// when wind was migrated from Pattern 1 (4-model, no UKMO) to "5-model with
/// UKMO restored", the training side updated correctly, but the predict
/// pipeline's output composition kept hardcoded `ModelUkmo = null` /
/// `RunTimeUkmo = null` lines that pre-dated the migration. The blend value
/// was correct (UKMO was being fed to the model via the WindRow), but the
/// audit fields lied.
///
/// To prevent this class of bug recurring, wind's per-model output mapping
/// is now extracted into a pure function (`MapToOutputModelFields`) and
/// these tests pin its behaviour. Any future ModelsForLead migration will
/// fail one of these tests rather than silently leave provenance stale.
/// </summary>
public class ElementPredictPipelineTests
{
    // ---------------------------------------------------------------
    // Wind: pure-function mapping from accessor-slot arrays to output fields
    // ---------------------------------------------------------------

    [Fact]
    public void Wind_MapToOutputModelFields_populates_all_accessor_slots()
    {
        // Distinct values per slot so any swap shows up. Accessor order is
        // (gfs, ecmwf, icon, ukmo, gem) — the 5-slot list, no MF.
        var speeds = new float[] { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f };
        var runTimes = new System.DateTime?[]
        {
            new System.DateTime(2026, 4, 26,  0, 0, 0, System.DateTimeKind.Utc),
            new System.DateTime(2026, 4, 26,  6, 0, 0, System.DateTimeKind.Utc),
            new System.DateTime(2026, 4, 26, 12, 0, 0, System.DateTimeKind.Utc),
            new System.DateTime(2026, 4, 26, 18, 0, 0, System.DateTimeKind.Utc),
            new System.DateTime(2026, 4, 27,  0, 0, 0, System.DateTimeKind.Utc),
        };

        var f = WindPredictPipeline.MapToOutputModelFields(speeds, runTimes);

        f.ModelGfs.Should().BeApproximately(1.1, 1e-6);
        f.ModelEcmwf.Should().BeApproximately(2.2, 1e-6);
        f.ModelIcon.Should().BeApproximately(3.3, 1e-6);
        // UKMO is the bug-bait: pre-fix this was hardcoded null.
        f.ModelUkmo.Should().BeApproximately(4.4, 1e-6,
            "wind ModelUkmo must come from accessor slot 3 — see 2026-04-26 audit-display bug");
        f.ModelGem.Should().BeApproximately(5.5, 1e-6);
        // MF is genuinely never sourced for wind (no MF wind data).
        f.ModelMf.Should().BeNull();

        f.RunTimeGfs.Should().Be(runTimes[0]);
        f.RunTimeEcmwf.Should().Be(runTimes[1]);
        f.RunTimeIcon.Should().Be(runTimes[2]);
        f.RunTimeUkmo.Should().Be(runTimes[3], "wind RunTimeUkmo must come from accessor slot 3");
        f.RunTimeGem.Should().Be(runTimes[4]);
        f.RunTimeMf.Should().BeNull();
    }

    [Fact]
    public void Wind_MapToOutputModelFields_translates_NaN_to_null()
    {
        // Live forecast tree may have a NaN slot when a particular model didn't
        // ship for that valid time. The output should surface that as null
        // (parquet-friendly) rather than as NaN.
        var speeds = new float[] { 1f, float.NaN, 3f, float.NaN, 5f };
        var runTimes = new System.DateTime?[] { null, null, null, null, null };

        var f = WindPredictPipeline.MapToOutputModelFields(speeds, runTimes);

        f.ModelGfs.Should().NotBeNull();
        f.ModelEcmwf.Should().BeNull();
        f.ModelIcon.Should().NotBeNull();
        f.ModelUkmo.Should().BeNull();
        f.ModelGem.Should().NotBeNull();
    }

    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(0)]
    public void Wind_MapToOutputModelFields_throws_on_wrong_arity(int wrongCount)
    {
        var speeds = Enumerable.Repeat(0f, wrongCount).ToArray();
        var runTimes = Enumerable.Repeat<System.DateTime?>(null, wrongCount).ToArray();
        var act = () => WindPredictPipeline.MapToOutputModelFields(speeds, runTimes);
        act.Should().Throw<System.ArgumentException>();
    }

    [Fact]
    public void Wind_MapToOutputModelFields_accessor_order_pinned()
    {
        // Belt-and-braces: if someone reorders WindFeatureBuilder.ModelAccessors
        // without updating MapToOutputModelFields, this catches it.
        var ids = WindFeatureBuilder.ModelAccessors.Select(a => a.ModelId).ToArray();
        ids.Should().Equal(
            "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "ukmo_seamless", "gem_seamless");
    }

    // ---------------------------------------------------------------
    // Sister-pipeline source-text invariants
    //
    // Cloud, humidity, radiation pipelines all have 6 accessor slots that map
    // 1:1 to the six output fields, so they don't have wind's 5→6 projection
    // problem. These tests still pin the structural property: each output
    // ModelXxx field must reference its slot's source array (e.g. p.Cc[N],
    // p.Rhs[N], p.Sw[N]) rather than be hardcoded null.
    //
    // String-based source inspection — brittle by nature, but the bug class
    // it catches (legacy hardcoded-null after a model-set migration) is
    // exactly what bit wind, so the brittleness is worth it.
    // ---------------------------------------------------------------

    private static readonly string SrcRoot = ResolveSrcRoot();

    private static string ResolveSrcRoot()
    {
        var dir = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "src", "WeatherBlend", "Train", "Element");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new System.InvalidOperationException("Could not locate src/WeatherBlend/Train/Element from test base dir");
    }

    [Theory]
    [InlineData("Cloud/CloudPredictPipeline.cs", "p.Cc")]
    [InlineData("Humidity/HumidityPredictPipeline.cs", "p.Rhs")]
    [InlineData("Radiation/RadiationPredictPipeline.cs", "p.Sw")]
    public void SisterPipelines_PerModelOutputFields_reference_source_array_not_hardcoded_null(
        string relativePath, string expectedSourceArrayPrefix)
    {
        var source = File.ReadAllText(Path.Combine(SrcRoot, relativePath));
        // Find the ElementPredictionRow construction blocks. Each must have
        // ModelGfs/Ecmwf/Icon/Mf/Ukmo/Gem assignments.
        foreach (var modelField in new[] { "ModelGfs", "ModelEcmwf", "ModelIcon", "ModelMf", "ModelUkmo", "ModelGem" })
        {
            // Find every assignment of this field. Match `ModelGfs   = ...,`
            // through to comma or newline.
            var pattern = new Regex($@"\b{modelField}\s*=\s*([^,\r\n]+)", RegexOptions.Compiled);
            var matches = pattern.Matches(source);
            matches.Count.Should().BeGreaterThan(0,
                $"{relativePath} should assign {modelField}");

            // For 6-accessor pipelines (cloud/humidity/radiation), every per-model
            // field should source from the per-model array, NOT be hardcoded null.
            foreach (Match m in matches)
            {
                var rhs = m.Groups[1].Value.Trim();
                rhs.Should().NotBe("null",
                    $"{relativePath}: {modelField} = null suggests a stale Pattern-1-style hardcoding (same bug class as wind 2026-04-26). " +
                    $"Should reference {expectedSourceArrayPrefix}[N] (or a Nz(...) wrapper) so provenance follows the actual model set.");
            }
        }
    }
}
