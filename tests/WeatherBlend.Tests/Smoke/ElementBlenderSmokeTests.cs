using FluentAssertions;
using WeatherBlend.Commands;
using WeatherBlend.Storage;
using WeatherBlend.Train.Element;
using WeatherBlend.Train.Element.Cloud;
using WeatherBlend.Train.Element.Gust;
using WeatherBlend.Train.Element.Humidity;
using WeatherBlend.Train.Element.Radiation;
using Xunit;
using Xunit.Abstractions;

namespace WeatherBlend.Tests.Smoke;

/// <summary>
/// End-to-end smokes for the single-truth Element blenders trained
/// against ERA5 reanalysis: humidity, shortwave-radiation, cloud-cover,
/// and wind-gust. Wind has its own file (<see cref="ElementWindSmokeTests"/>)
/// because its future state intermingles with the Phase 3 sibling
/// <c>wind_speed_lgb</c>.
///
/// Each test instantiates its blender directly so the typed
/// <see cref="XunitLogger{T}"/> stays generic-correct, then delegates to
/// <see cref="RunElementSmoke"/> for the shared fixture + assertions.
/// Per-lead LightGBM fit dominates the wall (~3-5s per blender on the
/// 30-day fixture).
/// </summary>
public class ElementBlenderSmokeTests
{
    private readonly ITestOutputHelper _output;

    public ElementBlenderSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task Humidity_train_writes_bundle_and_promotes_manifest()
    {
        using var ctx = await BuildContextAsync();
        await RunElementSmoke(ctx,
            target: "humidity",
            modelDirName: "humidity",
            blender: new HumidityBlender(
                new XunitLogger<HumidityBlender>(_output), ctx.Scope.Config));
    }

    [Fact]
    public async Task ShortwaveRadiation_train_writes_bundle_and_promotes_manifest()
    {
        using var ctx = await BuildContextAsync();
        await RunElementSmoke(ctx,
            target: "shortwave-radiation",
            modelDirName: "shortwave_radiation",
            blender: new RadiationBlender(
                new XunitLogger<RadiationBlender>(_output), ctx.Scope.Config));
    }

    [Fact]
    public async Task CloudCover_train_writes_bundle_and_promotes_manifest()
    {
        using var ctx = await BuildContextAsync();
        await RunElementSmoke(ctx,
            target: "cloud-cover",
            modelDirName: "cloud_cover",
            blender: new CloudBlender(
                new XunitLogger<CloudBlender>(_output), ctx.Scope.Config));
    }

    [Fact]
    public async Task WindGust_train_writes_suffixed_bundle_and_promotes_manifest()
    {
        using var ctx = await BuildContextAsync();
        // wind_gust has PhaseTag="wind_gust_lgb" != ModelDirName="wind_gust",
        // so the harness's auto-suffix logic kicks in: dir is
        // v{ts}_wind_gust_lgb/. Distinct shape from the other three
        // (PhaseTag == ModelDirName → unsuffixed).
        await RunElementSmoke(ctx,
            target: "wind-gust",
            modelDirName: "wind_gust",
            blender: new WindGustBlender(
                new XunitLogger<WindGustBlender>(_output), ctx.Scope.Config),
            expectedBundleSuffix: "_wind_gust_lgb");
    }

    // -----------------------------------------------------------------
    // Shared fixture context — built once per test (not shared across
    // tests since SmokeScope owns a per-test tempdir).
    // -----------------------------------------------------------------

    private sealed record TestContext(EnvScope Env, SmokeScope Scope) : IDisposable
    {
        public void Dispose()
        {
            Scope.Dispose();
            Env.Dispose();
        }
    }

    private async Task<TestContext> BuildContextAsync()
    {
        const string locationName = "bonehill_rocks";
        // 2025-02-01 + 30 days lands fully inside the UKMO-clean window
        // (≥2024-09-01) so any blender that has UKMO as required (cloud)
        // or optional doesn't end up with all-NaN rows.
        var trainStart = new DateTime(2025, 2, 1, 0, 0, 0, DateTimeKind.Utc);
        const int trainDays = 30;

        var env = new EnvScope();
        var scope = new SmokeScope(
            locationName,
            rainfallStations: new[] { ("smoke-bonehill", "Bellever Dartmoor") });

        await SmokeFixtures.WriteForecastTreeAsync(
            scope.ForecastsPath, locationName, trainStart, trainDays,
            runTimeSource: "offset_day");
        await SmokeFixtures.WriteEra5TruthAsync(
            scope.Era5Path, locationName, trainStart, trainDays);

        return new TestContext(env, scope);
    }

    private async Task RunElementSmoke(
        TestContext ctx,
        string target,
        string modelDirName,
        IElementBlender blender,
        string? expectedBundleSuffix = null)
    {
        const string locationName = "bonehill_rocks";
        var scope = ctx.Scope;

        var element = new ElementTrainCommand(
            new XunitLogger<ElementTrainCommand>(_output),
            new[] { blender });

        // Other train arms stubbed — cross-target dispatch only needs
        // them as non-null constructor args for target=element.
        var metadata = new ModelMetadataRepository(
            new XunitLogger<ModelMetadataRepository>(_output), scope.Config);
        var precipConformal = new PrecipConformalFitCommand(
            new XunitLogger<PrecipConformalFitCommand>(_output), scope.Config, metadata);
        var precip = new PrecipTrainCommand(
            new XunitLogger<PrecipTrainCommand>(_output), scope.Config, precipConformal);
        var dryWindowConformal = new DryWindowConformalFitCommand(
            new XunitLogger<DryWindowConformalFitCommand>(_output), scope.Config, metadata);
        var dryWindow = new DryWindowTrainCommand(
            new XunitLogger<DryWindowTrainCommand>(_output), scope.Config, metadata, dryWindowConformal);
        var train = new TrainCommand(
            new XunitLogger<TrainCommand>(_output), scope.Config,
            dryWindow, element, precip);

        var rc = await train.RunAsync(
            target: target, lead: "all",
            station: null, window: null, featureSet: "lean",
            ct: default);
        rc.Should().Be(0, $"{target} train should succeed on the synthetic fixture");

        // Per-location bundle dir. These three blenders all have
        // PhaseTag == ModelDirName so the harness emits an unsuffixed
        // v{ts} directory.
        var locationDir = Path.Combine(scope.ModelsPath, modelDirName, locationName);
        Directory.Exists(locationDir).Should().BeTrue(
            $"{target} train should create data/models/{modelDirName}/{locationName}/");
        var dirs = Directory.EnumerateDirectories(locationDir)
            .Select(d => Path.GetFileName(d)!)
            .ToList();
        dirs.Should().HaveCount(1, $"{target} train should mint exactly one bundle dir");
        var bundleName = dirs[0];
        bundleName.Should().StartWith("v",
            "version dir convention is v{yyyy-MM-dd_HHmmss}[_phaseTag]");
        if (expectedBundleSuffix is not null)
        {
            bundleName.Should().EndWith(expectedBundleSuffix,
                $"{target} bundle dir should be suffixed with PhaseTag when it differs from ModelDirName");
        }
        var bundleDir = Path.Combine(locationDir, bundleName);

        // Per-lead LightGBM artefacts + canonical sidecars.
        Directory.EnumerateFiles(bundleDir, "lead_*.zip").Should().NotBeEmpty(
            $"{target} bundle must contain LightGBM lead_*.zip artefacts");
        foreach (var sidecar in new[]
        {
            "training_metadata.json", "feature_schema.json",
            "feature_importance.json", "training_summary.json",
        })
        {
            File.Exists(Path.Combine(bundleDir, sidecar)).Should().BeTrue(
                $"{target} bundle should contain {sidecar}");
        }

        // Manifest promotion.
        var manifestPath = Path.Combine(scope.ModelsPath, modelDirName, "MANIFEST.json");
        File.Exists(manifestPath).Should().BeTrue(
            $"{target} MANIFEST.json must be promoted");
        var manifestJson = await File.ReadAllTextAsync(manifestPath);
        manifestJson.Should().Contain(bundleName,
            $"manifest Active list should carry the trained {target} version");
        manifestJson.Should().Contain(locationName,
            "manifest should be keyed under the trained location");
    }
}
