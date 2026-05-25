using System.Text.Json;

namespace WeatherBlend.Train.Oro;

/// <summary>
/// Per-station / per-location static orographic features, generated offline
/// by <c>scripts/OrographicFeatures/build_static_orographic.py</c> from the
/// CGIAR SRTM v4 90m DEM and read at training/predict time.
///
/// Slugs match <see cref="Models.StationSlug.WithEaPrefix"/> for EA gauges
/// (<c>ea_bellever_dartmoor</c>) and bare config slugs for locations
/// (<c>bonehill_rocks</c>). One JSON per slug under
/// <c>data/static/orographic/{slug}.json</c>.
///
/// v1 scope (3c-oro bake-off): only the fields the rich-oro feature builder
/// reads. The script also writes <c>relief_10/25km</c>, the full 8-sector
/// upwind tables at 10 km, and the raw <c>site_elevation_m</c> — those are
/// kept on disk for future v2 features without re-running the build, just
/// not loaded here.
/// </summary>
public sealed class OroStaticFeatures
{
    public required string Slug { get; init; }

    /// <summary>Site elevation minus mean elevation in ±4.5 km window (≈9 km NWP cell).</summary>
    public double ElevationVsCellM { get; init; }

    /// <summary>Max − min elevation within 5 km radius.</summary>
    public double Relief5kmM { get; init; }

    /// <summary>Mean |elev − site| within 5 km — local roughness.</summary>
    public double TerrainRuggedness5kmM { get; init; }

    /// <summary>8-sector mean (elev − site) over the upwind 5 km arc, keyed by
    /// compass sector ("N", "NE", "E", "SE", "S", "SW", "W", "NW"). Positive =
    /// terrain rises upwind, negative = terrain drops upwind.</summary>
    public required IReadOnlyDictionary<string, double> UpwindGain5km { get; init; }

    /// <summary>Local terrain gradient east-rise per metre east (m/m).</summary>
    public double TerrainGradientDx { get; init; }

    /// <summary>Local terrain gradient north-rise per metre north (m/m).</summary>
    public double TerrainGradientDy { get; init; }

    // -----------------------------------------------------------------------
    // v2 DEM aggregations (added 2026-05-25). Optional in the JSON for
    // back-compat with v1 bundles that don't have them — defaults to 0.
    // -----------------------------------------------------------------------

    /// <summary>Topographic Position Index at 200 m, 1 km, 5 km, 25 km radii
    /// — site_elev minus mean elev in the radius. Positive = local high point;
    /// negative = local hollow. Keys: "200", "1000", "5000", "25000".</summary>
    public IReadOnlyDictionary<string, double> TpiByRadiusM { get; init; }
        = new Dictionary<string, double>();

    /// <summary>Per-sector lee-obstruction in 10 km arc: max(elev along
    /// upwind arc) − site_elev. Positive = ridge crest sits between site and
    /// upwind boundary → potential rain-shadow. Sectors as in
    /// <see cref="Sectors"/>.</summary>
    public IReadOnlyDictionary<string, double> LeeObstruction10km { get; init; }
        = new Dictionary<string, double>();

    /// <summary>Mean per-pixel slope magnitude (m/m) within 5 km radius.
    /// Regional steepness, distinct from the single-point site gradient.</summary>
    public double MeanSlope5km { get; init; }

    /// <summary>Magnitude of slope-weighted unit-aspect vector mean over 5 km.
    /// Near 1 = terrain mostly faces one direction (coherent slope); near 0 =
    /// omnidirectional (peak / basin / complex).</summary>
    public double AspectDominance5km { get; init; }

    public static readonly string[] TpiRadiusKeys = { "200", "1000", "5000", "25000" };

    // -----------------------------------------------------------------------
    // v3 atmospheric climatology (added 2026-05-25). Per-(sector × month)
    // lookup table built offline by build_atm_climatology.py from GFS
    // pressure-level history pulled via Open-Meteo archive-api. Optional —
    // missing on older bundles, returns NaN-filled record for unseen bins.
    // -----------------------------------------------------------------------

    /// <summary>Climatology values for one (wind sector × month) bin.
    /// All values may be NaN if the bin had fewer than the minimum sample count.</summary>
    public sealed record ClimatologyBin(
        double Lapse850_500,
        double Lapse700_500,
        double Q850,
        double Wind500Speed,
        double Shear850_500,
        double ThicknessProxy,
        int NSamples);

    public const int ClimatologyFeatureCount = 6;
    public static readonly string[] ClimatologyFeatureNames =
    {
        "climo_lapse_850_500", "climo_lapse_700_500", "climo_q_850",
        "climo_wind_500_speed", "climo_shear_850_500", "climo_thickness_proxy",
    };

    /// <summary>Per-(sector × month-as-string) climatology lookup. Empty
    /// dictionary if the JSON has no climatology block (older v1/v2 bundles).</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, ClimatologyBin>> ClimatologyBySectorMonth { get; init; }
        = new Dictionary<string, IReadOnlyDictionary<string, ClimatologyBin>>();

    /// <summary>Lookup climatology by (wind direction from atan2(sin, cos))
    /// + month. Returns a NaN-filled bin if the bucket has no data — caller
    /// should pack the NaN values as features and let LightGBM's native NaN
    /// handling take care of it.</summary>
    public ClimatologyBin ClimatologyAt(double windDirRadFrom, int month)
    {
        if (ClimatologyBySectorMonth.Count == 0)
            return new ClimatologyBin(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, 0);

        // Sector lookup matches UpwindGainAt — nearest 45° bin.
        var deg = windDirRadFrom * (180.0 / Math.PI);
        deg = ((deg % 360.0) + 360.0) % 360.0;
        var sectorIdx = (int)Math.Floor((deg + 22.5) / 45.0) % 8;
        var sector = Sectors[sectorIdx];

        if (!ClimatologyBySectorMonth.TryGetValue(sector, out var monthMap))
            return new ClimatologyBin(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, 0);
        if (!monthMap.TryGetValue(month.ToString(), out var bin))
            return new ClimatologyBin(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, 0);
        return bin;
    }

    /// <summary>
    /// Map an NWP-mean wind-FROM direction (radians, atan2(sin_mean, cos_mean))
    /// to one of the 8 compass sectors, nearest-bin lookup. Returns the
    /// <see cref="UpwindGain5km"/> value for that sector.
    ///
    /// Convention: <paramref name="windDirRadFrom"/> is the meteorological
    /// "direction wind is FROM", measured clockwise from north (so 0 = N,
    /// π/2 = E, π = S, 3π/2 = W). Matches Open-Meteo's WindDirection10m
    /// field after deg→rad conversion. A NaN input returns 0 — treat as
    /// "no orographic context available, neutral prior".
    /// </summary>
    public double UpwindGainAt(double windDirRadFrom)
    {
        if (double.IsNaN(windDirRadFrom)) return 0.0;
        var deg = windDirRadFrom * (180.0 / Math.PI);
        // Wrap to [0, 360).
        deg = ((deg % 360.0) + 360.0) % 360.0;
        // Nearest 45° bin (±22.5°). Add 22.5 then floor by 45 = nearest sector index.
        var idx = (int)Math.Floor((deg + 22.5) / 45.0) % 8;
        return UpwindGain5km[Sectors[idx]];
    }

    public static readonly string[] Sectors = { "N", "NE", "E", "SE", "S", "SW", "W", "NW" };

    // ---------------------------------------------------------------------
    // Loader
    // ---------------------------------------------------------------------

    /// <summary>
    /// Load all per-slug JSONs under <paramref name="staticRoot"/> (typically
    /// <c>data/static/orographic</c>) into a slug → record map. Skips files that
    /// don't have the v1 fields we need (forward-compat: the build script
    /// might emit extra slugs we ignore here).
    /// </summary>
    public static IReadOnlyDictionary<string, OroStaticFeatures> LoadAll(string staticRoot)
    {
        var map = new Dictionary<string, OroStaticFeatures>(StringComparer.OrdinalIgnoreCase);
        if (!Directory.Exists(staticRoot)) return map;
        foreach (var path in Directory.EnumerateFiles(staticRoot, "*.json"))
        {
            var slug = Path.GetFileNameWithoutExtension(path);
            try
            {
                var rec = LoadOne(path, slug);
                map[slug] = rec;
            }
            catch (Exception ex) when (ex is JsonException or KeyNotFoundException)
            {
                // Skip silently — a partially-written or future-schema file
                // shouldn't crash training. Callers that need a specific slug
                // and find it missing will report the absence themselves.
            }
        }
        return map;
    }

    /// <summary>Load one slug's record from disk. Throws if the JSON is malformed
    /// or missing required v1 fields.</summary>
    public static OroStaticFeatures LoadOne(string path, string slug)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var upwind = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        var upwindEl = root.GetProperty("upwind_gain_5km");
        foreach (var s in Sectors)
            upwind[s] = upwindEl.GetProperty(s).GetDouble();

        // v2 fields — optional for back-compat with v1 bundles.
        var tpi = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (var k in TpiRadiusKeys)
        {
            if (root.TryGetProperty($"tpi_{k}m", out var v)) tpi[k] = v.GetDouble();
        }
        var lee = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        if (root.TryGetProperty("lee_obstruction_10km", out var leeEl))
        {
            foreach (var s in Sectors)
                if (leeEl.TryGetProperty(s, out var sv)) lee[s] = sv.GetDouble();
        }
        var meanSlope = root.TryGetProperty("mean_slope_5km", out var ms) ? ms.GetDouble() : 0.0;
        var aspectDom = root.TryGetProperty("aspect_dominance_5km", out var ad) ? ad.GetDouble() : 0.0;

        // v3 climatology lookup table (optional).
        var climo = new Dictionary<string, IReadOnlyDictionary<string, ClimatologyBin>>(StringComparer.OrdinalIgnoreCase);
        if (root.TryGetProperty("climatology_by_sector_month", out var climoEl))
        {
            foreach (var sectorProp in climoEl.EnumerateObject())
            {
                var monthMap = new Dictionary<string, ClimatologyBin>(StringComparer.Ordinal);
                foreach (var monthProp in sectorProp.Value.EnumerateObject())
                {
                    var bin = monthProp.Value;
                    double get(string n) => bin.TryGetProperty(n, out var v) ? v.GetDouble() : double.NaN;
                    monthMap[monthProp.Name] = new ClimatologyBin(
                        Lapse850_500: get("lapse_850_500"),
                        Lapse700_500: get("lapse_700_500"),
                        Q850:         get("q_850"),
                        Wind500Speed: get("wind_500_speed"),
                        Shear850_500: get("shear_850_500"),
                        ThicknessProxy: get("thickness_proxy"),
                        NSamples:     bin.TryGetProperty("n_samples", out var n) ? n.GetInt32() : 0);
                }
                climo[sectorProp.Name] = monthMap;
            }
        }

        return new OroStaticFeatures
        {
            Slug = slug,
            ElevationVsCellM = root.GetProperty("elevation_vs_cell_m").GetDouble(),
            Relief5kmM = root.GetProperty("relief_5km_m").GetDouble(),
            TerrainRuggedness5kmM = root.GetProperty("terrain_ruggedness_5km_m").GetDouble(),
            UpwindGain5km = upwind,
            TerrainGradientDx = root.GetProperty("terrain_gradient_dx").GetDouble(),
            TerrainGradientDy = root.GetProperty("terrain_gradient_dy").GetDouble(),
            TpiByRadiusM = tpi,
            LeeObstruction10km = lee,
            MeanSlope5km = meanSlope,
            AspectDominance5km = aspectDom,
            ClimatologyBySectorMonth = climo,
        };
    }
}
