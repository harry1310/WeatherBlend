namespace WeatherBlend.Config;

public sealed class AppConfig
{
    public LocationConfig Location { get; set; } = new();
    public List<ModelConfig> Models { get; set; } = new();
    public VariablesConfig Variables { get; set; } = new();
    public List<int> LeadHours { get; set; } = new();
    public int ForecastDays { get; set; } = 7;
    public StorageConfig Storage { get; set; } = new();
    public HttpConfig Http { get; set; } = new();
    public MetOfficeConfig MetOffice { get; set; } = new();
}

public sealed class VariablesConfig
{
    /// <summary>Variables pulled per-model from Open-Meteo (Live + Previous Runs).</summary>
    public List<string> Forecast { get; set; } = new();

    /// <summary>Variables pulled from the ERA5 archive endpoint as training truth.</summary>
    public List<string> Era5 { get; set; } = new();
}

public sealed class LocationConfig
{
    public string Name { get; set; } = "";
    public string DisplayName { get; set; } = "";
    public double Latitude { get; set; }
    public double Longitude { get; set; }
    public double ElevationMeters { get; set; }
    public MetarConfig Metar { get; set; } = new();
    public RainfallConfig Rainfall { get; set; } = new();
}

public sealed class MetarConfig
{
    public string Primary { get; set; } = "";
    public string Fallback { get; set; } = "";
}

public sealed class RainfallConfig
{
    public List<RainfallStationConfig> Stations { get; set; } = new();
}

public sealed class RainfallStationConfig
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
}

public sealed class ModelConfig
{
    public string Id { get; set; } = "";
    public string DisplayName { get; set; } = "";
}

public sealed class StorageConfig
{
    public string ForecastsPath { get; set; } = "data/forecasts";
    public string ObservationsPath { get; set; } = "data/truth/metar";
    public string Era5Path { get; set; } = "data/truth/era5";
    public string RainfallPath { get; set; } = "data/truth/rainfall";
    public string PredictionsPath { get; set; } = "data/predictions";
    public string ReportsPath { get; set; } = "data/reports";
    public string MetOfficeObsPath { get; set; } = "data/truth/met_office_obs";
}

public sealed class MetOfficeConfig
{
    public bool Enabled { get; set; } = true;
    public string SpotModelTag { get; set; } = "met_office_spot";
    public string SpotKeyEnvVar { get; set; } = "MET_OFFICE_SPOT_API_KEY";
    // Relative to the process cwd (typically the repo root when `dotnet run` is
    // invoked there). Keys live one level above the repo, outside version control.
    // Env var wins over file (see MetOfficeSecrets.TryLoad), and CI uses env vars
    // exclusively — so this relative path only matters for local dev convenience.
    public string SpotKeyFile { get; set; } = "../MetOfficeSpotKey.txt";
    public string ObsKeyEnvVar { get; set; } = "MET_OFFICE_OBS_API_KEY";
    public string ObsKeyFile { get; set; } = "../MetOfficeObsKey.txt";

    /// <summary>
    /// Geohash of the Land Observations location nearest our site. The DataHub
    /// Land Observations API is geohash-addressed (not station-id) — call
    /// <c>/nearest?lat=...&amp;lon=...</c> once, pin the result here, and reuse.
    /// Left null until a bootstrap run fills it in; leaving it null simply skips
    /// obs collection without blocking the rest of the cycle.
    /// </summary>
    public string? ObsGeohash { get; set; }

    /// <summary>Human-readable area the geohash resolves to (e.g. "Devon"). Informational.</summary>
    public string? ObsArea { get; set; }
}

public sealed class HttpConfig
{
    public string UserAgent { get; set; } = "WeatherBlend-PoC/0.1";
    public int TimeoutSeconds { get; set; } = 60;
    public int RetryCount { get; set; } = 3;
}
