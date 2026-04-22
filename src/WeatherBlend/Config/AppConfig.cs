namespace WeatherBlend.Config;

public sealed class AppConfig
{
    public LocationConfig Location { get; set; } = new();
    public List<ModelConfig> Models { get; set; } = new();
    public List<string> Variables { get; set; } = new();
    public List<int> LeadHours { get; set; } = new();
    public int ForecastDays { get; set; } = 7;
    public StorageConfig Storage { get; set; } = new();
    public HttpConfig Http { get; set; } = new();
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
}

public sealed class HttpConfig
{
    public string UserAgent { get; set; } = "WeatherBlend-PoC/0.1";
    public int TimeoutSeconds { get; set; } = 60;
    public int RetryCount { get; set; } = 3;
}
