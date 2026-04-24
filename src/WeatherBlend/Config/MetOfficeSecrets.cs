namespace WeatherBlend.Config;

public static class MetOfficeSecrets
{
    public static string? TryLoad(string envVar, string filePath)
    {
        var fromEnv = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(fromEnv)) return fromEnv.Trim();

        if (File.Exists(filePath))
        {
            var fromFile = File.ReadAllText(filePath).Trim();
            if (!string.IsNullOrWhiteSpace(fromFile)) return fromFile;
        }
        return null;
    }

    public static string LoadOrThrow(string envVar, string filePath, string product)
    {
        return TryLoad(envVar, filePath)
            ?? throw new InvalidOperationException(
                $"Met Office {product} API key not found. Set ${envVar} or place the key in {filePath}.");
    }
}
