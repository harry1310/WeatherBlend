namespace WeatherBlend.Config;

/// <summary>
/// Loads the WeatherLink (api.weatherlink.com v2) API key + secret. Mirrors
/// <see cref="MetOfficeSecrets"/> but WeatherLink needs TWO values (an
/// <c>api-key</c> query parameter and an <c>X-Api-Secret</c> header), so the
/// file fallback is a small two-line <c>key=…</c> / <c>secret=…</c> properties
/// file rather than a single bare token.
///
/// Resolution per value: the environment variable wins; the file is the
/// fallback (same precedence as <see cref="MetOfficeSecrets"/> and the R2
/// creds). CI sets env vars exclusively — the file path is local-dev only.
/// </summary>
public static class WeatherLinkSecrets
{
    public readonly record struct Credentials(string Key, string Secret);

    /// <summary>
    /// Returns the (key, secret) pair, or null if EITHER is missing. A partial
    /// credential is useless to the API, so callers treat null as "no creds —
    /// skip" exactly like the Met Office obs path.
    /// </summary>
    public static Credentials? TryLoad(string keyEnvVar, string secretEnvVar, string keyFile)
    {
        var key = Environment.GetEnvironmentVariable(keyEnvVar)?.Trim();
        var secret = Environment.GetEnvironmentVariable(secretEnvVar)?.Trim();

        // Only touch the file for whichever value the env vars didn't supply.
        if ((string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(secret)) && File.Exists(keyFile))
        {
            var (fileKey, fileSecret) = ParseKeyFile(File.ReadAllLines(keyFile));
            if (string.IsNullOrWhiteSpace(key)) key = fileKey;
            if (string.IsNullOrWhiteSpace(secret)) secret = fileSecret;
        }

        if (string.IsNullOrWhiteSpace(key) || string.IsNullOrWhiteSpace(secret)) return null;
        return new Credentials(key, secret);
    }

    /// <summary>
    /// Parse a two-line <c>key=…</c> / <c>secret=…</c> file. Order-independent,
    /// tolerant of blank lines and surrounding whitespace; ignores unknown keys.
    /// </summary>
    internal static (string? Key, string? Secret) ParseKeyFile(IEnumerable<string> lines)
    {
        string? key = null, secret = null;
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0) continue;
            var eq = line.IndexOf('=');
            if (eq <= 0) continue;
            var name = line[..eq].Trim().ToLowerInvariant();
            var value = line[(eq + 1)..].Trim();
            if (name == "key") key = value;
            else if (name == "secret") secret = value;
        }
        return (key, secret);
    }
}
