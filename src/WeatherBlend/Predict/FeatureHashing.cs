using System.Security.Cryptography;

namespace WeatherBlend.Predict;

/// <summary>
/// Deterministic SHA-256 hex of a feature vector. Used for the
/// <c>FeatureVectorHash</c> provenance field on every prediction row — lets us
/// prove post-hoc that a recorded prediction corresponds to a specific set of
/// feature inputs. Shared across the temperature and precipitation predict
/// commands; feature order is the caller's responsibility.
/// </summary>
public static class FeatureHashing
{
    /// <summary>Hash an ordered span of floats. Little-endian byte representation
    /// (platform default on all our targets); output is lowercase hex SHA-256.</summary>
    public static string HashFloats(ReadOnlySpan<float> floats)
    {
        Span<byte> buf = stackalloc byte[sizeof(float) * 64];  // plenty for ≤64 features
        if (floats.Length * sizeof(float) > buf.Length)
            return HashFloatsHeap(floats);

        var bytes = buf[..(floats.Length * sizeof(float))];
        for (int i = 0; i < floats.Length; i++)
            BitConverter.TryWriteBytes(bytes.Slice(i * sizeof(float), sizeof(float)), floats[i]);

        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(bytes, hash);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string HashFloatsHeap(ReadOnlySpan<float> floats)
    {
        var bytes = new byte[floats.Length * sizeof(float)];
        for (int i = 0; i < floats.Length; i++)
            BitConverter.TryWriteBytes(bytes.AsSpan(i * sizeof(float), sizeof(float)), floats[i]);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}
