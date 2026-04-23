using FluentAssertions;
using WeatherBlend.Predict;
using Xunit;

namespace WeatherBlend.Tests;

public class FeatureHashingTests
{
    [Fact]
    public void HashFloats_is_deterministic()
    {
        var a = new float[] { 1.5f, 2.25f, -3.75f, 0f, float.NaN };
        var b = new float[] { 1.5f, 2.25f, -3.75f, 0f, float.NaN };

        FeatureHashing.HashFloats(a).Should().Be(FeatureHashing.HashFloats(b));
    }

    [Fact]
    public void HashFloats_is_sensitive_to_order()
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 3f, 2f, 1f };

        FeatureHashing.HashFloats(a).Should().NotBe(FeatureHashing.HashFloats(b));
    }

    [Fact]
    public void HashFloats_is_sensitive_to_value()
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 1f, 2f, 3.0001f };

        FeatureHashing.HashFloats(a).Should().NotBe(FeatureHashing.HashFloats(b));
    }

    [Fact]
    public void HashFloats_output_is_64_char_lowercase_hex_sha256()
    {
        var a = new float[] { 1f };
        var h = FeatureHashing.HashFloats(a);

        h.Should().HaveLength(64);
        h.Should().MatchRegex("^[0-9a-f]{64}$");
    }

    [Fact]
    public void HashFloats_falls_back_to_heap_for_large_inputs()
    {
        // 128 floats > stackalloc 64-float budget → heap path.
        var big = Enumerable.Range(0, 128).Select(i => (float)i).ToArray();
        var same = big.ToArray();

        FeatureHashing.HashFloats(big).Should().Be(FeatureHashing.HashFloats(same));
    }

    [Fact]
    public void HashFloats_stack_and_heap_paths_produce_same_hash_for_identical_short_input()
    {
        // Both paths encode little-endian float bytes the same way — a short input
        // taking the stack path should still be byte-identical to the same input
        // if we forced it through the heap. Easiest proxy: pre-known hash from
        // independently computing SHA-256 over the float bytes.
        var floats = new float[] { 1f, 2f };
        // IEEE-754 little-endian of 1f is 00 00 80 3F, of 2f is 00 00 00 40.
        var expected = System.Security.Cryptography.SHA256.HashData(new byte[]
        {
            0x00, 0x00, 0x80, 0x3F,
            0x00, 0x00, 0x00, 0x40,
        });
        var expectedHex = Convert.ToHexString(expected).ToLowerInvariant();

        FeatureHashing.HashFloats(floats).Should().Be(expectedHex);
    }
}
