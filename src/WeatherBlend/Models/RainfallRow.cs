using System.Diagnostics.CodeAnalysis;

namespace WeatherBlend.Models;

/// <summary>
/// One 15-minute rainfall total from the Environment Agency Hydrology API. Each row
/// is the rainfall accumulated over the 15 minutes ending at <see cref="ObservedTimeUtc"/>
/// — EA timestamps are interval-end, which matches the forecast-row convention.
///
/// QC fields are recorded verbatim rather than pre-filtered: downstream verification
/// can decide whether "Good" only, "Good+Estimated", or everything is acceptable
/// for a given slice. Dropping rows at ingest time would make that choice forever.
/// </summary>
public sealed class RainfallRow
{
    [SetsRequiredMembers]
    public RainfallRow()
    {
        LocationName = "";
        StationId = "";
        StationName = "";
        Quality = "";
        Completeness = "";
    }

    /// <summary>The cluster-location name this station is associated with (e.g. "Bonehill Rocks").
    /// Multiple stations per location support spatial-crosscheck.</summary>
    public required string LocationName { get; init; }

    /// <summary>EA measure-prefix identifier. For versioned stations this is
    /// "&lt;guid&gt;_&lt;wiskiID&gt;" (e.g. Bellever = "723a8fc4-…_363307"); for
    /// others it is the bare GUID. Matches the id used in config.yaml.</summary>
    public required string StationId { get; init; }

    /// <summary>Human-readable station name, e.g. "Bellever Dartmoor".</summary>
    public required string StationName { get; init; }

    /// <summary>End of the 15-minute accumulation interval, UTC.</summary>
    public required DateTime ObservedTimeUtc { get; init; }

    /// <summary>Total rainfall in mm for the preceding 15 minutes. Nullable because
    /// EA occasionally returns a reading with no value (quality=Missing).</summary>
    public double? Value15MinMm { get; init; }

    /// <summary>One of {Good, Estimated, Suspect, Unchecked, Missing}. Required —
    /// always present on the `-qualified` measure series.</summary>
    public required string Quality { get; init; }

    /// <summary>{Complete, Incomplete}.</summary>
    public required string Completeness { get; init; }
}
