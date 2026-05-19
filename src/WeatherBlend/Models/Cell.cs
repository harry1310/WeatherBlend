namespace WeatherBlend.Models;

/// <summary>
/// One trainable / predictable / verifiable unit of work, fully qualified by
/// location. A <see cref="Cell"/> is the grain that predict, verify and render
/// iterate over once Phase C lands — replacing the ad-hoc nested
/// station → version → lead loops (and the mutable <c>_activeLocation</c>
/// command field) each of those commands grew independently.
///
/// <see cref="Location"/> is the configured location name (e.g.
/// <c>bonehill_rocks</c>); <see cref="Station"/> is the manifest station key —
/// an EA gauge slug for precipitation / dry-window, or the location name itself
/// for temperature / element targets that aren't gauge-partitioned.
///
/// Deliberately stops at <see cref="Lead"/>: a cell is a per-lead model, not a
/// per-valid-time prediction. The 24 hourly valid-times a predict cycle emits
/// per lead stay a loop nested *inside* a cell — folding them in would make a
/// <see cref="Cell"/> an ephemeral 100+-per-cycle value, useless as a stable
/// grouping key for verify and render.
/// </summary>
public sealed record Cell(
    string Location,
    string Station,
    string Target,
    string Phase,
    int Lead);

/// <summary>
/// A <see cref="Cell"/> bound to the concrete model version directory that
/// serves it. The same (location, station, target, phase, lead) cell can be
/// served by more than one version over time; predict / verify resolve the
/// version from the manifest's Active list, render picks the champion.
/// </summary>
public sealed record CellVersion(Cell Cell, string Version);
