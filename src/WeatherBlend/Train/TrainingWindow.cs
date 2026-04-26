namespace WeatherBlend.Train;

/// <summary>
/// Shared training-window constants for blenders that include UKMO as a feature.
///
/// Open-Meteo's Previous Runs archive only began populating UKMO weather fields
/// around 2024-08-15 (audit 2026-04-26: UKMO rows exist for 2024-01..2024-08 but
/// every weather field is NULL — the row schema is filled, the values are not).
/// A 6-model blender trained on the full backfill therefore sees a 7-month all-
/// NULL UKMO block at the start, and learns a dual regime ("UKMO=NaN for 7
/// months, then UKMO=valid forever after"). Restricted bake-off 2026-04-26 showed
/// the resulting model loses to a 5-model blender on the held-out test set even
/// though UKMO is fully present there.
///
/// Restricting to <see cref="UkmoCleanWindowStart"/> drops the 7-month polluted
/// block and trains the 6-model blender on a clean ~20-month window where UKMO
/// is consistently present (≥99% non-null). Bake-off on this window shows
/// 6-model wins or ties at every (target, station, lead) we tested.
///
/// Blenders that intentionally drop UKMO entirely (currently: shortwave-radiation,
/// where UKMO is 92–100% null in live collects) do NOT need this filter — there
/// is no UKMO column to poison their training, and the pre-2024-09 data is
/// otherwise normal.
/// </summary>
public static class TrainingWindow
{
    /// <summary>2024-09-01 UTC. First full month after Open-Meteo's UKMO archive
    /// became consistently populated. Used as a SQL TIMESTAMP literal.</summary>
    public const string UkmoCleanWindowStart = "2024-09-01";
}
