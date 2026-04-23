using Microsoft.ML.Data;

namespace WeatherBlend.Train.DryWindow;

/// <summary>
/// One training row for the dry-window blender. Day-level: one row per
/// (UTC target-date, lead, window-length).
///
/// Feature set is window-aware — <see cref="HasDryWindowGfs"/> etc. encode
/// "does this single model's forecast contain a length-N dry run" and
/// <see cref="AgreementHasDryWindow"/> aggregates the six self-predictions.
/// Other features (totals, wet-hour counts, longest-run) are window-agnostic.
/// Feature ordering is load-bearing and must match
/// <see cref="DryWindowFeatureBuilder.FeatureNames"/>; changing either requires
/// a bump of the feature-schema hash.
/// </summary>
public sealed class DryWindowTrainingRow
{
    public DateTime TargetDateUtc { get; set; }
    public int WindowHours { get; set; }

    // ---- Per-model features (6 × 6 = 36) ------------------------------------
    // Day totals and max hour.
    public float PrecipSumGfs   { get; set; }
    public float PrecipSumEcmwf { get; set; }
    public float PrecipSumIcon  { get; set; }
    public float PrecipSumMf    { get; set; }
    public float PrecipSumUkmo  { get; set; }
    public float PrecipSumGem   { get; set; }

    public float PrecipMaxHourGfs   { get; set; }
    public float PrecipMaxHourEcmwf { get; set; }
    public float PrecipMaxHourIcon  { get; set; }
    public float PrecipMaxHourMf    { get; set; }
    public float PrecipMaxHourUkmo  { get; set; }
    public float PrecipMaxHourGem   { get; set; }

    // Count of hours ≥ 0.1 mm in each model's 24-hour forecast for this day.
    public float WetHourCountGfs   { get; set; }
    public float WetHourCountEcmwf { get; set; }
    public float WetHourCountIcon  { get; set; }
    public float WetHourCountMf    { get; set; }
    public float WetHourCountUkmo  { get; set; }
    public float WetHourCountGem   { get; set; }

    // Longest consecutive run of hours < 0.1 mm inside each model's day.
    public float LongestDryRunGfs   { get; set; }
    public float LongestDryRunEcmwf { get; set; }
    public float LongestDryRunIcon  { get; set; }
    public float LongestDryRunMf    { get; set; }
    public float LongestDryRunUkmo  { get; set; }
    public float LongestDryRunGem   { get; set; }

    // Each model's own "dry window of length N exists" self-prediction
    // (window-specific). Cast to float so LightGBM treats it uniformly.
    public float HasDryWindowGfs   { get; set; }
    public float HasDryWindowEcmwf { get; set; }
    public float HasDryWindowIcon  { get; set; }
    public float HasDryWindowMf    { get; set; }
    public float HasDryWindowUkmo  { get; set; }
    public float HasDryWindowGem   { get; set; }

    // Per-model probability of precipitation (max across the 24 hours).
    public float ProbMaxGfs   { get; set; }
    public float ProbMaxEcmwf { get; set; }
    public float ProbMaxIcon  { get; set; }
    public float ProbMaxMf    { get; set; }
    public float ProbMaxUkmo  { get; set; }
    public float ProbMaxGem   { get; set; }

    // ---- Ensemble summaries (6) --------------------------------------------
    public float PrecipSumMean { get; set; }
    public float PrecipSumStd  { get; set; }
    public float PrecipSumMax  { get; set; }
    /// <summary>Fraction of models (with complete day coverage) whose forecast contains a length-N dry run.</summary>
    public float AgreementHasDryWindow { get; set; }
    public float LongestDryRunMean { get; set; }
    public float WetHourCountMean  { get; set; }

    // ---- Covariates (9) ----------------------------------------------------
    // Aggregates of the ensemble-mean time series across the 24 hours.
    public float RhMean { get; set; }
    public float RhMin  { get; set; }
    public float DewDepressionMax { get; set; }
    public float CloudLowMean  { get; set; }
    public float CloudMidMean  { get; set; }
    public float CloudHighMean { get; set; }
    public float CapeMax    { get; set; }
    public float WindMean   { get; set; }
    public float WindMax    { get; set; }

    // ---- Calendar (2) ------------------------------------------------------
    public float DoySin { get; set; }
    public float DoyCos { get; set; }

    // ---- Shape features (7, Phase 3d) --------------------------------------
    // Derived from the ensemble-mean hourly precip vector for the target day.
    // first_wet_hour uses sentinel 24 when the day is fully dry; last_wet_hour
    // uses -1. Both are monotonic in dryness so LightGBM can split on them.
    public float FirstWetHour { get; set; }
    public float LastWetHour  { get; set; }
    public float LongestForecastDryRunHours { get; set; }
    public float LongestForecastWetRunHours { get; set; }
    public float NRainEvents { get; set; }
    public float MorningPrecipSum   { get; set; }
    public float AfternoonPrecipSum { get; set; }

    // ---- Label -------------------------------------------------------------
    [ColumnName("Label")]
    public bool HasDryWindow { get; set; }

    /// <summary>Raw truth day total in mm — kept for diagnostics, never a feature.</summary>
    public float PrecipMmDay { get; set; }
}
