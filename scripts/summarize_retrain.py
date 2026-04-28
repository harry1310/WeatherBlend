"""Read every training_metadata.json under data/models/ and produce a single
markdown summary of (target, phase, station, lead) → blend vs best-single test.

Picks the latest artefact per (composite, phase). Filters to those trained
since 2026-04-28T22:00 — the post-met_office-deletion run.
"""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CUTOFF = datetime(2026, 4, 28, 22, 0, 0)
ROOT = Path("data/models")

def load(path: Path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def fmt_delta(blend, best):
    if best is None or best <= 0: return "—"
    pct = (blend - best) / best * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"

def short_metric(target):
    if target == "temperature":           return "MAE °C"
    if target in ("precipitation", "dry_window"): return "Brier"
    if target == "wind":                  return "MAE m/s"
    if target == "humidity":              return "MAE %"
    if target == "shortwave_radiation":   return "MAE W/m²"
    if target == "cloud_cover":           return "MAE %"
    return "score"

def collect_artefacts():
    """Yield (composite, phase, version_dir) for every training_metadata.json."""
    for meta_path in ROOT.rglob("training_metadata.json"):
        try:
            meta = load(meta_path)
        except Exception:
            continue
        ver_dir = meta_path.parent
        # composite = path from data/models/ to (version_dir.parent)
        rel = ver_dir.relative_to(ROOT).parent.as_posix()
        yield rel, meta.get("Phase", ""), ver_dir, meta

def latest_per_family(arts):
    """Keep the latest artefact per (composite, phase) by TrainedAtUtc."""
    latest = {}
    for composite, phase, ver_dir, meta in arts:
        key = (composite, phase)
        ts = datetime.fromisoformat(meta.get("TrainedAtUtc", "1970-01-01T00:00:00").rstrip("Z").split(".")[0])
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, composite, phase, ver_dir, meta)
    return [v[1:] for v in latest.values()]

def render_table(rows, metric_label):
    """rows: list of (lead, blend, best_name, best_test, best_val, n)."""
    out = []
    out.append(f"| Lead | Blend ({metric_label}) | Best single | Best on test | Δ blend-vs-best | N |")
    out.append("|---|---:|---|---:|---:|---:|")
    for lead, blend, best_name, best_test, best_val, n in rows:
        delta = fmt_delta(blend, best_test) if best_test and best_test > 0 else fmt_delta(blend, best_val)
        ref = best_test if best_test and best_test > 0 else best_val
        ref_label = f"{ref:.4f}" if ref else "—"
        if not (best_test and best_test > 0):
            ref_label += " (val)"
        out.append(f"| +{lead}h | **{blend:.4f}** | {best_name} | {ref_label} | {delta} | {n} |")
    return "\n".join(out)

def main():
    arts_all = [a for a in collect_artefacts()
            if datetime.fromisoformat(a[3].get("TrainedAtUtc","1970-01-01T00:00:00").rstrip("Z").split(".")[0]) >= CUTOFF]
    arts = latest_per_family(arts_all)

    if not arts:
        print("No artefacts since cutoff. Nothing to report.")
        return

    out = []
    out.append("# Retrain summary — 2026-04-29 (post-MetOffice JOIN-bug fix)\n")
    out.append(f"_Generated {datetime.utcnow():%Y-%m-%d %H:%M:%S}Z. Cutoff: artefacts trained ≥ {CUTOFF:%Y-%m-%d %H:%M}Z._\n")
    out.append("**What changed:** Met Office Python-collected partitions deleted from `data/forecasts/`. Forecast tree now uniformly TIMESTAMP (was promoted to TIMESTAMPTZ before, causing forecast↔truth JOINs to mis-align by 1–2h during BST). All blenders retrained on properly-aligned pairs. The Models page can now compare blend vs best-single on the **same test slice** (not val vs test).\n")
    out.append("**Δ convention:** negative = blend wins, positive = best-single wins. (Element blenders display + as blend-wins in their own log; here normalised.)\n")

    by_target = {}
    for composite, phase, ver_dir, meta in arts:
        target = composite.split("/")[0]
        by_target.setdefault(target, []).append((composite, phase, ver_dir, meta))

    target_order = ["temperature", "precipitation", "dry_window", "wind", "humidity", "shortwave_radiation", "cloud_cover"]
    for target in target_order:
        if target not in by_target: continue
        out.append(f"\n## {target}\n")
        rows_for_target = sorted(by_target[target], key=lambda x: (x[0], x[1]))
        for composite, phase, ver_dir, meta in rows_for_target:
            ver = ver_dir.name
            trained = meta.get("TrainedAtUtc", "")[:19]
            out.append(f"### `{composite}` · phase **{phase}**  ·  `{ver}`")
            out.append(f"_Trained {trained}Z_\n")
            metric = short_metric(target)
            table_rows = []
            for lead_str, s in sorted(meta["PerLead"].items(), key=lambda x: int(x[0])):
                lead = int(lead_str)
                blend = s.get("BlendTestMae")
                best_name = s.get("BestSingle", "")
                best_test = s.get("BestSingleTestMae", 0)
                best_val = s.get("BestSingleValMae", 0)
                n = s.get("TestRows", 0)
                table_rows.append((lead, blend, best_name, best_test, best_val, n))
            out.append(render_table(table_rows, metric))
            out.append("")

    report_path = Path("data/reports/retrain_summary_2026-04-29.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Written: {report_path}")

if __name__ == "__main__": main()
