/**
 * Tiny cron parser scoped to the expressions this scheduler-worker
 * actually uses. NOT a general-purpose cron implementation — handles
 * only the shapes in WORKFLOW_FOR_CRON (no `*\/N`, no ranges, no `?`,
 * no `L`, no `#`):
 *
 *   MIN HOUR DOM MONTH DOW
 *
 * Each field is one of:
 *   - `*`              wildcard
 *   - integer          single value
 *   - comma-list       e.g. `3,9,15,21`
 *   - DOW name list    e.g. `MON,THU` (only valid in the DOW field)
 *
 * Adding richer syntax (slash, range, name-prefixed month) requires
 * extending parseField + the matcher. We deliberately keep the surface
 * tight because every expression here is hand-curated and reviewed —
 * we'd rather throw at parse time on an unrecognised shape than guess.
 *
 * Times are UTC throughout. Cloudflare cron triggers fire in UTC and
 * JavaScript Date methods are used in their UTC variant.
 */

/** Parsed numeric values for one cron field, plus the `*` wildcard
 * sentinel. Empty array == wildcard. Non-empty == the exact allowed
 * set of values for that field. */
type Field = readonly number[];

interface ParsedCron {
  minute: Field;
  hour: Field;
  dom: Field;
  month: Field;
  /** Day-of-week, 0-6, Sunday = 0. JS getUTCDay() returns 0-6 with
   * Sunday=0; matches our representation exactly. We canonicalise 7→0
   * (some cron dialects treat 7 as Sunday) at parse time so the matcher
   * doesn't have to. */
  dow: Field;
}

const DOW_NAMES: Record<string, number> = {
  SUN: 0, MON: 1, TUE: 2, WED: 3, THU: 4, FRI: 5, SAT: 6,
};

function parseField(raw: string, name: string, min: number, max: number, dow = false): Field {
  if (raw === "*") return [];
  const parts = raw.split(",");
  const out: number[] = [];
  for (const part of parts) {
    const trimmed = part.trim();
    let n: number;
    if (dow && /^[A-Z]+$/.test(trimmed)) {
      if (!(trimmed in DOW_NAMES))
        throw new Error(`cron: unknown DOW name '${trimmed}' in field ${name}`);
      n = DOW_NAMES[trimmed];
    } else {
      n = parseInt(trimmed, 10);
      if (!Number.isInteger(n))
        throw new Error(`cron: non-integer '${trimmed}' in field ${name}`);
    }
    if (dow && n === 7) n = 0;  // canonicalise 7 → 0 for Sunday
    if (n < min || n > max)
      throw new Error(`cron: value ${n} out of range [${min},${max}] in field ${name}`);
    out.push(n);
  }
  return out;
}

export function parseCron(expr: string): ParsedCron {
  const fields = expr.trim().split(/\s+/);
  if (fields.length !== 5)
    throw new Error(`cron: expected 5 fields, got ${fields.length} in '${expr}'`);
  const [m, h, dom, month, dow] = fields;
  return {
    minute: parseField(m,     "minute", 0,  59),
    hour:   parseField(h,     "hour",   0,  23),
    dom:    parseField(dom,   "dom",    1,  31),
    month:  parseField(month, "month",  1,  12),
    dow:    parseField(dow,   "dow",    0,  7, /*dow*/ true),
  };
}

/** True iff every cron field matches the given UTC date. Wildcard
 * fields (empty array) always match. The classic POSIX "OR between
 * DOM and DOW when both are restricted" rule does NOT apply here — all
 * our expressions use `*` for DOM, so the OR rule never fires. If we
 * ever ship a `DOM,DOW` expression with both restricted, this matcher
 * would AND them, which is intentional for clarity over POSIX legacy. */
function matchesAt(parsed: ParsedCron, d: Date): boolean {
  const matchField = (f: Field, v: number) => f.length === 0 || f.includes(v);
  return (
    matchField(parsed.minute, d.getUTCMinutes()) &&
    matchField(parsed.hour,   d.getUTCHours())   &&
    matchField(parsed.dom,    d.getUTCDate())    &&
    matchField(parsed.month,  d.getUTCMonth() + 1) &&
    matchField(parsed.dow,    d.getUTCDay())
  );
}

/**
 * The most recent minute in `(fromMs, beforeMs]` at which `expr`
 * should have fired, or `null` if it shouldn't have fired in that
 * window. `beforeMs` is inclusive; `fromMs` is exclusive. Both in
 * Unix milliseconds.
 *
 * Searches backwards minute-by-minute. The watchdog calls this with a
 * 1-hour window (60 minute checks per cron, 4 crons = 240 evaluations
 * per tick — trivial). Pass a longer window only if the watchdog
 * itself missed a tick and you want broader recovery.
 *
 * @example
 *   // Did "5 3,9,15,21 * * *" fire between 14:00Z and 16:00Z on 2026-05-27?
 *   const before = Date.UTC(2026, 4, 27, 16, 0, 0);   // 16:00Z
 *   const from   = Date.UTC(2026, 4, 27, 14, 0, 0);   // 14:00Z
 *   const t = lastFireInWindow("5 3,9,15,21 * * *", from, before);
 *   // t === Date.UTC(2026, 4, 27, 15, 5, 0)   // 15:05Z
 */
export function lastFireInWindow(expr: string, fromMs: number, beforeMs: number): number | null {
  if (beforeMs <= fromMs) return null;
  const parsed = parseCron(expr);
  // Snap beforeMs down to the start of its minute — cron fires at the
  // start of a minute, so anything within that minute counts as a fire
  // at the minute boundary.
  const startMs = Math.floor(beforeMs / 60_000) * 60_000;
  for (let t = startMs; t > fromMs; t -= 60_000) {
    const d = new Date(t);
    if (matchesAt(parsed, d)) return t;
  }
  return null;
}
