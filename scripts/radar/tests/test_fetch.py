"""Regression tests for fetch_odim — notably the midnight day-boundary (the 2026-07-01 00:10 failure)."""
import fetch_odim


def test_latest_keys_crosses_midnight(monkeypatch):
    """Just after midnight the newest day holds < n frames; latest_keys must top up from the previous day
    so nowcast always gets >= 2 consecutive frames (otherwise _engine.nowcast crashes on frames[-2])."""
    def fake_list(prefix, delimiter=None, maxkeys=1000):
        if delimiter == "/":                                  # the year/month/day walk
            nxt = {"radar/": "radar/2026/", "radar/2026/": "radar/2026/07/",
                   "radar/2026/07/": "radar/2026/07/01/"}.get(prefix)
            return [], ([nxt] if nxt else [])
        if prefix == "radar/2026/07/01/":                     # new day — only ONE frame so far
            return ["radar/2026/07/01/202607010000_ODIM.h5"], []
        if prefix == "radar/2026/06/30/":                     # previous day — full
            return [f"radar/2026/06/30/2026063023{mm:02d}_ODIM.h5" for mm in (15, 30, 45)], []
        return [], []

    monkeypatch.setattr(fetch_odim, "_list", fake_list)
    keys = fetch_odim.latest_keys(4)
    assert len(keys) == 4                                     # not < 2 -> no IndexError downstream
    assert keys[-1].endswith("202607010000_ODIM.h5")         # newest is the new-day frame
    assert any("2026/06/30" in k for k in keys)              # topped up across the boundary


def test_latest_keys_normal_day(monkeypatch):
    """A full newest day needs no boundary crossing — just the newest n."""
    def fake_list(prefix, delimiter=None, maxkeys=1000):
        if delimiter == "/":
            nxt = {"radar/": "radar/2026/", "radar/2026/": "radar/2026/06/",
                   "radar/2026/06/": "radar/2026/06/30/"}.get(prefix)
            return [], ([nxt] if nxt else [])
        if prefix == "radar/2026/06/30/":
            return [f"radar/2026/06/30/202606301{h:03d}_ODIM.h5" for h in range(0, 900, 15)], []
        return [], []

    monkeypatch.setattr(fetch_odim, "_list", fake_list)
    keys = fetch_odim.latest_keys(4)
    assert len(keys) == 4
    assert all("2026/06/30" in k for k in keys)             # no previous-day fetch needed
