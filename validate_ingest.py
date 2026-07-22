"""Validate ingest.aggregate_events reproduces the cached La Liga tables exactly."""
import warnings
import numpy as np
import pandas as pd
from ingest import aggregate_events

warnings.filterwarnings("ignore")

events = pd.read_pickle("data/events_laliga_1516.pkl")
pm_calc, ps_calc = aggregate_events(events)

pm_gt = pd.read_csv("data/player_match_stats.csv")
ps_gt = pd.read_csv("data/player_season_stats.csv")

print(f"player_match  rows: calc={len(pm_calc)}  gt={len(pm_gt)}")
print(f"player_season rows: calc={len(ps_calc)} gt={len(ps_gt)}")

# ── player_match exact comparison ──
key = ["match_id", "player_id"]
m = pm_gt.merge(pm_calc, on=key, suffixes=("_gt", "_calc"))
print(f"joined player_match rows: {len(m)} (unmatched gt={len(pm_gt)-len(m)})")

num_cols = ["minutes", "shots", "xg", "goals", "np_goals", "npxg",
            "touches_in_box", "key_passes", "assists", "pressures", "np_goals_assists"]
print("\n=== player_match column agreement ===")
all_ok = True
for c in num_cols:
    a, b = m[f"{c}_gt"].values, m[f"{c}_calc"].values
    ok = np.allclose(a, b, atol=1e-4, equal_nan=True)
    nbad = int((~np.isclose(a, b, atol=1e-4, equal_nan=True)).sum())
    all_ok &= ok
    print(f"  {c:18s} match={ok}  mismatches={nbad}")
# categorical
pos_ok = (m["primary_position_gt"] == m["primary_position_calc"]).all()
print(f"  {'primary_position':18s} match={pos_ok}")
all_ok &= pos_ok

# ── player_season exact comparison ──
ms = ps_gt.merge(ps_calc, on="player_id", suffixes=("_gt", "_calc"))
print(f"\njoined player_season rows: {len(ms)} (gt={len(ps_gt)})")
print("=== player_season column agreement ===")
scols = ["matches", "total_minutes", "shots_p90", "npxg_p90", "touches_in_box_p90",
         "key_passes_p90", "pressures_p90", "npga_p90", "total_np_goals",
         "total_assists", "total_npga"]
for c in scols:
    a, b = ms[f"{c}_gt"].values, ms[f"{c}_calc"].values
    ok = np.allclose(a, b, atol=1e-3, equal_nan=True)
    nbad = int((~np.isclose(a, b, atol=1e-3, equal_nan=True)).sum())
    all_ok &= ok
    print(f"  {c:20s} match={ok}  mismatches={nbad}")

print("\n" + ("=" * 50))
print("FULL-SEASON VALIDATION:", "PASS" if all_ok else "MISMATCH")
print("=" * 50)

if not all_ok:
    # show a few mismatching player_match rows for the first bad column
    for c in num_cols:
        bad = m[~np.isclose(m[f"{c}_gt"], m[f"{c}_calc"], atol=1e-4, equal_nan=True)]
        if len(bad):
            print(f"\nSample mismatches for {c}:")
            print(bad[["match_id", "player_x", f"{c}_gt", f"{c}_calc"]].head(8).to_string(index=False)
                  if "player_x" in bad else bad[["match_id", "player_id", f"{c}_gt", f"{c}_calc"]].head(8).to_string(index=False))
            break
