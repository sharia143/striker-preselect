"""
StatsBomb events -> player-match -> player-season aggregation.

Reconstructs the (previously undocumented) ingestion layer so the pipeline can be
run for ANY competition/season in StatsBomb open data, not just the cached
La Liga 2015/16. Validated to reproduce the cached La Liga player-match and
player-season tables exactly (see validate_ingest.py).

Usage:
    python ingest.py --comp 2 --season 27 --label "premier_league_1516"
    # writes data/player_season_<label>.csv (and optionally --matchcsv)
"""
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FORWARD_POSITIONS = {
    "Center Forward",
    "Left Center Forward",
    "Right Center Forward",
    "Left Wing",
    "Right Wing",
}


# ─── Metric helpers ──────────────────────────────────────────────────────────
def _in_box(loc):
    """Touch in the opposition penalty area (StatsBomb 120x80 pitch, attacking->x=120)."""
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return False
    x, y = loc[0], loc[1]
    return (x >= 102) and (18 <= y <= 62)


def _minutes_for_match(mev):
    """Return {player_id: minutes} for a single match's events."""
    match_end = int(mev["minute"].max())
    subs = mev[mev["type"] == "Substitution"]
    off_min, on_min = {}, {}
    for _, r in subs.iterrows():
        if pd.notna(r.get("player_id")):
            off_min[int(r["player_id"])] = int(r["minute"])
        if pd.notna(r.get("substitution_replacement_id")):
            on_min[int(r["substitution_replacement_id"])] = int(r["minute"])
    starters = set()
    for _, row in mev[mev["type"] == "Starting XI"].iterrows():
        tac = row.get("tactics")
        if isinstance(tac, dict):
            for p in tac.get("lineup", []):
                starters.add(int(p["player"]["id"]))
    out = {}
    pids = mev.dropna(subset=["player_id"])["player_id"].astype(int).unique()
    for pid in pids:
        on = 0 if pid in starters else on_min.get(pid, 0)
        off = off_min.get(pid, match_end)
        out[pid] = off - on
    return out


def aggregate_events(events):
    """
    Full-season events DataFrame -> (player_match forwards-only, player_season).

    `events` must contain all matches of the season so that season-level
    primary_position (mode across all a player's events) is correct.
    """
    # NB: minutes are computed from the FULL events (Starting XI / Half End rows
    # carry no player_id, so they must not be dropped before the minutes pass).
    ev = events.dropna(subset=["player_id"]).copy()
    ev["player_id"] = ev["player_id"].astype(int)

    # ── vectorized per (match, player) metric columns ──
    is_shot = ev["type"] == "Shot"
    is_pen = ev.get("shot_type") == "Penalty"
    is_goal = ev.get("shot_outcome") == "Goal"
    xg = pd.to_numeric(ev.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0)

    ev["_shots"] = is_shot.astype(int)
    ev["_xg"] = np.where(is_shot, xg, 0.0)
    ev["_goals"] = (is_shot & is_goal).astype(int)
    ev["_np_goals"] = (is_shot & is_goal & ~is_pen).astype(int)
    ev["_npxg"] = np.where(is_shot & ~is_pen, xg, 0.0)
    ev["_touches_in_box"] = ev["location"].apply(_in_box).astype(int)
    ev["_key_passes"] = (ev.get("pass_shot_assist") == True).astype(int)  # noqa: E712
    ev["_assists"] = (ev.get("pass_goal_assist") == True).astype(int)     # noqa: E712
    ev["_pressures"] = (ev["type"] == "Pressure").astype(int)

    agg = (
        ev.groupby(["match_id", "player_id"])
        .agg(
            shots=("_shots", "sum"),
            xg=("_xg", "sum"),
            goals=("_goals", "sum"),
            np_goals=("_np_goals", "sum"),
            npxg=("_npxg", "sum"),
            touches_in_box=("_touches_in_box", "sum"),
            key_passes=("_key_passes", "sum"),
            assists=("_assists", "sum"),
            pressures=("_pressures", "sum"),
            player=("player", "first"),
            team=("team", "first"),
        )
        .reset_index()
    )

    # ── minutes per (match, player) — from FULL events (needs Starting XI / Half End) ──
    minutes_rows = []
    for mid, mev in events.groupby("match_id"):
        for pid, mins in _minutes_for_match(mev).items():
            minutes_rows.append((mid, pid, mins))
    minutes = pd.DataFrame(minutes_rows, columns=["match_id", "player_id", "minutes"])
    pm = agg.merge(minutes, on=["match_id", "player_id"], how="left")

    # ── season primary_position = mode of position across all events ──
    pos = ev.dropna(subset=["position"])
    season_pos = pos.groupby("player_id")["position"].agg(
        lambda s: s.value_counts().idxmax()
    )
    pm["primary_position"] = pm["player_id"].map(season_pos)
    pm["np_goals_assists"] = pm["np_goals"] + pm["assists"]

    # ── keep only forwards, order columns to match cached schema ──
    pm = pm[pm["primary_position"].isin(FORWARD_POSITIONS)].copy()
    pm = pm[
        [
            "match_id", "player_id", "minutes", "shots", "xg", "goals", "np_goals",
            "npxg", "touches_in_box", "key_passes", "assists", "pressures",
            "player", "team", "primary_position", "np_goals_assists",
        ]
    ]

    # ── season rollup (mirrors sql/aggregate.sql) ──
    ps = _season_rollup(pm)
    return pm.reset_index(drop=True), ps


def _season_rollup(pm):
    """Player-season per-90 table, matching sql/aggregate.sql."""
    g = pm.groupby("player_id")
    minutes = g["minutes"].sum()
    per90 = 90.0 / minutes

    ps = pd.DataFrame({
        "player_id": minutes.index,
        "player": g["player"].first(),
        "team": g["team"].agg(lambda s: s.value_counts().idxmax()),
        "primary_position": g["primary_position"].first(),
        "matches": g["match_id"].nunique(),
        "total_minutes": minutes,
        "shots_p90": g["shots"].sum() * per90,
        "npxg_p90": g["npxg"].sum() * per90,
        "touches_in_box_p90": g["touches_in_box"].sum() * per90,
        "key_passes_p90": g["key_passes"].sum() * per90,
        "pressures_p90": g["pressures"].sum() * per90,
        "npga_p90": g["np_goals_assists"].sum() * per90,
        "total_np_goals": g["np_goals"].sum(),
        "total_assists": g["assists"].sum(),
        "total_npga": g["np_goals_assists"].sum(),
    }).reset_index(drop=True)
    return ps.sort_values("npga_p90", ascending=False).reset_index(drop=True)


# ─── Live pull ───────────────────────────────────────────────────────────────
def pull_season(comp_id, season_id, verbose=True):
    """Pull all events for a competition/season from StatsBomb open data."""
    from statsbombpy import sb

    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()
    frames = []
    for i, mid in enumerate(match_ids, 1):
        try:
            ev = sb.events(match_id=int(mid))
            ev["match_id"] = int(mid)
            frames.append(ev)
        except Exception as e:  # noqa: BLE001
            print(f"  ! match {mid} failed: {e}")
        if verbose and (i % 25 == 0 or i == len(match_ids)):
            print(f"  pulled {i}/{len(match_ids)} matches", flush=True)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", type=int, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--matchcsv", action="store_true", help="also write player_match csv")
    args = ap.parse_args()

    print(f"Pulling comp={args.comp} season={args.season} ({args.label}) ...", flush=True)
    events = pull_season(args.comp, args.season)
    print(f"  {len(events):,} events across {events['match_id'].nunique()} matches", flush=True)
    pm, ps = aggregate_events(events)
    ps.to_csv(f"data/player_season_{args.label}.csv", index=False, encoding="utf-8")
    print(f"Wrote data/player_season_{args.label}.csv ({len(ps)} forwards)")
    if args.matchcsv:
        pm.to_csv(f"data/player_match_{args.label}.csv", index=False, encoding="utf-8")
        print(f"Wrote data/player_match_{args.label}.csv ({len(pm)} rows)")


if __name__ == "__main__":
    main()
